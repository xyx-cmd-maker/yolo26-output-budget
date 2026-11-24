"""
yolo26-detection / src/eval/budget_scan.py
==========================================
逐图输出预算扫描：把 NMS-free 检测器当作**固定预算选择器**来刻画。

研究问题（docs/research-framework.md §1）：
  H1 逐图最优预算 B* 的分布是否高度右偏（多数图 B≪300 即饱和，少数密集图需 B≫300）？
  H2 固定 B 下的召回损失是否与该图 GT 目标数近似成比例？
  H3 双头分歧能否预测逐图退化（在控制分类置信度后）？

★ 必须走 predict 路径，不得用 model.val()：
   实测 ultralytics 8.4.177 `models/yolo/detect/val.py:64-93` 的 `_check_max_det()`
   在 `max_det == DEFAULT_CFG.max_det`（300）时会**自动抬升**上限到观测最大值 →
   用 val() 做预算实验会**测不到天花板，得到假阴性**。

★ 预算语义：`max_det` 同时决定
   ① one-to-one 头的 top-k（`predictor.py:351` set_head_attr(max_det=max(args.max_det,300))）
   ② NMS 后的截断（one-to-many 路径）
   我们对两条路径都测，并显式声明用的是哪条。

产出（每个 (权重, 切分, 头, B) 一行）：
    runs/budget/<exp>/<split>_<head>_B<B>.json   —— 逐图记录 + 汇总
    results/tables/budget-<exp>-<split>.json     —— 聚合曲线

用法：
    python -m src.eval.budget_scan --weights runs/train/<run>/weights/best.pt \\
        --data data/processed/visdrone/visdrone.yaml --split val \\
        --budgets 1 5 10 25 50 100 300 1000 --head both --exp r1_budget
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BUDGETS_DEFAULT = [1, 5, 10, 25, 50, 100, 300, 1000]


# ---------------------------------------------------------------- 基础工具
def _require_finite(arr, what: str) -> np.ndarray:
    """校验数值数组里没有会被静默吞掉的 NaN/Inf（P1-F 新增）。

    为什么必须显式抛错而不是过滤：旧 `_iou_pairwise` 用
    `np.where(u > 0, inter / u, 0.0)`，而 `NaN > 0` 为 False，于是 **NaN 坐标会
    静默变成 IoU = 0**，在结果里表现为"这个预测不重叠"，与"真的不重叠"不可区分。
    坐标全部来自 ultralytics 输出与 VisDrone 标签，正常输入恒为有限值，
    因此这里抛 `ValueError`：异常必须暴露，不能被 0 掩盖。
    """
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"{what} 含 {bad} 个非有限值（NaN/Inf）")
    return a


def _iou_pairwise(a, b) -> np.ndarray:
    """逐对 IoU（xyxy，绝对像素坐标）。零面积框返回 0.0，不抛错。

    坐标语义：`load_split` 已把 GT 的归一化 `cx,cy,w,h` 还原成**原图像素** xyxy，
    预测框来自 ultralytics 的 `boxes.xyxy`（同为原图像素），两侧同一坐标空间，
    因此无需缩放即可直接算 IoU。

    零面积框（`w<=0` 或 `h<=0`）与任何框的并集为 0：此处按 **IoU = 0** 处理，
    保持历史取值语义不变（不静默过滤、不改变坐标定义）。
    """
    a = _require_finite(a, "iou.a").reshape(-1, 4)
    b = _require_finite(b, "iou.b").reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    u = aa[:, None] + ab[None, :] - inter
    # 分母先替零再相除：与 `inter / u` 在 u>0 时逐位相同，但不会产生 0 除警告
    return np.where(u > 0, inter / np.where(u > 0, u, 1.0), 0.0)


def _labels_dir_for(img_dir: Path) -> Path:
    """从右往左替换最后一个 `images` 组件（全局 replace 会改到祖先目录，见 P5）。"""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def load_split(data_yaml: str, split: str) -> tuple[list[Path], dict[str, np.ndarray]]:
    import yaml
    from PIL import Image

    p = Path(data_yaml)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    img_dir = Path(cfg[split])
    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()
    lbl_dir = _labels_dir_for(img_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() in exts)
    gt: dict[str, np.ndarray] = {}
    for img in imgs:
        lb = lbl_dir / (img.stem + ".txt")
        boxes = []
        if lb.exists():
            with Image.open(img) as im:
                W, H = im.size
            for ln in lb.read_text(encoding="utf-8").strip().splitlines():
                pr = ln.split()
                if len(pr) < 5:
                    continue
                _, cx, cy, w, h = (float(v) for v in pr[:5])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        gt[img.name] = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return imgs, gt


def canonical_order(pred_conf) -> np.ndarray:
    """置信度降序排列，**同分时按输入原始索引升序**（P1-B 的唯一定义处）。

    旧写法是 `np.argsort(-pred_conf)`（默认 quicksort，非稳定）：置信度并列时
    排列顺序取决于排序算法内部实现，同一份输入在不同 numpy 版本 / 平台上可能
    给出不同的 tie 顺序，进而改变贪心匹配的 TP 归属与前缀计数。
    这里用 `np.lexsort`：主键 `-conf`，次键 `arange(n)`（**输入原始索引**），
    因此排列对任意输入都唯一确定，且 `order` 本身就是"原始索引 → 排序位置"的映射。

    注意：次键用的是**输入数组的位置**而非模型内部张量序号。调用方的输入数组
    顺序就是评测看到的那份 `Results.boxes` 顺序（ultralytics 已按置信度降序返回），
    因此这个位置在该次评测内是稳定的原始索引。
    """
    c = _require_finite(pred_conf, "pred_conf").reshape(-1)
    return np.lexsort((np.arange(c.size), -c))


def validated_tp_count(tp, n_gt: int, image: str = "") -> int:
    """把 TP 布尔向量归约成计数，并校验 `0 <= TP <= n_gt`（P1-F 新增）。"""
    s = int(np.asarray(tp, dtype=bool).sum())
    if s < 0 or s > int(n_gt):
        raise ValueError(f"TP 计数越界（{s} 不在 [0, {int(n_gt)}] 内）{(' 图像=' + image) if image else ''}")
    return s


def validate_tp_prefix(prefix, n_gt: int, n_pred: int, image: str = "") -> None:
    """校验 `tp_prefix` 三条不变量：长度不超预测数、单调不减、末端不超 GT 数。"""
    p = [int(v) for v in prefix]
    tag = f"图像={image}" if image else "图像=?"
    if len(p) > int(n_pred):
        raise ValueError(f"tp_prefix 长度 {len(p)} 超过预测数 {int(n_pred)}（{tag}）")
    if any(p[i] > p[i + 1] for i in range(len(p) - 1)):
        raise ValueError(f"tp_prefix 非单调不减（{tag}）")
    if p and p[-1] > int(n_gt):
        raise ValueError(f"tp_prefix 末端 {p[-1]} 超过 GT 数 {int(n_gt)}（{tag}）")


def macro_recall(tp, n_gt) -> float:
    """macro 召回：**只在 `n_gt > 0` 的图上取均值**（P1-G）。

    旧写法 `np.where(m, tp/np.maximum(n,1), np.nan)` + `np.nanmean` 数值上与此相同，
    但把"有效集合"藏在 NaN 传播里：任何**非预期**的 NaN 都会被 `nanmean` 一并跳过，
    异常因此不可见。这里把有效集合写成显式掩码，并对非有限输入抛错。

    全部图像都是零 GT 时返回 NaN（与旧 `nanmean` 行为一致），由调用方决定是否使用。
    """
    t = _require_finite(tp, "tp").reshape(-1)
    n = _require_finite(n_gt, "n_gt").reshape(-1)
    if t.shape != n.shape:
        raise ValueError(f"tp 与 n_gt 长度不一致：{t.shape[0]} vs {n.shape[0]}")
    m = n > 0
    if not m.any():
        return float("nan")
    rec = t[m] / n[m]
    if not np.all(np.isfinite(rec)):
        raise ValueError("有效图（n_gt>0）上出现非有限的 recall")
    return float(rec.mean())


def greedy_match(pred_box, pred_conf, pred_cls, gt_box, iou_thr=0.5) -> np.ndarray:
    """confidence-ordered greedy matching **with unmatched-GT fallback**。

    语义（P1 主指标 = corrected class-agnostic localization recall, IoU=0.5）：

    1. 预测按置信度降序；**同分时按输入原始索引升序**（见 `canonical_order`，确定性）。
    2. 每个预测只在**尚未匹配**的 GT 中挑 IoU 最大者。
    3. IoU 同分时取 **GT 原始索引最小**者（`np.argmax` 返回首个最大值）。
    4. 该最大 IoU 仍 `< iou_thr` → 该预测记 FP。
    5. 一个 GT 至多匹配一个预测。
    6. **类别不参与**：`pred_cls` 仅为签名兼容而保留，本函数完全不使用它 → class-agnostic。

    ★ 这不是全局最大匹配（Hungarian），也不是最优指派：它按置信度顺序逐框决定，
      一旦某个 GT 被高置信度预测占用就不再归还。

    ★ 返回数组与**输入位置**对齐（不是排序后位置），因此调用方对返回值做前缀求和
      仍然对应该前缀位置上的预测。

    ★ 前缀性质：由于第 k 步的判定只依赖前 k-1 步的 `used`，对"取前 k 个预测"的
      子集单独调用本函数，其结果等于整集调用结果的 `cumsum(tp)[k-1]`。
      `run_head` 的 `tp_prefix` 与逐预算重算因此必须逐点相等（测试 F 固定此性质）。
    """
    box = _require_finite(pred_box, "pred_box").reshape(-1, 4)
    conf = _require_finite(pred_conf, "pred_conf").reshape(-1)
    gt = _require_finite(gt_box, "gt_box").reshape(-1, 4)
    if box.shape[0] != conf.shape[0]:
        raise ValueError(f"pred_box 与 pred_conf 长度不一致：{box.shape[0]} vs {conf.shape[0]}")
    thr = float(iou_thr)
    if not (0.0 <= thr <= 1.0):
        raise ValueError(f"iou_thr 必须在 [0,1] 内，收到 {iou_thr}")

    n = box.shape[0]
    tp = np.zeros(n, dtype=bool)
    if n == 0 or gt.shape[0] == 0:
        return tp

    order = canonical_order(conf)
    used = np.zeros(gt.shape[0], dtype=bool)
    for i in order:
        ious = _iou_pairwise(box[i : i + 1], gt)[0]
        # ★★ 核心修正（P1-A）：只能在**未匹配的 GT** 中取最大。
        #    旧实现是 `j = argmax(ious)` 取**全部** GT 的最大者，随后才检查 `used[j]`
        #    → 一旦该 GT 已被占用就直接丢弃本预测，**不回退到次优的未匹配 GT**，
        #    系统性少算 TP。此处把已匹配的 GT 置 -1（IoU 恒 >= 0，故不可能被选中）。
        cand = np.where(used, -1.0, ious)
        j = int(np.argmax(cand))
        if (not used[j]) and cand[j] >= thr:
            used[j] = True
            tp[i] = True
    return tp


# ---------------------------------------------------------------- 主流程
def run_head(
    weights: str,
    data: str,
    split: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    budgets: list[int],
    nms_free: bool,
    max_budget_run: int,
    tag: str,
    limit: int | None = None,
) -> dict:
    """对某一头跑一遍推理（用最大的 B），然后在 numpy 里对每个 B 做截断重算召回。

    ★ 只推理一次：因为预算截断 = 按置信度取前 B 个，可以在已排序的输出上离线模拟。
      但 **one-to-one 头的 top-k 是架构内的** —— 若 B < 实际 top-k，必须真正用该 B 推理。
      所以这里取 `max_budget_run = max(budgets)` 作为推理时的 max_det，
      再离线截断模拟更小的 B。这会**高估**小 B 的表现吗？不会——因为
      one-to-one 头的 top-k 就是 max_det，用大 B 推理得到的是"更多候选"，
      取前 B 个正是小 B 时该头会输出的东西（同一排序）。**该等价性由 B0 校验实验验证。**
    """
    from ultralytics import YOLO

    m = YOLO(weights)
    names = {int(k): v for k, v in (m.names or {}).items()}
    imgs, gt = load_split(data, split)
    if limit is not None and limit > 0:
        # ★ 抽样上限：本机系统 RAM 仅 15.7 GB，且 R1 训练同时占内存。
        #   一次性把全部 548/1610 张图交给 predict 会 MemoryError（实测踩到，见 P4）。
        #   `--limit` 用于管线验证与小规模试跑；正式结果必须不设 limit。
        imgs = imgs[:limit]
        keep = {p.name for p in imgs}
        gt = {k: v for k, v in gt.items() if k in keep}
        print(f"  [{tag}] ⚠️ 抽样模式：仅用前 {len(imgs)} 张图（--limit {limit}）")
    print(f"  [{tag}] 推理 {len(imgs)} 图（max_det={max_budget_run}, nms_free={nms_free}）…", flush=True)

    kw = {"nms": False} if nms_free else {}
    t0 = time.time()
    res = m.predict(
        source=[str(p) for p in imgs], imgsz=imgsz, conf=conf, iou=iou,
        max_det=max_budget_run, device=device, stream=True, verbose=False, **kw,
    )

    per_image: list[dict] = []
    for r in res:
        b = r.boxes
        if b is None or len(b) == 0:
            box = np.zeros((0, 4)); cf = np.zeros(0); cl = np.zeros(0, int)
        else:
            box = b.xyxy.cpu().numpy().astype(float)
            cf = b.conf.cpu().numpy().astype(float)
            cl = b.cls.cpu().numpy().astype(int)
        per_image.append({"name": Path(r.path).name, "box": box, "conf": cf, "cls": cl})
    infer_s = time.time() - t0

    # 逐图 GT 数
    n_gt_per = np.array([len(gt[p["name"]]) for p in per_image], dtype=int)

    rows = []
    # ★★ 逐图"前缀 TP 计数"：在**推理时的 max_det（= max_budget_run）**下，
    #    把该图全部预测按置信度降序排列，做一次贪心匹配，
    #    再取**前缀和** → `tp_prefix[k]` = 取前 k+1 个预测时的真阳性数。
    #
    #    为什么必需（第 36 轮新增，判据 v2 的基础）：
    #      `greedy_match` 对"按置信度取前 q 个"的输出给出唯一确定的 TP 数
    #      —— 因为贪心匹配本身按置信度降序进行，前缀上的匹配结果与"先在全集上匹配再取前 q"
    #      在**计数**上一致（同一 GT 不会被两个不同前缀各算一次）。
    #      因此有了 `tp_prefix`，就能**离线**算出**任意配额 q** 下的 `TPᵢ(q)`，
    #      无需重跑推理。这是判据 v2（`docs/prereg-decision-rule-v2-absolute.md`）
    #      比较"同一总花费下三种分配"的前提。
    #    只存一次（不随 B 重复）；上限 600 个点控制体积。
    _tp_prefix: dict[str, list[int]] = {}
    for pi in per_image:
        box, cf, cl = pi["box"], pi["conf"], pi["cls"]
        if len(box):
            # ★ P1-B：唯一确定性排序（同分按原始索引），与逐预算路径共用同一函数
            order = canonical_order(cf)
            tp_full = greedy_match(box[order], cf[order], cl[order], gt[pi["name"]])
            cum = np.cumsum(tp_full.astype(int))[:600]
            validate_tp_prefix(cum, len(gt[pi["name"]]), len(box), pi["name"])
            _tp_prefix[Path(pi["name"]).stem] = [int(v) for v in cum]
        else:
            _tp_prefix[Path(pi["name"]).stem] = []

    # ★ 逐图落盘：每个 B 下每图的 n_gt / n_pred / n_tp。
    #   这是 bstar_significance.py 的**唯一输入**——没有它就无法做配对 bootstrap，
    #   而配对检验是 C1 核心命题（排序反转）能否成立的唯一判据（见 P0-1b 红队 R2）。
    per_image_records: dict[str, dict] = {}
    for B in budgets:
        recs = []
        for pi in per_image:
            box, cf, cl = pi["box"], pi["conf"], pi["cls"]
            if len(box) > B:
                # ★ P1-B：与 `tp_prefix` 同一排序函数 → 前缀与逐预算结果逐点一致
                order = canonical_order(cf)[:B]
                box, cf, cl = box[order], cf[order], cl[order]
            # 该图的 GT：**全部类别一起匹配，类别不参与判定**（class-agnostic）。
            # 旧注释声称"类别一致性在 _is_tp 内处理"，但 `_is_tp` 并不存在，
            # 且本实现从未使用 `pred_cls` —— 该注释已按实际语义改正（P1-C）。
            g = gt[pi["name"]]
            tp = greedy_match(box, cf, cl, g) if len(box) else np.zeros(0, bool)
            n_tp = validated_tp_count(tp, len(g), pi["name"])
            n_gt = len(g)
            nm = Path(pi["name"]).stem
            entry = per_image_records.setdefault(
                nm, {
                    "name": pi["name"],
                    "n_gt": int(n_gt),
                    # ★★ `n_pred_at_run`：**在推理时的 max_det（= max_budget_run）下**该图的预测数。
                    #   第 36 轮新增。为什么必需：
                    #   判据 v2（docs/prereg-decision-rule-v2-absolute.md）要比较"同一总花费"下的
                    #   多种分配，而**真实能花出去的钱受检测器输出上限约束**：
                    #       spendᵢ = min(配额ᵢ, pᵢ)，其中 pᵢ = n_pred_at_run
                    #   没有这个字段就无法算真实花费，会把"给不出那么多预测的图"当成花满了。
                    #   注意它与 `n_pred_at[str(B)]` 不同：后者是**截断到 B 之后**的数。
                    "n_pred_at_run": int(len(pi["box"])),
                    # ★★ `tp_prefix`：前 k+1 个预测（按置信度降序）下的累计真阳性数。
                    #   有了它就能离线算任意配额 q 的 TPᵢ(q) = tp_prefix[min(q, len)-1]。
                    #   见上方生成处的说明。
                    "tp_prefix": _tp_prefix.get(nm, []),
                    "n_pred_at": {},
                    "tp_at": {},
                    # ★ `conf_list`：该图**全部**预测的置信度（降序）。
                    #   `src/eval/proxy_density.py` 的四个代理量里有三个依赖它
                    #   （sum_conf / n_hi_conf / entropy），缺了它 ρ 就无从算起。
                    #   只存一次（不随 B 重复），控制文件体积。
                    "conf_list": [round(float(c), 6) for c in np.sort(cf)[::-1][:500]],
                }
            )
            entry["n_pred_at"][str(B)] = int(len(box))
            entry["tp_at"][str(B)] = n_tp
            recs.append(
                {
                    "name": pi["name"],
                    "n_gt": int(n_gt),
                    "n_pred": int(len(box)),
                    "n_tp": n_tp,
                    "recall": float(tp.sum() / n_gt) if n_gt else None,
                    "mean_conf": float(cf.mean()) if len(cf) else None,
                }
            )
        rec = np.array([x["recall"] for x in recs if x["recall"] is not None], dtype=float)
        n_pred = np.array([x["n_pred"] for x in recs], dtype=float)
        # 只统计有 GT 的图
        mask = np.array([x["n_gt"] > 0 for x in recs])
        rows.append(
            {
                "budget": B,
                "n_images_with_gt": int(mask.sum()),
                "recall_mean": round(float(rec.mean()), 5) if len(rec) else None,
                "recall_std": round(float(rec.std(ddof=1)), 5) if len(rec) > 1 else None,
                "recall_p10": round(float(np.percentile(rec, 10)), 5) if len(rec) else None,
                "n_pred_mean": round(float(n_pred.mean()), 2),
                "n_pred_at_cap_ratio": round(float((n_pred >= B).mean()), 4),
                "saturated_images": int((n_pred < B).sum()),
            }
        )

    # H2：召回损失 vs GT 密度
    dens = []
    Bref = budgets[-1]
    for pi, n_gt in zip(per_image, n_gt_per):
        if n_gt == 0:
            continue
        box, cf, cl = pi["box"], pi["conf"], pi["cls"]
        # ★ P1-B：与前两处同一确定性排序
        order = canonical_order(cf)
        box, cf, cl = box[order], cf[order], cl[order]
        r_full = greedy_match(box, cf, cl, gt[pi["name"]]).sum() / n_gt
        boxB, cfB, clB = box[:Bref], cf[:Bref], cl[:Bref]
        r_B = greedy_match(boxB, cfB, clB, gt[pi["name"]]).sum() / n_gt if len(boxB) else 0.0
        dens.append({"name": pi["name"], "n_gt": int(n_gt), "recall_full": float(r_full), "recall_B": float(r_B)})

    return {
        "weights": weights,
        "data": data,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "nms_free": nms_free,
        "max_budget_run": max_budget_run,
        "n_images": len(imgs),
        "infer_seconds": round(infer_s, 2),
        "infer_ms_per_image": round(infer_s / max(len(imgs), 1) * 1000, 2),
        "classes": names,
        "curve": rows,
        "per_image": list(per_image_records.values()),
        "per_image_density": dens,
        "gt_density": {
            "mean": round(float(n_gt_per.mean()), 2),
            "median": int(np.median(n_gt_per)),
            "p90": int(np.percentile(n_gt_per, 90)),
            "p99": int(np.percentile(n_gt_per, 99)),
            "max": int(n_gt_per.max()),
            "n_over_300": int((n_gt_per > 300).sum()),
            "n_over_100": int((n_gt_per > 100).sum()),
        },
    }


def verify_equivalence(weights: str, data: str, split: str, budgets: list[int], *, imgsz: int, conf: float, iou: float, device: str, nms_free: bool, n_check: int = 40) -> dict:
    """验证核心假设：**用大 B 推理再离线截断  ==  直接用 B 推理**。

    为什么必须验（不能假设）：
        `predictor.py:351` 的 `set_head_attr(max_det=max(args.max_det,300))` 表明
        one-to-one 头的 **top-k 就是 max_det**，且 `max(B, 300)` 意味着 B<300 时
        头里实际上还是 300 —— 这条路径下等价性可能成立；
        但 **one-to-many 头**的 max_det 施加在 NMS 之后，语义不同。
        若不等价，之前所有"离线截断"的结果都不可信。

    做法：对同一批图，分别用 max_det=B 与 max_det=big 各跑一次，比前 B 个框的
          坐标/分数（容差 1e-4）与框数。
    """
    from ultralytics import YOLO

    m = YOLO(weights)
    imgs, _ = load_split(data, split)
    imgs = imgs[:n_check]
    kw = {"nms": False} if nms_free else {}
    big = max(budgets)

    def _run(B):
        res = m.predict(
            source=[str(p) for p in imgs], imgsz=imgsz, conf=conf, iou=iou,
            max_det=B, device=device, stream=True, verbose=False, **kw,
        )
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append((np.zeros((0, 4)), np.zeros(0)))
            else:
                out.append((b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float)))
        return out

    ref_big = _run(big)
    report: dict = {"n_check": len(imgs), "nms_free": nms_free, "big": big, "per_budget": {}}
    for B in budgets:
        ref_B = _run(B)
        same_n = 0
        max_dev = 0.0
        for (bb, cb), (bB, cB) in zip(ref_big, ref_B):
            n_expect = min(len(bb), B)
            if len(bB) != n_expect:
                continue
            if n_expect == 0:
                same_n += 1
                continue
            # 大 B 的前 n 个应按置信度降序 = 小 B 的输出
            # ★ P1-B：这里同样属于"预测置信度排序"，必须与生产路径用同一确定性
            #    排序，否则置信度并列时会选出不同的子集，把排序不确定性误报成
            #    "X.xyxy 与 max_det 不等价"的坐标偏差。
            order = canonical_order(cb)[:n_expect]
            dev = float(np.abs(bb[order] - bB).max()) if n_expect else 0.0
            max_dev = max(max_dev, dev)
            if dev < 1e-4:
                same_n += 1
        report["per_budget"][B] = {
            "images_matching": same_n,
            "images_total": len(imgs),
            "max_coord_deviation": round(max_dev, 6),
            "equivalent": same_n == len(imgs),
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="逐图输出预算扫描")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--exp", required=True, help="实验标识，用于输出文件名")
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS_DEFAULT)
    ap.add_argument("--head", choices=["both", "o2m", "o2o"], default="both")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--verify-equivalence", action="store_true",
                    help="★ 先验证「大 B 推理+离线截断 == 直接用 B 推理」，再做扫描")
    ap.add_argument("--limit", type=int, default=None,
                    help="★ 只用前 N 张图（管线验证/小规模试跑用）。"
                         "⚠️ 正式结果**不得**设此项——系统 RAM 15.7GB，"
                         "一次性 predict 全部 548/1610 张会 MemoryError（实测），"
                         "故验证阶段必须设小值。见 AGENTS-LESSONS P4。")
    a = ap.parse_args()

    budgets = sorted(set(a.budgets))
    max_b = max(budgets)
    out_dir = PROJECT_ROOT / "runs" / "budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = PROJECT_ROOT / "results" / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    heads = {"o2m": False, "o2o": True}
    if a.head != "both":
        heads = {a.head: heads[a.head]}

    if a.verify_equivalence:
        print("★ 等价比对：大 B 推理+离线截断  vs  直接用 B 推理\n")
        eq_all = True
        for tag, nms_free in heads.items():
            eq = verify_equivalence(
                a.weights, a.data, a.split, budgets,
                imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=a.device, nms_free=nms_free,
            )
            print(f"  [{tag}] nms_free={nms_free} big={eq['big']}")
            for B, r in eq["per_budget"].items():
                flag = "✅" if r["equivalent"] else "❌"
                print(f"     B={B:5d}  匹配 {r['images_matching']}/{r['images_total']}  最大坐标偏差={r['max_coord_deviation']}  {flag}")
                eq_all &= r["equivalent"]
            p = out_dir / f"{a.exp}_equivalence_{tag}.json"
            p.write_text(json.dumps(eq, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{'✅ 等价性成立，离线截断结果可信' if eq_all else '❌ 等价性不成立！必须改为逐 B 真实推理'}")
        if not eq_all:
            print("   → 这是重要发现，须记入 AGENTS-LESSONS.md 并修改扫描实现")
            return 1
        print()

    all_rep = {}
    for tag, nms_free in heads.items():
        rep = run_head(
            a.weights, a.data, a.split, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
            device=a.device, budgets=budgets, nms_free=nms_free,
            max_budget_run=max_b, tag=tag, limit=a.limit,
        )
        all_rep[tag] = rep
        p = out_dir / f"{a.exp}_{a.split}_{tag}.json"
        p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{tag}] 曲线：")
        for row in rep["curve"]:
            print(
                f"     B={row['budget']:5d}  recall={row['recall_mean']}"
                f"  pred/img={row['n_pred_mean']:7.2f}  触及上限的图占比={row['n_pred_at_cap_ratio']}"
            )
        print(f"  GT 密度：{rep['gt_density']}")

    summary = tbl_dir / f"budget-{a.exp}-{a.split}.json"
    summary.write_text(json.dumps(all_rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总 -> {summary}")
    print("\n⚠️ 判定 H1/H2 需看：① 逐图 B* 分布（curve 的 recall 是否在 B≪300 时已饱和）")
    print("   ② per_image_density 里 recall_B 与 n_gt 的关系。详见 docs/research-framework.md §6 的 G1。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
