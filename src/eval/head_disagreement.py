"""
yolo26-detection / src/eval/head_disagreement.py
================================================
把 YOLO26 的**双头分歧**当作零成本的检测不确定性信号——并检验它是否真的有效。

机制依据（本项目实读 ultralytics 8.4.177 源码，2026-10-14）：
  1. `nn/modules/head.py`：`self.one2one_cv2 = copy.deepcopy(self.cv2)`
     → 两个头是**同架构拷贝**（容量相同）。
  2. `nn/modules/head.py`：
         x_detach = [xi.detach() for xi in x] if self.training else x
         # detach keeps one2one out of the backbone
     → 训练时 **one-to-one 头的梯度不回传到骨干**。
  3. `utils/loss.py`：
         self.one2many = v8DetectionLoss(model, tal_topk=10)
         self.one2one  = v8DetectionLoss(model, tal_topk=1)
     → 两头用**不同的标签分配目标**（o2m topk=10 / o2o topk=1）。

  ⇒ 同容量 + 不同训练信号 + 共享骨干 ⇒ 两头的分歧应是**认识论不确定性**
    （对同一输入"学到的东西不同"），而不是容量差异造成的噪声。
    若成立，它就是一个**免费**的不确定性估计器：两次头前向共享同一次骨干前向。

本模块做两件事：
  A. 用 `nms=False` 与默认两种推理路径取两头输出，按 IoU 匹配，得到逐框分歧度。
  B. 用 risk–coverage / AURC / 与误差的相关性检验它是否**真的**能预测错误。

⚠️ 判据必须**先于结果**声明（写在 RESULTS 里，不是事后挑）：
    H1（有效）: 按分歧度升序丢弃预测后，risk–coverage 曲线**优于**随机丢弃；
                且 AURC 显著低于用"分类置信度"排序的 AURC。
    H0（无效）: 分歧度与误差无关，AURC 与随机无差别。
    只有 H1 成立，这条才算发现；H0 也要如实报告。

用法：
    python -m src.eval.head_disagreement \\
        --weights weights/yolo26n.pt \\
        --data    src/configs/datasets/<ds>.yaml \\
        --split   val --imgsz 640 --conf 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

__all__ = ["dual_head_predict", "match_heads", "risk_coverage", "aurc", "run"]

RNG = np.random.default_rng(20261014)


def _iou_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    u = aa[:, None] + ab[None, :] - inter
    return np.where(u > 0, inter / u, 0.0)


def dual_head_predict(
    weights: str,
    images: list[str],
    *,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    device: str = "0",
) -> list[dict]:
    """对每张图分别取 one-to-one（nms=False）与 one-to-many（默认）两头的输出。

    注意：**不能用 `model.val()`** —— 8.4.135+ 在默认 max_det 下会自动抬升上限，
    会掩盖配额效应（见 docs/decisions.md D2）。这里走 predict 路径。
    """
    from ultralytics import YOLO

    m = YOLO(weights)
    out: list[dict] = []

    def _run(nms_free: bool):
        kw = {"nms": False} if nms_free else {}
        res = m.predict(
            source=images, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
            device=device, stream=True, verbose=False, **kw,
        )
        got = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                got.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), (r.orig_shape[0], r.orig_shape[1])))
            else:
                got.append((
                    b.xyxy.cpu().numpy().astype(float),
                    b.conf.cpu().numpy().astype(float),
                    b.cls.cpu().numpy().astype(int),
                    (r.orig_shape[0], r.orig_shape[1]),
                ))
        return got

    a = _run(False)   # one-to-many（需 NMS）
    b = _run(True)    # one-to-one（NMS-free）
    for i, img in enumerate(images):
        out.append(
            {
                "name": Path(img).name,
                "shape": a[i][3],
                "o2m_box": a[i][0], "o2m_conf": a[i][1], "o2m_cls": a[i][2],
                "o2o_box": b[i][0], "o2o_conf": b[i][1], "o2o_cls": b[i][2],
            }
        )
    return out


def match_heads(rec: dict, iou_thr: float = 0.5) -> dict:
    """把 one-to-one 的每个预测与 one-to-many 中同类别的预测按 IoU 匹配。

    one-to-one 是**部署时真正使用的头**（NMS-free 路径），因此以它为分析主体：
    对它的每个框，找 one-to-many 里同类且 IoU 最大的框，记下：
      * iou_agree    : 几何一致性（0 表示 one-to-many 根本不认同这个框）
      * conf_gap     : 置信度差 |c_o2o - c_o2m|
      * conf_gap_rel : 相对置信度差
    分歧度 = 1 - iou_agree（几何分歧）；也单独保留 conf_gap 供对比。
    """
    ob, oc, ok = rec["o2o_box"], rec["o2o_conf"], rec["o2o_cls"]
    mb, mc, mk = rec["o2m_box"], rec["o2m_conf"], rec["o2m_cls"]
    n = len(ob)
    if n == 0:
        return {"n": 0, "iou_agree": np.zeros(0), "conf_gap": np.zeros(0),
                "conf_gap_rel": np.zeros(0), "o2o_conf": np.zeros(0),
                "o2o_box": ob, "o2o_cls": ok}

    iou_agree = np.zeros(n)
    conf_gap = np.zeros(n)
    conf_gap_rel = np.zeros(n)
    for i in range(n):
        same = np.where(mk == ok[i])[0]
        if len(same) == 0:
            iou_agree[i] = 0.0
            conf_gap[i] = oc[i]
            conf_gap_rel[i] = 1.0
            continue
        ious = _iou_pairwise(ob[i : i + 1], mb[same])[0]
        j = int(np.argmax(ious))
        iou_agree[i] = float(ious[j])
        conf_gap[i] = abs(float(oc[i]) - float(mc[same[j]]))
        conf_gap_rel[i] = conf_gap[i] / max(float(oc[i]), 1e-9)
    return {
        "n": n,
        "iou_agree": iou_agree,
        "conf_gap": conf_gap,
        "conf_gap_rel": conf_gap_rel,
        "o2o_conf": oc,
        "o2o_box": ob,
        "o2o_cls": ok,
    }


def _load_gt(data_yaml: str, split: str, names: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """从 labels 目录读真值，返回 {图像名: (xyxy(N,4), cls(N,))}。"""
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
    # 从右往左替换最后一个 images 组件（不能用全局 replace，见 sample_efficiency.py 的注释）
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    lbl_dir = Path(*parts)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        lb = lbl_dir / (img.stem + ".txt")
        boxes, cls = [], []
        if lb.exists():
            with Image.open(img) as im:
                W, H = im.size
            for ln in lb.read_text(encoding="utf-8").strip().splitlines():
                pr = ln.split()
                if len(pr) < 5:
                    continue
                c = int(float(pr[0]))
                _, cx, cy, w, h = (float(v) for v in pr[:5])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
                cls.append(c)
        out[img.name] = (
            np.asarray(boxes, dtype=float).reshape(-1, 4),
            np.asarray(cls, dtype=int).reshape(-1),
        )
    return out


def _is_tp(box: np.ndarray, cls: int, gt_b: np.ndarray, gt_c: np.ndarray, used: np.ndarray, iou_thr: float) -> bool:
    same = np.where((gt_c == cls) & (~used))[0]
    if len(same) == 0:
        return False
    ious = _iou_pairwise(box[None, :], gt_b[same])[0]
    j = int(np.argmax(ious))
    if ious[j] >= iou_thr:
        used[same[j]] = True
        return True
    return False


def risk_coverage(scores: np.ndarray, correct: np.ndarray, n_points: int = 50) -> dict:
    """Risk–Coverage 曲线。

    这里"coverage"=保留比例（按 scores 降序保留最可信的一部分预测），
    "risk"=保留部分的错误率。也可以用"按 scores 升序丢弃"来理解，二者等价。
    随机基线 = 全局错误率。
    """
    n = len(scores)
    if n == 0:
        return {"coverage": [], "risk": [], "risk_random": [], "global_error_rate": None}
    order = np.argsort(-scores)  # 最可信在前
    cor = correct[order].astype(float)
    cov, risk = [], []
    for k in np.linspace(1, n, n_points).astype(int):
        k = max(1, min(k, n))
        cov.append(k / n)
        risk.append(1.0 - cor[:k].mean())
    return {
        "coverage": [round(float(c), 4) for c in cov],
        "risk": [round(float(r), 4) for r in risk],
        "risk_random": [round(float(1.0 - cor.mean()), 4)] * len(cov),
        "global_error_rate": round(float(1.0 - cor.mean()), 4),
    }


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk–Coverage curve（越低越好）。数值积分（梯形）。"""
    rc = risk_coverage(scores, correct)
    if not rc["coverage"]:
        return float("nan")
    return float(np.trapezoid(rc["risk"], rc["coverage"]))


def run(
    weights: str,
    data: str,
    *,
    split: str,
    imgsz: int,
    conf: float,
    device: str,
    iou_thr: float,
    max_det: int,
) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    names = cfg.get("names") or {}
    img_dir = Path(cfg[split])
    root = Path(cfg["path"])
    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [str(f) for f in sorted(img_dir.rglob("*")) if f.suffix.lower() in exts]

    print(f"推理 {len(images)} 张（{split}），双头各一遍 …", flush=True)
    recs = dual_head_predict(weights, images, imgsz=imgsz, conf=conf, device=device, max_det=max_det)
    gt = _load_gt(data, split, names)

    all_iou_agree, all_conf_gap, all_corr, all_o2o_conf = [], [], [], []
    per_image = []
    for r in recs:
        m = match_heads(r, iou_thr=iou_thr)
        if m["n"] == 0:
            continue
        gb, gc = gt.get(r["name"], (np.zeros((0, 4)), np.zeros(0, int)))
        used = np.zeros(len(gb), dtype=bool)
        corr = np.array(
            [_is_tp(m["o2o_box"][i], int(m["o2o_cls"][i]), gb, gc, used, iou_thr) for i in range(m["n"])],
            dtype=bool,
        )
        all_iou_agree.append(m["iou_agree"])
        all_conf_gap.append(m["conf_gap"])
        all_corr.append(corr)
        all_o2o_conf.append(m["o2o_conf"])
        per_image.append({"name": r["name"], "n_pred": m["n"], "n_tp": int(corr.sum())})

    if not all_corr:
        return {"error": "one-to-one 头没有任何预测，无法分析"}

    iou_agree = np.concatenate(all_iou_agree)
    conf_gap = np.concatenate(all_conf_gap)
    correct = np.concatenate(all_corr)
    o2o_conf = np.concatenate(all_o2o_conf)

    # ★ 分歧度 = 1 - 几何一致性。分歧越小 → 越可信。
    disagree = 1.0 - iou_agree

    res = {
        "weights": weights,
        "data": data,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "max_det": max_det,
        "iou_match_threshold": iou_thr,
        "n_images": len(images),
        "n_one2one_preds": int(len(correct)),
        "global_error_rate": round(float(1 - correct.mean()), 4),
        "aurc_by_head_disagreement": round(aurc(-disagree, correct), 5),
        "aurc_by_o2o_confidence": round(aurc(o2o_conf, correct), 5),
        "aurc_random_baseline": round(float(1 - correct.mean()), 5),
        "risk_coverage_disagreement": risk_coverage(-disagree, correct, 20),
        "risk_coverage_confidence": risk_coverage(o2o_conf, correct, 20),
        "corr_disagreement_vs_error": round(float(np.corrcoef(disagree, (~correct).astype(float))[0, 1]), 4)
        if len(np.unique(disagree)) > 1
        else None,
        "corr_confidence_vs_error": round(float(np.corrcoef(o2o_conf, (~correct).astype(float))[0, 1]), 4)
        if len(np.unique(o2o_conf)) > 1
        else None,
        "iou_agree_quantiles": {
            q: round(float(np.quantile(iou_agree, q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "per_image_headline": per_image[:20],
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="双头分歧 → 检测不确定性信号的有效性检验")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="判 TP 的 IoU 阈值")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rep = run(
        a.weights, a.data, split=a.split, imgsz=a.imgsz, conf=a.conf,
        device=a.device, iou_thr=a.iou_thr, max_det=a.max_det,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("已写入", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
