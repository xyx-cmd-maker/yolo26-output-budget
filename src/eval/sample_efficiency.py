"""
yolo26-detection / src/eval/sample_efficiency.py
================================================
"标注量 vs 检测精度"的样本效率曲线：回答「这个类还需要标多少张」。

为什么需要（本项目实测暴露的问题）：
    p02 三个种子上 `Deceleration_sign` 的 AP 稳定偏低（0.594 ± 0.043），
    而 Pedestrian 0.844、TrafficLight 0.841。但**低 AP 不等于"这个类更难"**——
    它可能只是"这个类的测试样本太少，估计本身不可靠"。
    要区分这两种解释，需要一条 **AP ~ 该类标注框数** 的曲线。

方法（bootstrap 子采样，不改动任何训练结果）：
    固定模型权重与评测集，对**某一类的真值框**做有放回/无放回子采样，
    在该子集上重算该类 AP，得到 (n_boxes, AP) 的散点与分位数带。
    这直接回答：**要在 ±δ 的精度上测出这个类的 AP，需要多少个框？**

⚠️ 本方法测的是「**评测侧**的样本量需求」，不是「训练侧的数据需求」。
   两者是不同问题：
     * 评测侧：给定一个固定模型，n 个测试框能把 AP 估到多准？（本脚本）
     * 训练侧：要训出好模型，需要多少训练样本？（需要重训，成本高）
   本项目先做评测侧，因为它是**零训练成本**的，且直接决定"逐类 AP 能不能写进论文"。

用法：
    python -m src.eval.sample_efficiency --weights runs/train/<run>/weights/best.pt \\
        --data src/configs/datasets/<ds>.yaml --split test --class-name Deceleration_sign
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

__all__ = ["collect_predictions", "ap_at_iou", "sample_efficiency_curve"]

RNG = np.random.default_rng(20261014)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _split_images(data_yaml: str, split: str) -> list[str]:
    import yaml

    p = Path(data_yaml)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    sp = Path(cfg[split])
    if not sp.is_absolute():
        sp = (root / sp).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [str(f) for f in sorted(sp.rglob("*")) if f.suffix.lower() in exts]


def _labels_dir_for(img_dir: Path) -> Path:
    """把 .../images/<split> 映射到 .../labels/<split>。

    ⚠️ 必须**只替换最后一个** `images` 路径组件。
    踩过的坑：用 `str(img_dir).replace("images", "labels")` 会连祖先目录一起改，
    例如本项目数据在 `...\\yolo\\images\\dataset125\\temporal_split\\images\\test`，
    全局替换会得到 `...\\yolo\\labels\\dataset125\\...` —— 完全错的路径，
    而且**不报错**，只是安静地返回"没有真值框"。`data_card.py` 用的是正确的
    从右往左找组件的写法，这里保持一致。
    """
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def load_gt_from_labels(data_yaml: str, split: str, class_id: int) -> dict[str, tuple[np.ndarray, int]]:
    """直接从 labels 目录读真值，返回 {图像名: (xyxy 绝对坐标, 该类框数)}。

    这是 GT 的可靠来源（不依赖 ultralytics 内部结构）。
    """
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

    out: dict[str, tuple[np.ndarray, int]] = {}
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lb = lbl_dir / (img.stem + ".txt")
        boxes = []
        if lb.exists():
            with Image.open(img) as im:
                W, H = im.size
            for ln in lb.read_text(encoding="utf-8").strip().splitlines():
                pr = ln.split()
                if len(pr) < 5 or int(float(pr[0])) != class_id:
                    continue
                _, cx, cy, w, h = (float(v) for v in pr[:5])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        out[img.name] = (np.asarray(boxes, dtype=float).reshape(-1, 4), len(boxes))
    return out


def ap_at_iou(
    gt_by_image: dict[str, np.ndarray],
    pred: list[tuple[str, np.ndarray, np.ndarray]],
    iou_thr: float = 0.5,
) -> float:
    """单类、单 IoU 阈值的 AP（101 点插值，COCO 风格）。

    Args:
        gt_by_image: {图像名: (M,4) xyxy}
        pred: [(图像名, (N,4) xyxy, (N,) conf), ...]
    """
    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0:
        return float("nan")

    records = []
    for name, boxes, confs in pred:
        g = gt_by_image.get(name, np.zeros((0, 4)))
        if len(boxes) == 0:
            continue
        iou = _box_iou(boxes, g)
        matched = np.zeros(len(g), dtype=bool)
        for i in np.argsort(-confs):
            tp = False
            if len(g):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= iou_thr and not matched[j]:
                    matched[j] = True
                    tp = True
            records.append((float(confs[i]), tp))

    if not records:
        return 0.0
    records.sort(key=lambda x: -x[0])
    tp = np.array([r[1] for r in records], dtype=float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)

    # 单调化 + 101 点插值
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    out = np.zeros_like(grid)
    valid = idx < len(prec)
    out[valid] = prec[idx[valid]]
    return float(out.mean())


def sample_efficiency_curve(
    gt_by_image: dict[str, np.ndarray],
    pred: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    n_repeats: int = 200,
    seed: int = 20261014,
) -> dict:
    """对**图像**做无放回子采样，得到 (总框数, AP) 的曲线。

    采样单位是**图像**而不是框：因为 AP 是在图像集合上算的，逐框采样会破坏
    "同一张图的多个框一起出现"的结构，得到偏乐观的估计。
    """
    rng = np.random.default_rng(seed)
    names = [n for n in gt_by_image if len(gt_by_image[n]) > 0]
    if not names:
        return {"error": "该类没有任何真值框"}
    total_boxes = int(sum(len(gt_by_image[n]) for n in names))

    levels = sorted({1, 2, 3, 5, 8, 13, 21, 34, 55, len(names)})
    levels = [L for L in levels if L <= len(names)]

    curve = []
    for L in levels:
        aps = []
        for _ in range(n_repeats):
            pick = rng.choice(names, size=L, replace=False)
            sub_gt = {n: gt_by_image[n] for n in pick}
            sub_pred = [p for p in pred if p[0] in set(pick)]
            a = ap_at_iou(sub_gt, sub_pred)
            if not np.isnan(a):
                aps.append(a)
        if not aps:
            continue
        aps_a = np.asarray(aps)
        n_boxes = int(sum(len(gt_by_image[n]) for n in names[:L])) if False else None
        curve.append(
            {
                "n_images": L,
                "n_boxes_expected": round(total_boxes * L / len(names), 1),
                "ap_mean": round(float(aps_a.mean()), 4),
                "ap_std": round(float(aps_a.std(ddof=1)) if len(aps_a) > 1 else 0.0, 4),
                "ap_p05": round(float(np.percentile(aps_a, 5)), 4),
                "ap_p95": round(float(np.percentile(aps_a, 95)), 4),
                "ap_ci_width": round(float(np.percentile(aps_a, 95) - np.percentile(aps_a, 5)), 4),
            }
        )

    return {
        "n_classes_images": len(names),
        "total_boxes": total_boxes,
        "n_repeats": n_repeats,
        "curve": curve,
        "full_set_ap": curve[-1]["ap_mean"] if curve else None,
    }


def sweep_all_classes(
    weights: str,
    data: str,
    *,
    split: str,
    imgsz: int,
    conf: float,
    device: str,
    iou_thr: float,
    repeats: int,
) -> dict:
    """对**每个类**跑一遍样本效率曲线，并做一次推理复用（避免重复推理）。"""
    from ultralytics import YOLO

    m = YOLO(weights)
    names = {int(k): v for k, v in (m.names or {}).items()}
    imgs = _split_images(data, split)
    print(f"推理 {len(imgs)} 张（{split}）…", flush=True)
    rs = m.predict(source=imgs, imgsz=imgsz, conf=conf, iou=0.7, max_det=300, device=device, stream=True, verbose=False)

    pred_by_cls: dict[int, list] = {c: [] for c in names}
    for r in rs:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        cl = r.boxes.cls.cpu().numpy().astype(int)
        bx = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        nm = Path(r.path).name
        for c in np.unique(cl):
            k = cl == c
            pred_by_cls.setdefault(int(c), []).append((nm, bx[k], cf[k]))

    out: dict = {
        "weights": weights,
        "data": data,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou_threshold": iou_thr,
        "repeats": repeats,
        "classes": {},
    }
    for cid, cname in sorted(names.items()):
        gt = {k: v[0] for k, v in load_gt_from_labels(data, split, cid).items()}
        n_box = int(sum(len(v) for v in gt.values()))
        n_img = int(sum(1 for v in gt.values() if len(v)))
        rec = sample_efficiency_curve(gt, pred_by_cls.get(cid, []), n_repeats=repeats)
        rec.update(class_id=cid, class_name=cname, n_gt_images=n_img, n_gt_boxes=n_box)
        out["classes"][cname] = rec
        print(f"  [{cid}] {cname:20s} {n_img:4d} 图 / {n_box:5d} 框 | full AP@{iou_thr}={rec.get('full_set_ap')}")
    return out


def plot_curve(report: dict, out_png: Path) -> Path | None:
    """画 AP vs 框数 曲线（含 5–95% 分位带）。需要 matplotlib。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未装 matplotlib，跳过画图")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    for cname, rec in report["classes"].items():
        curve = rec.get("curve") or []
        if not curve:
            continue
        x = [c["n_boxes_expected"] for c in curve]
        y = [c["ap_mean"] for c in curve]
        lo = [c["ap_p05"] for c in curve]
        hi = [c["ap_p95"] for c in curve]
        (line,) = ax.plot(x, y, marker="o", ms=4, lw=1.6, label=f"{cname} (n={rec['n_gt_boxes']})")
        ax.fill_between(x, lo, hi, alpha=0.16, color=line.get_color())

    ax.set_xscale("log")
    ax.set_xlabel("Number of ground-truth boxes of that class (evaluation set)")
    ax.set_ylabel(f"AP@{report['iou_threshold']}")
    ax.set_title(
        f"Per-class AP estimate vs. annotation count\n"
        f"({report['split']} split, imgsz={report['imgsz']}, bootstrap n={report['repeats']})",
        fontsize=10,
    )
    ax.grid(alpha=0.28, which="both", ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description="逐类样本效率曲线（AP vs 标注框数）")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--class-id", type=int, default=None, help="只分析某一个类；不给则分析全部类")
    ap.add_argument("--class-name", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None, help="JSON 输出路径")
    ap.add_argument("--png", type=Path, default=None, help="PNG 输出路径")
    a = ap.parse_args()

    if a.class_id is None:
        rep = sweep_all_classes(
            a.weights, a.data, split=a.split, imgsz=a.imgsz, conf=a.conf,
            device=a.device, iou_thr=a.iou_thr, repeats=a.repeats,
        )
        rep["_summary"] = {
            c: {
                "n_gt_boxes": r["n_gt_boxes"],
                "n_gt_images": r["n_gt_images"],
                "full_set_ap": r.get("full_set_ap"),
                "ci_width_at_1_image": (r.get("curve") or [{}])[0].get("ap_ci_width"),
            }
            for c, r in rep["classes"].items()
        }
    else:
        gt = {k: v[0] for k, v in load_gt_from_labels(a.data, a.split, a.class_id).items()}
        n_total = sum(len(v) for v in gt.values())
        print(f"类 id={a.class_id} ({a.class_name})：{sum(1 for v in gt.values() if len(v))} 图 / {n_total} 框")
        if n_total == 0:
            print("该类没有真值框，无法分析")
            return 1
        from ultralytics import YOLO

        m = YOLO(a.weights)
        names = m.names
        rs = m.predict(source=_split_images(a.data, a.split), imgsz=a.imgsz, conf=a.conf, iou=0.7,
                       max_det=300, device=a.device, stream=True, verbose=False)
        pred = []
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            keep = r.boxes.cls.cpu().numpy().astype(int) == a.class_id
            if not keep.any():
                continue
            pred.append((Path(r.path).name, r.boxes.xyxy.cpu().numpy()[keep], r.boxes.conf.cpu().numpy()[keep]))
        rec = sample_efficiency_curve(gt, pred, n_repeats=a.repeats)
        cname = a.class_name or str(names.get(a.class_id))
        rec.update(class_id=a.class_id, class_name=cname, n_gt_boxes=n_total)
        rep = {
            "weights": a.weights, "data": a.data, "split": a.split, "imgsz": a.imgsz,
            "conf": a.conf, "iou_threshold": a.iou_thr, "repeats": a.repeats,
            "classes": {cname: rec},
        }

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("已写入", a.out)
    if a.png:
        print("图 ->", plot_curve(rep, a.png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
