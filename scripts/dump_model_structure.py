"""
scripts/dump_model_structure.py —— 从**真实权重**导出模型结构（`docs/MODEL-STRUCTURE.md` 的数据源）

═══════════════════════════════════════════════════════════════════════════
为什么需要这个脚本
═══════════════════════════════════════════════════════════════════════════
`docs/MODEL-STRUCTURE.md` 里每个数字都必须能**一条命令复现**，否则它就是散文而非文档。
本脚本打印：
  · 汇总（参数量 / GFLOPs / 层数）
  · 24 个顶层 module（含真实拓扑 `f` 字段与逐层参数量）
  · 双头检测头的**逐尺度权重形状**（证明两个头是完整副本）
  · 类别名（**读自权重**，非 yaml）
  · 训练元信息与权重内自带的评测指标

用法：
    python scripts/dump_model_structure.py
    python scripts/dump_model_structure.py --weights runs/train/<run>/weights/best.pt
    python scripts/dump_model_structure.py --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT = ROOT / "runs" / "train" / "r1_yolo26n_visdrone_640__s42" / "weights" / "best.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description="从权重导出模型结构")
    ap.add_argument("--weights", type=Path, default=DEFAULT)
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "tables" / "model-structure.json")
    ap.add_argument("--detailed", action="store_true",
                    help="同时把逐权重明细写到 results/tables/model-structure-detail.txt")
    a = ap.parse_args()

    if not a.weights.exists():
        print(f"❌ 找不到权重：{a.weights}")
        return 1

    from ultralytics import YOLO
    import torch

    print("=" * 100)
    print(f"权重：{a.weights}")
    print("=" * 100)

    m = YOLO(str(a.weights))
    net = m.model
    layers = net.model            # nn.Sequential（DetectionModel 本身不可下标）

    buf = io.StringIO()
    with redirect_stdout(buf):
        net.info(detailed=False)
    print(buf.getvalue().strip())

    n_par = sum(p.numel() for p in net.parameters())
    print(f"\n参数量 = {n_par:,}")

    # ── 顶层 module（真实拓扑）────────────────────────────────
    print(f"\n{'idx':>4} {'from':>10} {'type':<32} {'params':>11}")
    print("-" * 62)
    mods = []
    for i, L in enumerate(layers):
        f = getattr(L, "f", None)
        t = type(L).__name__
        n = sum(p.numel() for p in L.parameters())
        print(f"{i:>4} {str(f):>10} {t:<32} {n:>11,}")
        mods.append({"i": i, "from": f if not hasattr(f, "tolist") else f.tolist(),
                     "type": t, "params": n})
    print("-" * 62)
    print(f"{'':>4} {'':>10} {'合计':<32} {n_par:>11,}")
    print(f"\n顶层 module 数 = {len(layers)}")

    # ── 检测头 ───────────────────────────────────────────────
    h = layers[-1]
    print(f"\n检测头 = {type(h).__name__}")
    head = {}
    for k in ("nc", "nl", "reg_max", "no", "max_det", "end2end", "stride"):
        v = getattr(h, k, None)
        if v is not None:
            head[k] = v.tolist() if hasattr(v, "tolist") else v
            print(f"  {k} = {head[k]}")
    for nm in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        seq = getattr(h, nm, None)
        if seq is None:
            continue
        # ★ 取每个尺度分支的**最后一个 Conv2d**（即输出层）—— 那才是论文引用的形状，
        #   如 cv2 末层 [4,16,1,1]（4 个回归值）、cv3 末层 [10,80,1,1]（10 类）。
        #   取第一个会是 [16,64,3,3] 这类**中间层**，不反映输出维度。
        shapes = []
        for s in seq:
            last = None
            for x in s.modules():
                if getattr(x, "weight", None) is not None and x.weight.dim() == 4:
                    last = x
            if last is not None:
                shapes.append(list(last.weight.shape))
        tot = sum(p.numel() for p in seq.parameters())
        head[nm] = {"params": tot, "out_shapes": shapes}
        print(f"  {nm:<14} {tot:>8,} 参数   输出层权重形状 {shapes}")
    head["dfl_type"] = type(getattr(h, "dfl", None)).__name__
    print(f"  dfl 类型 = {head['dfl_type']}"
          f"   ← Identity 表示**已去掉 DFL**（对照 YOLOv8/v11 的 4×reg_max 输出）")

    # ── 类别（读自权重）──────────────────────────────────────
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    print(f"\n类别（{len(names)}，读自权重）：")
    for i, nm in names.items():
        print(f"  {i}: {nm}")

    # ── 训练元信息 ───────────────────────────────────────────
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    ta = ck.get("train_args", {}) or {}
    print("\n训练元信息：")
    for k in ("data", "epochs", "batch", "imgsz", "seed", "optimizer", "workers", "patience"):
        if k in ta:
            print(f"  {k} = {ta[k]}")
    tr = ck.get("train_metrics", {}) or {}
    for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
              "metrics/precision(B)", "metrics/recall(B)"):
        if k in tr:
            print(f"  {k} = {tr[k]:.4f}")

    if a.detailed:
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            net.info(detailed=True)
        dp = ROOT / "results" / "tables" / "model-structure-detail.txt"
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(buf2.getvalue(), encoding="utf-8")
        print(f"\n逐权重明细（{len(buf2.getvalue().splitlines())} 行）→ {dp.relative_to(ROOT)}")

    # ── 落盘 ─────────────────────────────────────────────────
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({
        "weights": str(a.weights.relative_to(ROOT)) if a.weights.is_relative_to(ROOT) else str(a.weights),
        "n_params": n_par,
        "n_modules": len(layers),
        "modules": mods,
        "head": head,
        "names": {str(k): v for k, v in names.items()},
        "train_args": {k: ta[k] for k in ta
                       if isinstance(ta[k], (str, int, float, bool, type(None)))},
        "train_metrics": {k: float(v) for k, v in tr.items()
                          if isinstance(v, (int, float))},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写出 {a.json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
