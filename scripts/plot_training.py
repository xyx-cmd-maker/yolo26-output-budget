"""
scripts/plot_training.py —— **训练曲线可视化**（可复用）

用法：
    python scripts/plot_training.py                          # 自动画所有 r1_* 种子
    python scripts/plot_training.py --runs r1_x__s42 r1_x__s43 --out results/figures/curve.png
    python scripts/plot_training.py --live                   # 只看最新一个还在跑的

产出：`results/figures/training-<tag>.png`

设计要点（为什么这样画）：
  · **两个种子画在同一张图上** —— 这是"单种子"缺口的唯一可视化证据，
    审稿人要看的正是"种子间差多少"
  · **mAP 与 loss 分开两栏** —— 量纲不同，混在一起会误导
  · **标出最佳 epoch 与其值** —— 论文表格里的数字要能在图上找到出处
  · **y 轴不从 0 起**（mAP 栏）—— 否则早期 epoch 的差异被压扁；loss 栏从 0 起
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无窗口后端（本机是服务器式运行）
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs" / "train"
FIGDIR = PROJECT_ROOT / "results" / "figures"


def read_curve(run: Path) -> dict | None:
    p = run / "results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        return None

    def col(name: str) -> np.ndarray:
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out, dtype=float)

    ep = np.arange(1, len(rows) + 1)
    return {
        "name": run.name,
        "epoch": ep,
        "mAP50": col("metrics/mAP50(B)"),
        "mAP5095": col("metrics/mAP50-95(B)"),
        "precision": col("metrics/precision(B)"),
        "recall": col("metrics/recall(B)"),
        "box_loss": col("train/box_loss"),
        "cls_loss": col("train/cls_loss"),
        "dfl_loss": col("train/dfl_loss"),
        "val_box_loss": col("val/box_loss"),
        "lr": col("lr/pg0"),
        "time": col("time"),
        "n": len(rows),
    }


def find_runs(patterns: list[str] | None) -> list[Path]:
    if patterns:
        return [RUNS / p for p in patterns if (RUNS / p).is_dir()]
    return sorted(d for d in RUNS.iterdir()
                  if d.is_dir() and d.name.startswith("r1_") and (d / "results.csv").exists())


def main() -> int:
    ap = argparse.ArgumentParser(description="训练曲线可视化")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=140)
    a = ap.parse_args()

    runs = find_runs(a.runs)
    if not runs:
        print("❌ 没找到含 results.csv 的运行目录")
        return 1
    curves = [c for c in (read_curve(r) for r in runs) if c]
    if not curves:
        print("❌ 曲线为空")
        return 1

    FIGDIR.mkdir(parents=True, exist_ok=True)
    tag = "all" if len(curves) > 1 else curves[0]["name"]
    out = Path(a.out) if a.out else FIGDIR / f"training-{tag}.png"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"YOLO26n @ VisDrone2019-DET — training curves ({len(curves)} seed(s))",
                 fontsize=14, fontweight="bold")

    # ── ① mAP ────────────────────────────────────────────────
    ax = axes[0][0]
    for c in curves:
        ax.plot(c["epoch"], c["mAP50"], lw=1.8, label=f"{c['name']}  mAP50")
        ax.plot(c["epoch"], c["mAP5095"], lw=1.4, ls="--", alpha=0.75,
                label=f"{c['name']}  mAP50-95")
        i = int(np.nanargmax(c["mAP50"]))
        ax.plot(c["epoch"][i], c["mAP50"][i], "o", ms=6, mfc="none", mew=1.6)
        ax.annotate(f"{c['mAP50'][i]:.4f}\n(ep {c['epoch'][i]})",
                    (c["epoch"][i], c["mAP50"][i]), textcoords="offset points",
                    xytext=(6, -16), fontsize=8)
    ax.axhline(0.24, color="tab:red", lw=1.0, ls=":", label="G0 gate mAP50 >= 0.24")
    ax.axhline(0.130, color="tab:orange", lw=1.0, ls=":", label="G0 gate mAP50-95 >= 0.130")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP")
    ax.set_title("(a) Validation mAP  [y-axis not from 0]")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=1, loc="lower right")

    # ── ② 训练损失 ───────────────────────────────────────────
    ax = axes[0][1]
    for c in curves:
        for k, ls in (("box_loss", "-"), ("cls_loss", "--"), ("dfl_loss", ":")):
            ax.plot(c["epoch"], c[k], lw=1.3, ls=ls, label=f"{c['name'][-3:]} {k}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (normalized, ultralytics units)")
    ax.set_title("(b) Training losses")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

    # ── ③ Precision / Recall ─────────────────────────────────
    ax = axes[1][0]
    for c in curves:
        ax.plot(c["epoch"], c["precision"], lw=1.5, label=f"{c['name']}  precision")
        ax.plot(c["epoch"], c["recall"], lw=1.5, ls="--", label=f"{c['name']}  recall")
    ax.set_xlabel("epoch"); ax.set_ylabel("value")
    ax.set_title("(c) Validation precision / recall")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # ── ④ 收敛速度（每 epoch 的 mAP50 增量）────────────────────
    ax = axes[1][1]
    for c in curves:
        d = np.diff(c["mAP50"])
        # 滑动平均以看清趋势（原始增量噪声大）
        k = 5
        if d.size >= k:
            sm = np.convolve(d, np.ones(k) / k, mode="valid")
            ax.plot(c["epoch"][k:], sm, lw=1.6, label=f"{c['name']} ({k}-ep moving avg)")
        else:
            ax.plot(c["epoch"][1:], d, lw=1.6, label=c["name"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("d(mAP50)/d(epoch)")
    ax.set_title("(d) Convergence rate  [trend toward 0 = converged]")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    print(f"✅ 已保存 {out.relative_to(PROJECT_ROOT)}")

    # ── 控制台摘要（论文表格用）──────────────────────────────
    print("\n" + "=" * 88)
    print("摘要（可直接进论文表格）")
    print("=" * 88)
    print(f"  {'运行':<34} {'epochs':>7} {'最佳 mAP50':>11} {'@ep':>4} "
          f"{'最佳 mAP50-95':>13} {'@ep':>4} {'末轮 mAP50':>11}")
    print("  " + "-" * 92)
    for c in curves:
        i50 = int(np.nanargmax(c["mAP50"]))
        i95 = int(np.nanargmax(c["mAP5095"]))
        print(f"  {c['name']:<34} {c['n']:>7} {c['mAP50'][i50]:>11.4f} {c['epoch'][i50]:>4} "
              f"{c['mAP5095'][i95]:>13.4f} {c['epoch'][i95]:>4} {c['mAP50'][-1]:>11.4f}")

    if len(curves) > 1:
        finals = np.array([np.nanmax(c["mAP50"]) for c in curves])
        print(f"\n  ★ 种子间（最佳 mAP50）：均值 {finals.mean():.4f}  "
              f"标准差 {finals.std(ddof=1) if finals.size > 1 else float('nan'):.4f}  "
              f"范围 [{finals.min():.4f}, {finals.max():.4f}]")
        print("     ⚠️ 只有 >= 3 个种子才能进 `paper` 档聚合（`results_io` 的 min_seeds=3）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
