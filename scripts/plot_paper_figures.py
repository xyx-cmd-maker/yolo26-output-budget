"""
scripts/plot_paper_figures.py —— 论文正文图（数据全部来自已落盘产物，零 GPU）

产出：
  results/figures/fig1-density-distributions.png   逐图目标数分布（跨数据集）
  results/figures/fig2-delta-vs-budget.png         Δworst 随预算 B 的 U 形 + 三个 regime 区间

用法：
    python scripts/plot_paper_figures.py            # 画全部
    python scripts/plot_paper_figures.py --only 1   # 只画 Figure 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════
def figure1() -> Path:
    """逐图目标数分布：VisDrone / CrowdHuman / COCO。

    ★ 为什么画 CDF 而不是直方图：直方图要选 bin 宽（人为参数），
      而 **CDF 无参数**，且**直接读出 `frac(nᵢ > B)`** —— 那正是论文用的量。
    ★ CDF 由 `n-gt-distributions.json` 里的 `hist_edges`/`hist_counts` 重建
      （第 47 轮加的分箱字段），无需重新解析原始标注。
    """
    d = _load(TAB / "n-gt-distributions.json")

    series = [
        ("visdrone_val", "VisDrone-DET val (aerial, dense)", "#d62728"),
        ("highd", "highD (aerial highway)", "#9467bd"),
        ("crowdhuman_train", "CrowdHuman train (ground, crowded)", "#1f77b4"),
        ("coco_val2017", "COCO val2017 (ground, sparse)", "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    fig.suptitle("Per-image annotated object counts across detection benchmarks",
                 fontsize=13, fontweight="bold")

    # ── (a) 互补 CDF：P(nᵢ > B) ──────────────────────────────
    ax = axes[0]
    for key, lab, c in series:
        if key not in d or "hist_edges" not in d[key]["stats"]:
            continue
        st = d[key]["stats"]
        e = np.array(st["hist_edges"], dtype=float)      # bin 左边界（去重后）
        cts = np.array(st["hist_counts"], dtype=float)   # 长度 = len(e)（最后一箱含右尾）
        tot = cts.sum()
        # ★ 用 len(cts) 显式对齐，不假设 e 的长度（np.unique 去重会使长度变化）
        n_bins = len(cts)
        right = np.concatenate([e[1:n_bins], [e[n_bins - 1] + 1.0]]) if n_bins > 1 else e[:1]
        ccdf = np.array([cts[i:].sum() for i in range(n_bins)]) / tot * 100
        ax.step(right, ccdf, where="post", color=c, lw=2.0, label=lab)
        ax.annotate(f"mean $n_i$ = {st['mean']:.1f},  max = {st['max']}",
                    xy=({"coco_val2017": (0.30, 0.30), "crowdhuman_train": (0.03, 0.60),
                         "visdrone_val": (0.30, 0.78)}[key]),
                    xycoords="axes fraction", ha="left", fontsize=8, color=c,
                    bbox=dict(fc="white", ec=c, alpha=0.95, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("output budget $B$  (boxes per image, log scale)")
    ax.set_ylabel(r"$P(n_i > B)$   (% of images)")
    ax.set_title("(a) How often a fixed budget binds")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-3, 105)

    # ── (b) 预算利用率 ───────────────────────────────────────
    ax = axes[1]
    for key, lab, c in series:
        if key not in d:
            continue
        Bs = [u["B"] for u in d[key]["util"]]
        ut = [u["utilisation"] * 100 for u in d[key]["util"]]
        ax.plot(Bs, ut, "-s", color=c, lw=2.0, ms=4, label=lab)
    ax.axhline(100, color="k", lw=0.8, ls=":")
    ax.text(11, 96, "budget fully absorbed", fontsize=7.5, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("output budget $B$  (boxes per image, log scale)")
    ax.set_ylabel(r"utilisation  $\Sigma\min(B,n_i)\,/\,(B\!\cdot\!N)$   (%)")
    ax.set_title("(b) How much of the budget the labels can absorb")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-3, 110)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "fig1-density-distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════
def figure2() -> Path:
    """Δworst 随 B 的 U 形，标出三个 regime 区间 + 判据的 regime 标签。"""
    bs = _load(TAB / "boundary-sweep.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle(r"Gain in worst-image recall from reallocating a fixed output budget",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    colors = {"o2m": "#d62728", "o2o": "#1f77b4"}
    for head in ("o2m", "o2o"):
        rows = bs.get(head, [])
        ok = [r for r in rows if r.get("N1")]
        deg = [r for r in rows if not r.get("N1")]
        if ok:
            ax.plot([r["B"] for r in ok], [r["d"] for r in ok], "-o", ms=5,
                    lw=2.0, color=colors[head], label=f"{head} ({'one-to-many' if head=='o2m' else 'one-to-one'})")
        if deg:
            ax.plot([r["B"] for r in deg], [r["d"] for r in deg], "x", ms=6,
                    color=colors[head], alpha=0.35,
                    label=f"{head}, not spendable / degenerate (excluded)")

    ax.axhline(0, color="k", lw=1.0)
    ax.axhspan(5, 13, color="#2ca02c", alpha=0.07)
    ax.text(2.3, 11.6, "budget binds\n(gain available)", fontsize=8, color="#1a6b1a")
    ax.text(300, 1.0, "detector binds\n(gain exactly 0)", fontsize=8, color="#7a3b00")

    # 分界竖线：B=70 起为 DETECTOR_BOUND
    ax.axvline(60, color="gray", lw=1.0, ls="--")
    ax.text(63, 9.6, r"$B \geq 70$: worst image reaches $\min(n_i,p_i)$",
            fontsize=8, color="gray", rotation=0)

    ax.set_xscale("log")
    ax.set_xlabel("output budget $B$ (boxes per image, log scale)")
    ax.set_ylabel(r"$\Delta$ worst-image recall  (percentage points)")
    ax.set_title(r"(a) $\Delta$ vs budget, both prediction heads")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="center left", framealpha=0.95)
    ax.set_ylim(-3.4, 13.2)

    # ── (b) 判据的两侧对比（示意，用真实数值）───────────────
    ax = axes[1]
    labels = ["$B=25$\nbudget binds", "$B=70$\ndetector binds"]
    uni = [8.202, 25.000]
    wat = [19.355, 25.000]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, uni, w, label="uniform quota", color="#7f7f7f")
    ax.bar(x + w / 2, wat, w, label="proportional", color="#d62728")
    for xi, (a, b) in enumerate(zip(uni, wat)):
        ax.text(xi - w / 2, a + 0.6, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.6, f"{b:.2f}", ha="center", fontsize=8)
        d = b - a
        ax.annotate(f"$\\Delta$ = {d:+.2f} pt", xy=(xi, max(a, b) + 3.2),
                    ha="center", fontsize=9, fontweight="bold",
                    color="#1a6b1a" if d > 1 else "#7a3b00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("worst-image recall (%)")
    ax.set_title("(b) The criterion's two regimes")
    ax.set_ylim(0, 34)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "fig2-delta-vs-budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="论文正文图")
    ap.add_argument("--only", type=int, choices=[1, 2], default=None)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    todo = [1, 2] if a.only is None else [a.only]
    for i in todo:
        p = figure1() if i == 1 else figure2()
        print(f"✅ Figure {i} → {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
