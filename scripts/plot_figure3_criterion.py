"""
scripts/plot_figure3_criterion.py —— Figure 3：判据示意（逐图取数）

═══════════════════════════════════════════════════════════════════════════
这张图要说清什么
═══════════════════════════════════════════════════════════════════════════
判据的全部内容是：**看最坏那张图的配额是否已触及 `capᵢ = min(nᵢ, pᵢ)`。**

    (a) **B = 25（预算卡住）**：均匀配额把最密的那张图（n=317）钉成最坏（8.20%）；
        重分配把配额挪给它，最坏图**换人** → Δ = **+11.15 pt**
    (b) **B = 70（检测器卡住）**：两种分配都让同一张图拿到 `q = n = 12`（**到顶**），
        它召回 3/12 = **25.000%** → Δ = **0.00 pt**

**画法**：散点 = 每张图（x = `nᵢ`，y = 该图在两个分配下的召回），
用**箭头/连线**显示它是被"移动"还是"没被移动"；并用一条 `capᵢ` 参考线标出到顶位置。
最坏图用大号标记 + 注释。

数据：`runs/budget/r1_val_{head}.json` 的 `per_image`（含 `tp_prefix`）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.allocation_regime import (  # noqa: E402
    cap_of, plan_uniform_eq, plan_waterfill_eq, recall_of, spend_of, topup,
)

FIG = ROOT / "results" / "figures"
HEAD = "o2m"
CASES = [(25, "budget binds", "#1a6b1a"), (70, "detector binds", "#7a3b00")]


def load(head: str) -> list[dict]:
    obj = json.loads((ROOT / "runs" / "budget" / f"r1_val_{head}.json").read_text(encoding="utf-8"))
    return obj["per_image"]


def main() -> int:
    recs = load(HEAD)
    n, p, cap = cap_of(recs)
    N = len(recs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("The criterion: does the worst image's quota reach $\\min(n_i,\\,p_i)$?",
                 fontsize=13, fontweight="bold")

    for ax, (B, tag, col) in zip(axes, CASES):
        S = float(B * N)
        q_u = topup(recs, plan_uniform_eq(recs, S), S)
        q_w = topup(recs, plan_waterfill_eq(recs, S), S)
        sp_u, sp_w = spend_of(recs, q_u), spend_of(recs, q_w)
        r_u, r_w = recall_of(recs, q_u), recall_of(recs, q_w)

        iu = int(np.nanargmin(r_u))
        iw = int(np.nanargmin(r_w))

        # ── 全部图：淡色散点（均匀 vs 水位用同色系的两种深浅）────
        ax.scatter(n, r_u * 100, s=12, c="#9e9e9e", alpha=0.45,
                   label="uniform quota", zorder=2, linewidths=0)
        ax.scatter(n, r_w * 100, s=12, c="#d62728", alpha=0.45,
                   label="proportional", zorder=3, linewidths=0)

        # ── 两图之间的移动（只画变化明显的，避免糊成一片）────
        moved = np.abs(r_w - r_u) > 0.02
        ax.vlines(n[moved], r_u[moved] * 100, r_w[moved] * 100,
                  color="#bdbdbd", lw=0.5, alpha=0.55, zorder=1)

        # ── 最坏图高亮 ──────────────────────────────────────────
        for idx, r, sp, mk in ((iu, r_u, sp_u, "o"), (iw, r_w, sp_w, "s")):
            at_cap = sp[idx] >= cap[idx] - 1e-9
            ax.scatter([n[idx]], [r[idx] * 100], s=190, marker=mk,
                       facecolors="none", edgecolors=col if at_cap else "#000000",
                       linewidths=2.6, zorder=6)
            note = (f"worst image\n$n_i$={int(n[idx])}, quota={int(sp[idx])}"
                    f"{'  = $n_i$ (at cap)' if at_cap else ''}\nrecall={r[idx] * 100:.2f}%")
            ax.annotate(note, (n[idx], r[idx] * 100),
                        textcoords="offset points",
                        xytext=(26, 26 if mk == "o" else -58),
                        fontsize=8, color="black", zorder=7,
                        bbox=dict(fc="white", ec=col, alpha=0.95, lw=1.0),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

        d = (r_w[iw] - r_u[iu]) * 100
        ax.set_xscale("log")
        ax.set_xlabel("annotated objects $n_i$  (log scale)")
        ax.set_ylabel("per-image recall (%)")
        ax.set_title(f"(B = {B})  {tag}   →   $\\Delta$ = {d:+.2f} pt",
                     color=col, fontweight="bold")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right", markerscale=1.6)
        ax.set_ylim(-4, 104)

        # ── 到顶事实：报告**数量**而不是把所有图都标出来（否则越读越糊涂）──
        capped_all = (sp_u >= cap - 1e-9) & (sp_w >= cap - 1e-9)
        n_cap = int(capped_all.sum())
        ax.plot([], [], "x", color="#2ca02c", ms=7, mew=1.8, alpha=0.9,
                label=f"both allocations give $q_i \\geq \\min(n_i,p_i)$  ({n_cap}/{N} imgs)")
        # 只把**低召回且已到顶**的图标绿 ×，以说明"到顶 ≠ 召回高"
        weak_cap = capped_all & (r_u < 0.45)
        if weak_cap.any():
            ax.scatter(n[weak_cap], r_u[weak_cap] * 100, s=34, marker="x",
                       c="#2ca02c", alpha=0.9, zorder=4, linewidths=1.6)
            ax.annotate(f"{int(weak_cap.sum())} images already at\n$\\min(n_i,p_i)$ yet recall < 45%\n"
                        f"$\\Rightarrow$ the detector, not the budget, is binding",
                        xy=(0.03, 0.04), xycoords="axes fraction", fontsize=7.5,
                        color="#1a6b1a", ha="left",
                        bbox=dict(fc="white", ec="#2ca02c", alpha=0.9, lw=0.8))
        ax.legend(fontsize=7.5, loc="upper right", markerscale=1.4, framealpha=0.95)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig3-criterion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 3 → {out.relative_to(ROOT)}")

    # 控制台摘要（供 caption 引用，且与手稿一致）
    for B, tag, _ in CASES:
        S = float(B * N)
        q_u = topup(recs, plan_uniform_eq(recs, S), S)
        q_w = topup(recs, plan_waterfill_eq(recs, S), S)
        sp_u, sp_w = spend_of(recs, q_u), spend_of(recs, q_w)
        r_u, r_w = recall_of(recs, q_u), recall_of(recs, q_w)
        iu, iw = int(np.nanargmin(r_u)), int(np.nanargmin(r_w))
        print(f"   B={B:<4} 均匀最坏 idx={iu:<4} n={int(n[iu]):<4} q={int(sp_u[iu]):<4} "
              f"r={r_u[iu] * 100:>7.3f}%   水位最坏 idx={iw:<4} n={int(n[iw]):<4} "
              f"q={int(sp_w[iw]):<4} r={r_w[iw] * 100:>7.3f}%   "
              f"Δ={((r_w[iw] - r_u[iu]) * 100):+.3f} pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
