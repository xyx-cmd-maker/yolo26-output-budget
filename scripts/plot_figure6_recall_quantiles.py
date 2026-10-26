"""
scripts/plot_figure6_recall_quantiles.py —— Figure 6：重分配把整条 recall 分布往下尾推

═══════════════════════════════════════════════════════════════════════════
为什么需要这张图
═══════════════════════════════════════════════════════════════════════════
worst-image recall 是 min，一个**极端序统计量**：审稿人最自然的质疑是
"你只改善了一张图的运气，还是整个下尾真的抬起来了？"

实测 `results/tables/percentile-recall.json`：
  Δmin  +4.25 ~ +10.21（10/10 同号）
  Δp5   +5.84 ~ +9.68 （10/10 同号）
  Δp10  +2.92 ~ +7.70 （10/10 同号）
  Δp25  +0.41 ~ +4.26 （10/10 同号）
  Δmed  −3.50 ~ −2.31 （0/10 同号）
⇒ 增益是**下尾整体的位移**，代价落在中位数。这张图把这个梯度画出来。

用法：
    python scripts/plot_figure6_recall_quantiles.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "results" / "figures"
OUT_TABLE = ROOT / "results" / "tables" / "percentile-recall.json"

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

W_DOUBLE = 7.2
PT = {"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 6,
      "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5,
      "xtick.major.size": 2, "ytick.major.size": 2}


def _rdr():
    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def recalls(rdr, recs, q):
    n, p, _ = rdr._cap_of(recs)
    q_eff = np.minimum(q, np.minimum(n, p))
    tp = np.array([rdr.tp_at(r["tp_prefix"], int(round(x))) for r, x in zip(recs, q_eff)])
    return tp / np.maximum(n, 1)


def main() -> None:
    rdr = _rdr()
    plt.rcParams.update(PT)
    palette = {"uniform": OI["sky"], "proportional": OI["vermillion"]}

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.7))
    summary = []

    for ax, head, tag in zip(axes, ("o2m", "o2o"), ("a", "b")):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        n, _, _ = rdr._cap_of(recs)
        S = 0.50 * float(n.sum())
        qu = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
        qw = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
        ru, rw = recalls(rdr, recs, qu), recalls(rdr, recs, qw)

        xs = np.linspace(0, 1, 501)
        qu_curve = np.quantile(ru, xs) * 100
        qw_curve = np.quantile(rw, xs) * 100
        ax.plot(xs * 100, qu_curve, lw=1.1, color=palette["uniform"], label="uniform")
        ax.plot(xs * 100, qw_curve, lw=1.1, color=palette["proportional"],
                label="proportional")

        # 阴影：只在 proportional 高于 uniform 的区间（下尾）
        # ★ EPS 不支持透明度：用**浅实色**填充，打印与 EPS 均正确
        LIGHT = {"uniform": "#D6E9F8", "proportional": "#FBE0D0"}
        ax.fill_between(xs * 100, qu_curve, qw_curve,
                        where=qw_curve >= qu_curve,
                        color=LIGHT["proportional"], lw=0, interpolate=True)
        ax.fill_between(xs * 100, qu_curve, qw_curve,
                        where=qw_curve < qu_curve,
                        color=LIGHT["uniform"], lw=0, interpolate=True)

        for qv, lab in ((0.0, "min"), (0.05, "p5"), (0.10, "p10")):
            x = max(qv * 100, 0.6)
            ax.axvline(x, color="0.75", lw=0.4, ls=":", zorder=0)

        ax.set_xlabel("Per-image recall percentile")
        ax.set_ylabel("Recall (%)" if head == "o2m" else "")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.text(0.015, 0.965, tag, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
        title = "one-to-many" if head == "o2m" else "one-to-one"
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.legend(loc="lower right", frameon=False, handlelength=1.6)
        ax.spines[["top", "right"]].set_visible(False)

        summary.append({
            "head": head,
            "d_min": round((rw.min() - ru.min()) * 100, 3),
            "d_p5": round((np.quantile(rw, 0.05) - np.quantile(ru, 0.05)) * 100, 3),
            "d_p10": round((np.quantile(rw, 0.10) - np.quantile(ru, 0.10)) * 100, 3),
            "d_median": round((np.median(rw) - np.median(ru)) * 100, 3),
        })

    fig.tight_layout(pad=0.4, w_pad=1.4)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf", "png"):
        fig.savefig(OUT_FIG / f"fig6-recall-quantiles.{ext}", dpi=600,
                    bbox_inches="tight")
    # 灰度版（打印可读性）
    plt.close(fig)

    print("  Figure 6 已写出（eps/pdf/png）")
    for s in summary:
        print(f"    {s['head']}: Δmin={s['d_min']:+.2f} Δp5={s['d_p5']:+.2f} "
              f"Δp10={s['d_p10']:+.2f} Δmedian={s['d_median']:+.2f}")

    # 与 percentile-recall.json 对账
    if OUT_TABLE.exists():
        tbl = json.loads(OUT_TABLE.read_text(encoding="utf-8"))
        r50 = [r for r in tbl["rows"] if abs(r["rho"] - 0.50) < 1e-9]
        for s in summary:
            match = next(r for r in r50 if r["head"] == s["head"])
            same = (abs(match["d_p5"] - s["d_p5"]) < 0.01
                    and abs(match["d_p10"] - s["d_p10"]) < 0.01)
            print(f"    {s['head']} 与 percentile-recall.json 一致: {same}")


if __name__ == "__main__":
    main()
