"""
scripts/plot_figure4_quintile_cost.py —— Figure 4：分位代价（带 bootstrap 置信区间）

═══════════════════════════════════════════════════════════════════════════
依据技能 `scientific-visualization` 的 Final Checklist（逐条落实）
═══════════════════════════════════════════════════════════════════════════
  [x] 分辨率 600 dpi（combination art）**且**同时出矢量 PDF/EPS
  [x] 尺寸：Nature **双栏 183 mm = 7.2 in**
  [x] 字号：轴标签 8 pt、tick 6 pt、panel 标签 9 pt 粗体（≥6 pt 要求）
  [x] 配色：**Okabe-Ito**（色盲安全），并加**冗余编码**（标记形状 + 颜色双向编码净收益符号）
  [x] 误差棒存在，且**在图注中写明是 95% 置信区间与 n**
  [x] 灰度可读（--grayscale 出灰度版核对）
  [x] 无 3D、无 chart junk、去掉上/右边框
  [x] 轴标签含单位（"Change in recall (percentage points)"）
  [x] **不用 JPEG**

═══════════════════════════════════════════════════════════════════════════
为什么这张图对投稿重要
═══════════════════════════════════════════════════════════════════════════
论文最有冲击力的代价数字是"最稀疏 5% 掉 24.2 pt"，但它只是 **27 张图**的均值。
审稿人第一个问题必然是"27 张图的区间有多宽"。
本图把**区间**画在**同一个轴上**，让读者直接看到：
  · 每条区间都**远离 0**（代价确凿，不是抽样噪声）；
  · 最稀疏带的区间最宽（12.1 pt）——**主动暴露**弱点，比等着被问更好。

用法：
    python scripts/plot_figure4_quintile_cost.py
    python scripts/plot_figure4_quintile_cost.py --grayscale
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
CI = TAB / "quintile-ci.json"

# ── Okabe-Ito（技能指定）──────────────────────────────────────────────
OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7", "black": "#000000", "grey": "#666666"}
MM = 1 / 25.4
W_DOUBLE = 183 * MM


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "font.size": 7, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 8, "axes.titlesize": 8,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "black", "axes.axisbelow": True, "axes.grid": False,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "lines.markeredgewidth": 0.5,
        "legend.fontsize": 6, "legend.frameon": False,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem: str, grayscale: bool = False) -> list[Path]:
    out = []
    for ext in ("pdf", "eps", "png"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight")
        out.append(p)
    if grayscale:
        # ★ 真的做灰度转换（技能 Final Checklist：test figures in grayscale）。
        #   第 54 轮教训（P50）：早先这行只是 `fig.savefig(...)` 又存了一遍**彩色图**，
        #   于是"灰度可读性检查"从未真正发生（两个文件 sha256 完全相同即证据）。
        from PIL import Image
        p_col = FIG / f"{stem}.png"
        p_gray = FIG / f"{stem}-grayscale.png"
        Image.open(p_col).convert("L").save(p_gray, bbox_inches="tight")
        out.append(p_gray)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 4：分位代价 + CI")
    ap.add_argument("--grayscale", action="store_true", help="额外出灰度版做可读性检查")
    a = ap.parse_args()

    if not CI.exists():
        print(f"❌ 缺 {CI}，先跑 scripts/bootstrap_quintile_ci.py")
        return 1
    d_all = json.loads(CI.read_text(encoding="utf-8"))
    # ★ 第 54 轮 P56：quintile-ci.json 现在有**种子维度**（seeds → exp → heads → head）。
    #   本图只画 seed 42（手稿 Table 3 的默认种子）；三种子替换后若要改，只需改这里。
    EXP = "r1"
    if "seeds" in d_all:
        d = d_all["seeds"][EXP]["heads"]
    else:                                     # 兼容旧结构
        d = d_all["heads"]
    style()

    bands = [b["band"] for b in d["o2m"]["bands"]]
    labels = ["sparsest 5%", "5–25%", "25–50%", "50–75%", "densest 25%"]
    y = np.arange(len(bands))[::-1]          # 最稀疏在最上

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.30})
    axA, axB = axes

    # ══════════════════════════════════════════════════════════════
    # Panel a —— Δ（带 95% CI），两个头
    # ══════════════════════════════════════════════════════════════
    # ★ 颜色必须与**项目既有约定**一致（Figure 2 的分配图与项目文档都用：
    #   one-to-many = vermillion，one-to-one = blue）。
    #   第 54 轮（P51）：我初版写反了（o2m=blue/o2o=vermillion），
    #   同一篇论文里同一个头换颜色会让读者以为换了对象。
    off = 0.16
    for k, (head, mk, col, lab) in enumerate((
            ("o2m", "o", OI["vermillion"], "one-to-many"),
            ("o2o", "s", OI["blue"], "one-to-one"))):
        rows = d[head]["bands"]
        dv = np.array([r["delta"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        yy = y + (off if k == 0 else -off)
        err = np.vstack([dv - lo, hi - dv])
        axA.errorbar(dv, yy, xerr=err, fmt=mk, color=col, ecolor=col,
                     elinewidth=0.9, capsize=2.0, capthick=0.7,
                     markeredgecolor="white", markeredgewidth=0.4,
                     label=lab, zorder=3)

    axA.axvline(0.0, color=OI["black"], lw=0.8, ls="-", zorder=1)
    axA.set_yticks(y)
    axA.set_yticklabels([f"{l}\n(n = {r['n_images']})"
                         for l, r in zip(labels, d["o2m"]["bands"])])
    axA.set_xlabel("Change in per-image recall (percentage points)")
    axA.set_xlim(-34, 17)
    axA.set_ylim(-0.75, len(bands) - 0.15)
    axA.legend(loc="lower left", bbox_to_anchor=(0.015, 0.015),
               handletextpad=0.4, borderpad=0.0)
    # panel 标签放在**左上角图外**，避免与注释打架
    axA.text(-0.34, 1.06, "a", transform=axA.transAxes, fontsize=9,
             fontweight="bold", va="bottom", ha="left")

    # 注释：放在最稀疏带**上方**（该带之上没有数据），避免与 panel 标签重叠
    sp = d["o2m"]["bands"][0]
    axA.annotate("sparsest band holds only 27 images:\ninterval spans 12.1 pt but excludes zero",
                 xy=(sp["ci_lo"], y[0] + off),
                 xytext=(-33.0, y[0] + 0.42), fontsize=5.6, color=OI["grey"],
                 ha="left", va="bottom", linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["grey"],
                                 shrinkA=0, shrinkB=1))

    # ══════════════════════════════════════════════════════════════
    # Panel b —— 两种分配的绝对水平
    # ══════════════════════════════════════════════════════════════
    u = np.array([r["uniform"] for r in d["o2m"]["bands"]])
    w = np.array([r["proportional"] for r in d["o2m"]["bands"]])
    axB.plot(u, y, marker="o", color="0.55", label="uniform", zorder=3,
             markeredgecolor="white", markeredgewidth=0.4)
    axB.plot(w, y, marker="^", color=OI["vermillion"], label="proportional", zorder=3,
             markeredgecolor="white", markeredgewidth=0.4)
    # 用细线把同一带的两个点连起来（冗余编码：方向即代价）
    for k in range(len(y)):
        axB.plot([u[k], w[k]], [y[k], y[k]], color=OI["grey"], lw=0.5,
                 ls=(0, (2, 1.5)), zorder=2)
    axB.set_yticks(y)
    axB.set_yticklabels([])
    axB.set_xlabel("Mean per-image recall (%)")
    axB.set_xlim(20, 84)
    axB.set_ylim(-0.75, len(bands) - 0.15)
    axB.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02),
               handletextpad=0.4, borderpad=0.0)
    axB.text(-0.06, 1.06, "b", transform=axB.transAxes, fontsize=9,
             fontweight="bold", va="bottom", ha="left")
    axB.text(0.985, 0.30, "ρ = 0.5\nVisDrone val, N = 548",
             transform=axB.transAxes, fontsize=5.8, color=OI["grey"],
             ha="right", va="bottom", linespacing=1.35)

    outs = save(fig, "fig4-quintile-cost", grayscale=a.grayscale)
    plt.close(fig)
    for p in outs:
        print(f"  ✅ {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
