"""
scripts/plot_paper_figures_nature.py —— 论文正文图（**Nature 规范**）

═══════════════════════════════════════════════════════════════════════════
依据技能 `scientific-visualization`（MIT，已审查）与 `assets/nature.mplstyle`
═══════════════════════════════════════════════════════════════════════════
规范（逐条落实）：
  · 宽度：双栏 **183 mm = 7.2 in**；单栏 89 mm = 3.5 in
  · 字体：sans-serif（Arial → Helvetica → DejaVu Sans 回退）；基准 7 pt，
          axis label 8 pt，tick 6 pt，legend 6 pt
  · 分辨率：**600 dpi**（combination art）；同时出 **PDF 矢量**
  · 配色：**Okabe-Ito**（色盲安全），且**加冗余编码**（线型 + 标记形状）
  · 去上/右边框；panel 标签用小写粗体字母放左上
  · 灰度可读性检查（技能 Final Checklist 要求）
  · **绝不**用 JPEG；不用 jet/rainbow

用法：
    python scripts/plot_paper_figures_nature.py                # 画全部
    python scripts/plot_paper_figures_nature.py --only 1
    python scripts/plot_paper_figures_nature.py --grayscale     # 额外出灰度版做可读性检查
"""

from __future__ import annotations

import argparse
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

# ── Okabe-Ito 色盲安全配色（技能指定）────────────────────────────────
OI = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}

# ── Nature 尺寸（mm → in）─────────────────────────────────────────────
MM = 1 / 25.4
W_DOUBLE = 183 * MM          # 7.20 in
W_SINGLE = 89 * MM           # 3.50 in

# ★★ 第 54 轮 P58：`--grayscale` 此前**从未生效**的两重原因：
#   ① `save()` 里的灰度分支写的是 `fig.savefig(...)`（又存一遍彩色图）；
#   ② CLI 标志 `--grayscale` 解析后**从未传给任何 `save()` 调用**，参数一直是默认 False。
#   用模块级开关统一接上，避免改四个函数签名。
_WANT_GRAYSCALE = False


def apply_nature_style() -> None:
    """按 `assets/nature.mplstyle` 显式设置（不依赖技能目录，避免路径耦合）。"""
    mpl.rcParams.update({
        "figure.dpi": 150,               # 屏幕预览；保存时用 600
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=[
            OI["orange"], OI["skyblue"], OI["green"], OI["yellow"],
            OI["blue"], OI["vermillion"], OI["purple"]]),
        "xtick.major.size": 2.5, "xtick.minor.size": 1.5,
        "xtick.major.width": 0.5, "xtick.minor.width": 0.4,
        "xtick.labelsize": 6, "xtick.direction": "out",
        "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "ytick.major.width": 0.5, "ytick.minor.width": 0.4,
        "ytick.labelsize": 6, "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.4,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.25,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "image.cmap": "viridis",
        "pdf.fonttype": 42,              # ★ 嵌入 TrueType（期刊要求可编辑文本）
        "ps.fonttype": 42,
    })


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Nature 用小写粗体 panel 字母（技能 §5 明确：lowercase for Nature）。"""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def save(fig, stem: str, grayscale: bool | None = None) -> list[Path]:
    """出 PDF + **EPS**（矢量）+ PNG 600 dpi；`grayscale` 为真时额外出**真灰度**版。

    ★ 为什么加 EPS：Sci Rep 官方 submission guidelines 原文
      "you should supply all line art, graphs, charts and schematics in
       **vector format, such as EPS or AI**" —— 官方举例是 EPS/AI，故同时给 EPS。
    ★ 绝不出 JPEG（技能 `publication_guidelines.md` 禁止；JPEG 会产生压缩伪影）。
    ★ `grayscale=None` 时取模块级 `_WANT_GRAYSCALE`（由 CLI 设置）。
    """
    if grayscale is None:
        grayscale = _WANT_GRAYSCALE
    out = []
    for ext, kw in (("pdf", {}), ("eps", {}), ("png", {"dpi": 600})):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, **kw, bbox_inches="tight")
        out.append(p)
    if grayscale:
        # ★★ 第 54 轮 P58 修：**这里原本是 `fig.savefig(...)`，即又存了一遍彩色图。**
        #    于是"灰度可读性检查"**从未真正发生**（证据：灰度文件与彩色文件 sha256 相同）。
        #    我在 Figure 4 的脚本里修过同一个 bug（P50），**但漏了这个主图脚本** ——
        #    同一类错误在同一项目里出现了两次，说明修一处时必须全局搜索同类。
        #    现在做真的灰度转换：PIL 的 `convert("L")`。
        from PIL import Image
        p_col = FIG / f"{stem}.png"
        p_gray = FIG / f"{stem}-grayscale.png"
        Image.open(p_col).convert("L").save(p_gray)
        out.append(p_gray)
    return out


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════
def figure1(saveit: bool = True) -> plt.Figure:
    """逐图目标数分布：VisDrone / CrowdHuman / COCO。双栏宽。"""
    d = _load(TAB / "n-gt-distributions.json")
    series = [
        ("visdrone_val", "VisDrone-DET val (aerial)", OI["vermillion"], "-", "o"),
        ("crowdhuman_train", "CrowdHuman train (ground)", OI["blue"], "--", "s"),
        ("coco_val2017", "COCO val2017 (ground)", OI["green"], ":", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.6))

    ax = axes[0]
    for key, lab, c, ls, mk in series:
        if key not in d or "hist_edges" not in d[key]["stats"]:
            continue
        st = d[key]["stats"]
        e = np.array(st["hist_edges"], dtype=float)
        cts = np.array(st["hist_counts"], dtype=float)
        nb = len(cts)
        tot = cts.sum()
        right = np.concatenate([e[1:nb], [e[nb - 1] + 1.0]]) if nb > 1 else e[:1]
        ccdf = np.array([cts[i:].sum() for i in range(nb)]) / tot * 100
        ax.step(right, ccdf, where="post", color=c, ls=ls, lw=1.2, label=lab)
        # 少量标记点（冗余编码，灰度也能区分）
        k = max(1, nb // 7)
        ax.plot(right[::k], ccdf[::k], mk, color=c, ms=2.6, mew=0.4, ls="none")
    ax.set_xscale("log")
    ax.set_xlabel("Output budget $B$ (boxes per image)")
    ax.set_ylabel(r"Images with $n_i > B$  (%)")
    # ★ 技能规范：多面板图**一般不设标题**（解释交给 caption）。
    #   但 panel 的身份必须留在图内 → 改用图内彩色标识，去掉 ax.set_title。
    ax.text(0.035, 0.955, "VisDrone", transform=ax.transAxes, fontsize=6,
            color=OI["vermillion"], va="top", fontweight="bold")
    ax.text(0.035, 0.885, "CrowdHuman", transform=ax.transAxes, fontsize=6,
            color=OI["blue"], va="top", fontweight="bold")
    ax.text(0.035, 0.815, "COCO", transform=ax.transAxes, fontsize=6,
            color=OI["green"], va="top", fontweight="bold")
    ax.set_ylim(-3, 105)
    ax.legend(loc="upper right")
    panel_label(ax, "a")

    ax = axes[1]
    for key, lab, c, ls, mk in series:
        if key not in d:
            continue
        Bs = [u["B"] for u in d[key]["util"]]
        ut = [u["utilisation"] * 100 for u in d[key]["util"]]
        ax.plot(Bs, ut, color=c, ls=ls, lw=1.2, marker=mk, ms=2.6, mew=0.4, label=lab)
    ax.axhline(100, color="0.35", lw=0.5, ls="-")
    ax.text(11, 91, "budget fully absorbed", fontsize=5.5, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("Output budget $B$ (boxes per image)")
    ax.set_ylabel(r"Utilisation $\Sigma\min(B,n_i)/(B\,N)$  (%)")
    ax.set_ylim(-3, 110)
    ax.legend(loc="lower left")
    panel_label(ax, "b")

    if saveit:
        for p in save(fig, "fig1-density-distributions"):
            print(f"   → {p.relative_to(ROOT)}")
    return fig


# ══════════════════════════════════════════════════════════════════════
def figure2(saveit: bool = True) -> plt.Figure:
    """Δworst 随 B 的 U 形（双栏宽，含判据两侧对比）。"""
    bs = _load(TAB / "boundary-sweep.json")
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.7),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    for head, lab, c, ls, mk in (("o2m", "one-to-many", OI["vermillion"], "-", "o"),
                                 ("o2o", "one-to-one", OI["blue"], "--", "s")):
        rows = bs.get(head, [])
        ok = [r for r in rows if r.get("N1")]
        deg = [r for r in rows if not r.get("N1")]
        if ok:
            ax.plot([r["B"] for r in ok], [r["d"] for r in ok], color=c, ls=ls,
                    lw=1.2, marker=mk, ms=2.8, mew=0.4, label=lab)
        if deg:
            ax.plot([r["B"] for r in deg], [r["d"] for r in deg], "x", color=c,
                    ms=3.2, mew=0.7, alpha=0.45,
                    label=f"{lab}, not spendable")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(60, color="0.45", lw=0.5, ls="--")
    ax.text(64, 2.4, r"$B \geq 70$", fontsize=6, color="0.35")
    ax.text(2.2, 11.6, "budget binds", fontsize=6, color=OI["green"], fontweight="bold")
    ax.text(210, 1.0, "detector binds", fontsize=6, color="0.35", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Output budget $B$ (boxes per image)")
    ax.set_ylabel(r"$\Delta$ worst-image recall (pp)")
    ax.set_ylim(-3.4, 13.4)
    # ★ 图例放右上空白区（左下会压住 B≈6–10 的负增益段）
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), ncol=1,
              columnspacing=0.6)
    panel_label(ax, "a")

    # ── (b) 判据的两侧：**从落盘结果读，不硬编码** ──────────────
    #    教训（P43/P44 家族）：手抄数字会错。这里读 decision-rule JSON 的 p1/p2。
    ax = axes[1]
    x = np.arange(2)
    w = 0.32
    # ★★ 第 54 轮 P58 修：**取消静默回退。**
    #    原代码在读取失败时把**硬编码的旧数字** `[11.67,11.67] / [20.00,16.67]` 画进图，
    #    且**不报错** —— 于是"数据文件丢了"会表现为"图看起来正常但数字是旧的"。
    #    这比崩溃危险得多：崩溃会被发现，静默的错图可能直接进论文。
    #    现在改为**失败即报错并给出修复指引**。
    dj = _load(TAB / "decision-rule-r1-val.json")
    o2m = next((r for r in dj if "o2m" in r["label"]), None)
    o2o = next((r for r in dj if "o2o" in r["label"]), None)
    if o2m is None or o2o is None:
        raise RuntimeError(
            "Figure 2b: decision-rule-r1-val.json 里找不到 o2m/o2o 记录。"
            "先跑 scripts/run_decision_rule.py --exp r1 --splits val --heads o2m o2o。"
            "（不要回退到硬编码数字：那会让错图静默进论文。）")
    uni = [o2m["p1"]["worst"] * 100, o2o["p1"]["worst"] * 100]
    wat = [max(o2m["p2"]["worst"], o2m["p3"]["worst"]) * 100,
           max(o2o["p2"]["worst"], o2o["p3"]["worst"]) * 100]
    ax.bar(x - w / 2, uni, w, color="0.62", edgecolor="black", lw=0.4, label="uniform")
    ax.bar(x + w / 2, wat, w, color=OI["vermillion"], edgecolor="black", lw=0.4,
           label="proportional")
    for xi, (a, b) in enumerate(zip(uni, wat)):
        ax.text(xi - w / 2, a + 0.7, f"{a:.2f}", ha="center", fontsize=5.5)
        ax.text(xi + w / 2, b + 0.7, f"{b:.2f}", ha="center", fontsize=5.5)
        d = b - a
        ax.text(xi, max(a, b) + 3.4, f"$\\Delta$ = {d:+.2f} pp", ha="center",
                fontsize=6.5, fontweight="bold",
                color=OI["green"] if d > 1 else "0.25")
    ax.set_xticks(x)
    # ★★ 第 54 轮 P58 修：**这两根柱子是同一个 ρ=0.5 条件、两条预测头**，
    #    而原标签写的是 `$B=25$ budget binds` / `$B=70$ detector binds` ——
    #    那是**预算扫描**的两个档位，与本面板的数据**不是一回事**。
    #    （ρ=0.5 的有效每图预算是 S/N = 0.5·mean(nᵢ) ≈ 35.4，既不是 25 也不是 70。）
    #    Figure 2a 的横轴是预算 B；本面板的横轴是**预测头**。两者必须区分清楚，
    #    否则读者会把"同一条件下的两条头"误读成"两个不同预算"。
    ax.set_xticklabels(["one-to-many\n(ρ = 0.5)", "one-to-one\n(ρ = 0.5)"])
    ax.set_ylabel("Worst-image recall (%)")
    ax.set_ylim(0, 34)
    ax.legend(loc="upper left")
    panel_label(ax, "b", dx=-0.20)

    if saveit:
        for p in save(fig, "fig2-delta-vs-budget"):
            print(f"   → {p.relative_to(ROOT)}")
    return fig


# ══════════════════════════════════════════════════════════════════════
def figure3(saveit: bool = True) -> plt.Figure:
    """判据示意：逐图取数，双栏宽。"""
    from src.eval.allocation_regime import (cap_of, plan_uniform_eq, plan_waterfill_eq,
                                            recall_of, spend_of, topup)
    obj = json.loads((ROOT / "runs" / "budget" / "r1_val_o2m.json").read_text(encoding="utf-8"))
    recs = obj["per_image"]
    n, p, cap = cap_of(recs)
    N = len(recs)

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.8))
    for ax, (B, tag, col) in zip(axes, ((25, "budget binds", OI["green"]),
                                        (70, "detector binds", "0.30"))):
        S = float(B * N)
        q_u = topup(recs, plan_uniform_eq(recs, S), S)
        q_w = topup(recs, plan_waterfill_eq(recs, S), S)
        sp_u, sp_w = spend_of(recs, q_u), spend_of(recs, q_w)
        r_u, r_w = recall_of(recs, q_u), recall_of(recs, q_w)
        iu, iw = int(np.nanargmin(r_u)), int(np.nanargmin(r_w))

        ax.scatter(n, r_u * 100, s=2.2, c="0.62", alpha=0.40, linewidths=0,
                   label="uniform")
        ax.scatter(n, r_w * 100, s=2.2, c=OI["vermillion"], alpha=0.40, linewidths=0,
                   label="proportional")
        moved = np.abs(r_w - r_u) > 0.02
        ax.vlines(n[moved], r_u[moved] * 100, r_w[moved] * 100,
                  color="0.80", lw=0.3, alpha=0.5, zorder=1)
        for idx, r, sp, mk in ((iu, r_u, sp_u, "o"), (iw, r_w, sp_w, "s")):
            at_cap = sp[idx] >= cap[idx] - 1e-9
            ax.scatter([n[idx]], [r[idx] * 100], s=80, marker=mk, facecolors="none",
                       edgecolors=col if at_cap else "black", linewidths=1.2, zorder=6)
            ax.annotate(f"$n_i$={int(n[idx])}, quota={int(sp[idx])}"
                        f"{' = $n_i$' if at_cap else ''}\nrecall={r[idx] * 100:.2f}%",
                        (n[idx], r[idx] * 100), textcoords="offset points",
                        xytext=(64, 26) if mk == "o" else (58, -46), fontsize=5.5,
                        bbox=dict(fc="white", ec=col, alpha=0.95, lw=0.5),
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
        d = (r_w[iw] - r_u[iu]) * 100
        capped_all = (sp_u >= cap - 1e-9) & (sp_w >= cap - 1e-9)
        weak = capped_all & (r_u < 0.45)
        if weak.any():
            ax.scatter(n[weak], r_u[weak] * 100, s=6, marker="x", c=OI["green"],
                       alpha=0.85, zorder=4, linewidths=0.7)
            ax.text(0.03, 0.035, f"{int(weak.sum())} images at $\\min(n_i,p_i)$\n"
                                 f"yet recall < 45%", transform=ax.transAxes,
                    fontsize=5.5, color=OI["green"], va="bottom",
                    bbox=dict(fc="white", ec=OI["green"], alpha=0.9, lw=0.4))
        ax.set_xscale("log")
        ax.set_xlabel("Annotated objects $n_i$")
        ax.set_ylabel("Per-image recall (%)")
        # ★ 规范：多面板图不设标题；面板身份用图内彩色标识
        ax.text(0.03, 0.965, f"$B$ = {B}   ({tag})", transform=ax.transAxes,
                fontsize=6.5, fontweight="bold", color=col, va="top")
        ax.text(0.97, 0.965, f"$\\Delta$ = {d:+.2f} pp", transform=ax.transAxes,
                fontsize=6.5, fontweight="bold", color=col, va="top", ha="right")
        ax.set_ylim(-4, 104)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), markerscale=3.5)
        panel_label(ax, "a" if B == 25 else "b")

    if saveit:
        for p in save(fig, "fig3-criterion"):
            print(f"   → {p.relative_to(ROOT)}")
    return fig


def main() -> int:
    global _WANT_GRAYSCALE
    ap = argparse.ArgumentParser(description="论文正文图（Nature 规范）")
    ap.add_argument("--only", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--grayscale", action="store_true", help="额外出灰度版核对可读性")
    a = ap.parse_args()
    _WANT_GRAYSCALE = bool(a.grayscale)      # ★ 接上 CLI 标志（P58：此前从未传递）
    apply_nature_style()
    FIG.mkdir(parents=True, exist_ok=True)
    print("=" * 92)
    print(f"Nature 规范绘图：双栏 {W_DOUBLE:.2f} in (183 mm)，600 dpi，Okabe-Ito 色盲安全")
    print("=" * 92)
    todo = [1, 2, 3] if a.only is None else [a.only]
    for i in todo:
        print(f"\nFigure {i}")
        {1: figure1, 2: figure2, 3: figure3}[i]()
        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
