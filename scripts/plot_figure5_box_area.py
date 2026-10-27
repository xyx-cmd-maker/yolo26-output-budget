"""
scripts/plot_figure5_box_area.py —— Figure 5：框面积与图像物体数的关系

═══════════════════════════════════════════════════════════════════════════
为什么需要这张图（第 54 轮 P67）
═══════════════════════════════════════════════════════════════════════════
`docs/manuscript-results-methods.md` 正文报告了：稀疏图的框**更大**
（稀疏 5% 平均框面积 2.3139e-2 vs 密集 25% 1.1780e-3，Spearman = −0.4594）。
这是"**为什么最稀疏的图损失 24 分**"的机制证据，**但此前只有数字、没有图**。
绘图审计（`.probe/audit_figures.py`）把这一条列为覆盖缺口，本脚本补上。

同时把该统计量**落盘为可交付数据** `results/tables/box-area-vs-count.json`，
此前只在 `.probe/verify_box_area.py` 里复算过、未落盘 → 无法被溯源脚本核验。

用法：
    python scripts/plot_figure5_box_area.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "data" / "processed" / "visdrone" / "labels" / "val"
OUT_TABLE = ROOT / "results" / "tables" / "box-area-vs-count.json"
OUT_FIG = ROOT / "results" / "figures"

# Okabe-Ito 色盲友好配色（全项目统一）
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

W_DOUBLE = 7.2       # Nature 双栏 183 mm
PT = {"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 6,
      "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5,
      "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42}


def read_counts() -> tuple[np.ndarray, np.ndarray]:
    """读 VisDrone 验证集标签，返回 (每图物体数, 每图平均归一化框面积)。

    ★ 格式是**已转为 YOLO 的归一化格式**（空格分隔 `class cx cy w h`），
      不是 VisDrone 原始格式（逗号分隔 `x y w h score cat ...`）。
      我第一版按原始格式解析 → **读到 0 张图**。
      用 `w*h`（归一化面积，量纲 = 图像面积占比），
      这样**不需要引入未核验的图像尺寸**，且与模型实际训练所见一致。
    """
    if not LABELS.exists():
        raise SystemExit(f"❌ 找不到标签目录：{LABELS}")
    counts, areas = [], []
    files = sorted(LABELS.glob("*.txt"))
    for f in files:
        n, a = 0, []
        for ln in f.read_text(encoding="utf-8").split("\n"):
            parts = ln.split()                       # ★ 空格分隔
            if len(parts) < 5:
                continue
            try:
                w, h = float(parts[3]), float(parts[4])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            n += 1
            a.append(w * h)
        if n:
            counts.append(n)
            areas.append(float(np.mean(a)))
    return np.asarray(counts, float), np.asarray(areas, float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ = 平均秩上的 Pearson 相关（scipy 未安装，手算）。"""
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        # 并列取平均秩
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c > 1:
                r[inv == i] = r[inv == i].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main() -> int:
    plt.rcParams.update(PT)
    n, area = read_counts()
    print(f"  读入 {len(n)} 张图（VisDrone val 官方 548）")
    rho = spearman(n, area)

    # ★★ 分带必须**与手稿和 Figure 4 一致**：最稀疏 **5%** vs 最密集 **25%**。
    #   我第一版用 20%/20% 五分位 → 得到 9.30e-3 / 1.16e-3（比值 8.03×），
    #   **与手稿的 2.3139e-2 / 1.1794e-3（19.62×）不一致**。
    #   独立复算确认手稿值正确（Σn = 38,759 与项目已知总量一致）。
    order = np.argsort(n, kind="stable")
    n_s, a_s = n[order], area[order]
    k5 = max(1, int(round(0.05 * len(n_s))))
    k25 = max(1, int(round(0.25 * len(n_s))))
    sparsest = float(a_s[:k5].mean())
    densest = float(a_s[-k25:].mean())
    ratio = sparsest / densest if densest else float("nan")
    # ★★ P96：**分带必须与 Table 3 / Figure 4 完全一致**（5% / 20% / 25% / 25% / 25%）。
    #   此前用 `np.array_split(arange(548), 5)` 即**等张数五等分**（110/110/110/109/109），
    #   而 Table 3 用的是分级带（27/110/137/137/137）。两者是**不同分带**，
    #   但 Figure 5 图注声称 "the same bands used in Figure 4" —— 图文不符。
    N = len(n_s)
    _cuts = [0,
             max(1, int(round(0.05 * N))),
             max(2, int(round(0.25 * N))),
             max(3, int(round(0.50 * N))),
             max(4, int(round(0.75 * N))),
             N]
    bands = []
    for i in range(5):
        lo_i, hi_i = _cuts[i], _cuts[i + 1]
        idx = np.arange(lo_i, hi_i)
        bands.append({"quintile": i + 1, "n_images": int(len(idx)),
                      "n_range": [float(n_s[idx].min()), float(n_s[idx].max())],
                      "mean_n": float(n_s[idx].mean()),
                      "mean_box_area_norm": float(a_s[idx].mean())})
    print(f"  Spearman(n_i, 平均框面积) = {rho:+.4f}")
    print(f"  最稀疏 5%  ({k5} 图, n_i {n_s[:k5].min():.0f}–{n_s[:k5].max():.0f})"
          f" 平均框面积 = {sparsest:.4e}")
    print(f"  最密集 25% ({k25} 图, n_i {n_s[-k25:].min():.0f}–{n_s[-k25:].max():.0f})"
          f" 平均框面积 = {densest:.4e}")
    print(f"  → 比值 {ratio:.2f}×（手稿：2.3139e-2 / 1.1794e-3 = 19.62×）")

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(json.dumps({
        "source": "data/processed/visdrone/labels/val (VisDrone2019-DET validation)",
        "n_images": int(len(n)),
        "area_unit": "normalised (YOLO w*h as fraction of image area; image size not used)",
        "spearman_n_vs_mean_box_area": round(rho, 4),
        "sparsest5_mean_box_area_norm": round(sparsest, 6),
        "densest25_mean_box_area_norm": round(densest, 6),
        "quintiles": [{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in b.items()} for b in bands],
        "sparsest5_over_densest25_area_ratio": round(ratio, 4),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ 已写出 {OUT_TABLE.relative_to(ROOT)}")

    # ── 绘图：两个面板 ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.7))

    # a：散点 + 分箱中位数
    ax = axes[0]
    ax.scatter(n, area, s=2.0, c="0.62", alpha=0.45, linewidths=0,
               label="one image")
    bin_edges = np.array([1, 5, 10, 20, 40, 80, 160, 320])
    cx, cy = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (n >= lo) & (n < hi)
        if m.sum() >= 3:
            cx.append(np.sqrt(lo * hi))
            cy.append(np.median(area[m]))
    ax.plot(cx, cy, "-o", color=OI["vermillion"], ms=3.0, lw=1.2,
            label="median per count bin")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Annotated objects per image, $n_i$")
    ax.set_ylabel("Mean box area per image\n(fraction of image area)")
    ax.text(0.04, 0.06, f"Spearman $\\rho$ = {rho:+.3f}", transform=ax.transAxes,
            fontsize=6, va="bottom",
            bbox=dict(fc="white", ec="0.6", lw=0.4, alpha=0.95))
    ax.legend(loc="upper right", frameon=False)

    # b：五等分平均框面积（与 Figure 4 同一分带）
    ax = axes[1]
    xs = np.arange(1, 6)
    ys = [b["mean_box_area_norm"] for b in bands]
    ax.bar(xs, ys, 0.62, color=[OI["vermillion"]] + ["0.72"] * 3 + [OI["blue"]],
           edgecolor="black", lw=0.4)
    _ymax0 = max(b["mean_box_area_norm"] for b in bands)
    for x, b in zip(xs, bands):
        ax.text(x, b["mean_box_area_norm"] + _ymax0 * 0.025, f"{b['n_images']}",
                ha="center", va="bottom", fontsize=5.5, zorder=12,
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.4))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(b['n_range'][0])}–{int(b['n_range'][1])}"
                        for b in bands], fontsize=5.5)
    # P103: top bar label (27) was clipped -> relax the y upper limit by 18%
    _ymax = max(b["mean_box_area_norm"] for b in bands)
    ax.set_ylim(0, _ymax * 1.28)
    ax.set_xlabel("Object-count quintile (range of $n_i$)")
    ax.set_ylabel("Mean box area\n(fraction of image area)")
    # 注释框贴到 axes 顶部（0.988，仍在黑色框内），给首个柱的图数标签 27 让出空间
    ax.text(0.03, 0.988, f"sparsest / densest = {ratio:.2f}$\\times$",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(fc="white", ec="0.6", lw=0.4, alpha=0.95))

    for lbl, ax in zip("ab", axes):
        ax.text(-0.16, 1.06, lbl, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
    fig.tight_layout(pad=0.4)

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        p = OUT_FIG / f"fig5-box-area-vs-count.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  ✅ {p.relative_to(ROOT)}")
    # 灰度可读版（与其余四图一致）
    from PIL import Image
    png = OUT_FIG / "fig5-box-area-vs-count.png"
    gray = OUT_FIG / "fig5-box-area-vs-count-grayscale.png"
    Image.open(png).convert("L").save(gray)
    print(f"  ✅ {gray.relative_to(ROOT)}（真实灰度，mode=L）")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
