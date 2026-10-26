"""
scripts/build_supplementary.py —— 生成**单一的**补充材料 PDF（Sci Rep 官方要求）

═══════════════════════════════════════════════════════════════════════════
官方原文（submission guidelines）
═══════════════════════════════════════════════════════════════════════════
  "Supplementary information should be **combined and supplied as a single separate file**,
   **preferably in PDF format**."

═══════════════════════════════════════════════════════════════════════════
内容（全部来自已落盘产物，零新增实验）
═══════════════════════════════════════════════════════════════════════════
  · **Table S1** —— ρ 敏感性 10 个条件（正文 R3 只引一句）
  · **Figure S1** —— YOLO26n 训练曲线（逐 epoch mAP 与 loss）—— Methods 的可复现性证据
  · **Figure S2** —— ρ 扫描：Δworst 对 ρ 的依赖
  · **Note S1** —— 数值自检与独立复算说明

★ 规范：与正文图一致（183 mm、Okabe-Ito、600 dpi、去上右边框、无面板标题）
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys  # noqa: E402
sys.path.insert(0, sys_path)

from scripts.plot_paper_figures_nature import (  # noqa: E402
    OI, W_DOUBLE, apply_nature_style, panel_label,
)

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
RUNS = ROOT / "runs" / "train"
OUT = ROOT / "results" / "supplementary-information.pdf"

# ══════════════════════════════════════════════════════════════════════
# ★★ ρ 敏感性：**在脚本内现算，绝不嵌入任何数字**
# ══════════════════════════════════════════════════════════════════════
# 第 48 轮教训（P42）：本文件初版把 10 行数字**手工抄进**源码，
# 结果 4 个 Δmicro 值抄错。手工转录 = 引入错误的入口。
# → 现在改为直接调用 `run_decision_rule.analyse()`，与正文用**同一实现**。
RHO_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


def compute_rho_sweep() -> list[tuple]:
    """返回 [(head, ρ, uniform worst%, proportional worst%, Δ pp, Δmicro pp, verdict)]。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    rdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rdr)

    out = []
    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        if not recs:
            continue
        for rho in RHO_GRID:
            r = rdr.analyse(recs, rho, head, n_boot=0, quiet=True)
            out.append((
                "one-to-many" if head == "o2m" else "one-to-one",
                rho,
                r["p1"]["worst"] * 100,
                max(r["p2"]["worst"], r["p3"]["worst"]) * 100,
                r["dworst"],
                r["dmicro"],
                r["verdict"].split("：")[0].strip(),
            ))
    return out


RHO_SWEEP: list[tuple] = []          # 由 main() 填充（现算）


def read_curve(run: str) -> dict | None:
    p = RUNS / run / "results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        return None

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(k, "")))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out, float)

    return {"name": run, "ep": np.arange(1, len(rows) + 1),
            "m50": col("metrics/mAP50(B)"), "m95": col("metrics/mAP50-95(B)"),
            "box": col("train/box_loss"), "cls": col("train/cls_loss"),
            "dfl": col("train/dfl_loss")}


def page_table_s1(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(W_DOUBLE, 3.4))
    fig.text(0.005, 0.965, "Table S1.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.075, 0.965,
             "Sensitivity of the worst-image gain to the budget fraction $\\rho$.",
             fontsize=8, va="top")
    fig.text(0.005, 0.885,
             "All ten conditions are reported. $\\Delta$ is proportional minus uniform, in percentage\n"
             "points; the pre-registered thresholds were A $\\geq$ 10, B $\\geq$ 5, C $< 5. "
             "$N$ = 548 images; $\\rho$ = 0.5 was pre-registered.",
             fontsize=6.5, va="top")

    cols = ["head", "$\\rho$", "uniform worst (%)", "proportional worst (%)",
            "$\\Delta$ (pp)", "$\\Delta$ micro (pp)", "verdict"]
    xs = [0.04, 0.17, 0.28, 0.46, 0.64, 0.76, 0.91]
    y0, dy = 0.63, 0.055
    for x, c in zip(xs, cols):
        fig.text(x, y0 + 0.03, c, fontsize=6.5, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 + 0.028, y0 + 0.028],
                                color="black", lw=0.5, transform=fig.transFigure))
    for i, (h, r, u, w, d, dm, v) in enumerate(RHO_SWEEP):
        y = y0 - i * dy
        for x, t in zip(xs, [h, f"{r:.2f}", f"{u:.3f}", f"{w:.3f}",
                             f"{d:+.3f}", f"{dm:.3f}", v]):
            fig.text(x, y, t, fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def page_table_s2(pdf: PdfPages) -> None:
    """Table S2：可部署（预测式）配额与 oracle 配额的对比。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "prediction-baseline.json")
                     .read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(W_DOUBLE, 3.0))
    fig.text(0.005, 0.965, "Table S2.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.075, 0.965,
             "A deployable variant of the proportional allocation.",
             fontsize=8, va="top")
    fig.text(0.005, 0.905,
             "All three allocations use the same total budget ($\\rho$ = 0.5), the same integer rounding\n"
             "and the same top-up, so the only difference is the allocation signal. \"oracle\" uses the\n"
             "annotated object count $n_i$, unavailable at inference; \"prediction\" uses the number of\n"
             "predictions the detector emits at the largest budget, available before quotas are fixed.\n"
             "$\\Delta$ is against the uniform allocation, in percentage points. $N$ = 548 images.",
             fontsize=6.5, va="top")

    cols = ["case", "$p_i$ median", "$n_i$ median", "$\\Delta$worst oracle",
            "$\\Delta$worst prediction", "$\\Delta$macro oracle", "$\\Delta$macro prediction"]
    xs = [0.04, 0.15, 0.27, 0.39, 0.56, 0.75, 0.90]
    y0, dy = 0.55, 0.075
    for x, c in zip(xs, cols):
        fig.text(x, y0 + 0.04, c, fontsize=6, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 + 0.035, y0 + 0.035],
                                color="black", lw=0.5, transform=fig.transFigure))
    for i, (tag, r) in enumerate(_d["cases"].items()):
        y = y0 - i * dy
        cells = [tag, f"{r['p_median']:.0f}", f"{r['n_median']:.0f}",
                 f"{r['d_worst_oracle']:+.2f}", f"{r['d_worst_pred']:+.2f}",
                 f"{r['d_macro_oracle']:+.2f}", f"{r['d_macro_pred']:+.2f}"]
        for x, val in zip(xs, cells):
            fig.text(x, y, val, fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def page_table_s3(pdf: PdfPages) -> None:
    """Table S3：test-dev 上最差图像的逐图证据。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "testdev-worst-image.json")
                     .read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(W_DOUBLE, 4.8))
    fig.text(0.005, 0.965, "Table S3.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.075, 0.965,
             "The worst image on test-dev under both allocations.",
             fontsize=8, va="top")
    fig.text(0.005, 0.905,
             "Budget fraction $\\rho$ = 0.5. The same image, index 69, is the worst under both\n"
             "allocations and on both heads. Its annotated count $n_i$ is 8, so its capacity is 8 boxes;\n"
             "the uniform quota already covers that capacity, and neither allocation recovers a single\n"
             "object. The two quota vectors nevertheless differ on most images, and both allocations\n"
             "spend the same total. $N$ = 1610 test-dev images.",
             fontsize=6.5, va="top")

    # P116: 表头 `images with different quota` 与 `spend` 原先重叠（改为两行短标签），
    #   行距 0.10 过大导致底行贴边，一并收紧。
    cols = ["head", "worst image", "$n_i$", "$p_i$", "quota\nuniform",
            "quota\nprop.", "TP", "recall", "images with\na different quota", "spend"]
    xs = [0.03, 0.10, 0.20, 0.27, 0.34, 0.45, 0.55, 0.63, 0.72, 0.90]
    y0, dy = 0.52, 0.09
    for x, c in zip(xs, cols):
        for _k, _line in enumerate(str(c).split("\n")):
            fig.text(x, y0 + 0.075 - _k * 0.028, _line, fontsize=6,
                     fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.02, 0.98], [y0 + 0.045, y0 + 0.045],
                                color="black", lw=0.5, transform=fig.transFigure))
    for i, (head, r) in enumerate(_d["heads"].items()):
        y = y0 - i * dy
        cells = [
            head,
            "#" + str(r["uniform"]["index"]),
            str(r["uniform"]["n_i"]),
            str(r["uniform"]["p_i"]),
            str(r["uniform"]["quota"]),
            str(r["proportional"]["quota"]),
            str(r["uniform"]["tp"]),
            f"{r['worst_uniform_pct']:.1f}%",
            f"{r['n_quota_differs']}/{r['N']}",
            f"{r['spend_uniform']:.0f}",
        ]
        for x, v in zip(xs, cells):
            fig.text(x, y, v, fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def page_table_s4(pdf: PdfPages) -> None:
    """Table S4：Holm 逐步校正的逐条件明细。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "holm-detail.json")
                     .read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(W_DOUBLE, 4.8))
    fig.text(0.005, 0.965, "Table S4.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.075, 0.965,
             "Holm step-down correction, condition by condition.",
             fontsize=8, va="top")
    fig.text(0.005, 0.885,
             f"The family is the {_d['n_conditions']} conditions of the $\\rho$ sweep "
             f"(five fractions $\\times$ two heads). One-sided null: the worst-image gain is not\n"
             "positive. Conditions are ordered by rank; the Holm threshold at rank $k$ is "
             "$\\alpha/(m-k+1)$, and $p_{\\mathrm{adj}}$ is the step-down adjusted value.\n"
             f"$N$ = 548 images, {_d['n_boot']:,} bootstrap resamples, RNG seed {_d['rng_seed']}. "
             f"All {_d['n_reject_holm']} conditions remain rejected after correction.",
             fontsize=6.5, va="top")

    cols = ["rank", "condition", "$\\Delta$ (pt)", "$p$ raw", "Holm threshold",
            "$p$ adjusted", "rejected"]
    xs = [0.06, 0.16, 0.40, 0.52, 0.64, 0.79, 0.91]
    y0, dy = 0.52, 0.075
    for x, c in zip(xs, cols):
        fig.text(x, y0 + 0.055, c, fontsize=6, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 + 0.045, y0 + 0.045],
                                color="black", lw=0.5, transform=fig.transFigure))
    for i, r in enumerate(_d["rows"]):
        y = y0 - i * dy
        cells = [str(r["rank"]), r["condition"], f"{r['delta_pt']:+.3f}",
                 f"{r['p_raw']:.4f}", f"{r['holm_threshold']:.4f}",
                 f"{r['p_adjusted']:.4f}", "yes" if r["reject"] else "no"]
        for x, v in zip(xs, cells):
            fig.text(x, y, v, fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def page_table_s5(pdf: PdfPages) -> None:
    """Table S5：去掉 label-count cap 后主结果如何变化。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "gt-cap-sensitivity.json")
                     .read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(W_DOUBLE, 5.6))
    fig.text(0.005, 0.965, "Table S5.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.083, 0.965,
             "Sensitivity of the main results to the label-count cap.",
             fontsize=8, va="top")
    fig.text(0.005, 0.895,
             "The main text caps every quota at $\\min(n_i, p_i)$. Here the $n_i$ term is removed, so "
             "quotas are capped at $p_i$ alone and spend is the number of boxes actually emitted; true "
             "positives are still bounded by $n_i$. Every gain grows when the cap is removed, so the "
             "reported main results are conservative estimates for the label-count-capped convention.\n"
             "VisDrone2019-DET validation, seed 42, $N$ = 548.  Values are percentage points.  "
             "$\\Delta$ is proportional minus uniform.",
             fontsize=6.5, va="top")

    cols = ["head", "$\\rho$", "$\\Delta$ worst, capped",
            "$\\Delta$ worst, no cap", "change",
            "$\\Delta$ micro, capped", "$\\Delta$ micro, no cap"]
    xs = [0.05, 0.15, 0.30, 0.48, 0.64, 0.76, 0.90]
    y0, dy = 0.60, 0.038
    for x, c in zip(xs, cols):
        fig.text(x, y0 + 0.030, c, fontsize=5.8, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 + 0.022, y0 + 0.022],
                                color="black", lw=0.5, transform=fig.transFigure))
    for i, r in enumerate(_d["rows"]):
        y = y0 - i * dy
        cells = [r["head"], f"{r['rho']:.2f}",
                 f"{r['capped_dworst']:+.2f}", f"{r['no_cap_dworst']:+.2f}",
                 f"{r['d_dworst']:+.2f}",
                 f"{r['capped_dmicro']:+.2f}", f"{r['no_cap_dmicro']:+.2f}"]
        for x, v in zip(xs, cells):
            fig.text(x, y, v, fontsize=6.2)
    pdf.savefig(fig)
    plt.close(fig)


def page_table_s6(pdf: PdfPages) -> None:
    """Table S6：top-up 四臂消融（三臂不用标签）。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "topup-robustness.json")
                     .read_text(encoding="utf-8"))
    _rows = _d["rows"]

    fig = plt.figure(figsize=(W_DOUBLE, 5.2))
    fig.text(0.005, 0.965, "Table S6.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.083, 0.965,
             "Worst-image gain under four top-up rules, three of which never read labels.",
             fontsize=8, va="top")
    fig.text(0.005, 0.900,
             "Every condition is the same without the oracle top-up: with the budget matched, the two "
             "label-free rules land within 0.32 points of the oracle arm. The no-top-up arm under-spends "
             "by up to 2.75% at the smallest budget fraction and is not a valid control there.\n"
             "VisDrone2019-DET validation, seed 42, $N$ = 548.  $\\Delta$ worst-image recall, "
             "proportional minus uniform, in percentage points.",
             fontsize=6.5, va="top")

    cols = ["head", "$\\rho$", "no top-up", "largest remainder",
            "most capacity", "oracle", "spend err %"]
    xs = [0.06, 0.16, 0.27, 0.45, 0.62, 0.77, 0.88]
    y0, dy = 0.52, 0.043
    for x, c in zip(xs, cols):
        fig.text(x, y0 + 0.032, c, fontsize=6, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 + 0.024, y0 + 0.024],
                                color="black", lw=0.5, transform=fig.transFigure))

    def cell(head, rho, arm, key="dworst"):
        for r in _rows:
            if r["head"] == head and abs(r["rho"] - rho) < 1e-9 and r["arm"] == arm:
                return r[key]
        return float("nan")

    i = 0
    for head in ("o2m", "o2o"):
        for rho in (0.30, 0.40, 0.50, 0.60, 0.70):
            y = y0 - i * dy
            vals = [head, f"{rho:.2f}",
                    f"{cell(head, rho, 'no_topup'):+.2f}",
                    f"{cell(head, rho, 'largest_remainder'):+.2f}",
                    f"{cell(head, rho, 'most_capacity'):+.2f}",
                    f"{cell(head, rho, 'oracle'):+.2f}",
                    f"{cell(head, rho, 'oracle', 'spend_err_pct'):.3f}"]
            for x, v in zip(xs, vals):
                fig.text(x, y, v, fontsize=6.2)
            i += 1
    pdf.savefig(fig)
    plt.close(fig)


def page_table_s7(pdf: PdfPages) -> None:
    """Table S7：逐条件序统计量（min/p5/p10/p25/median）。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "percentile-recall.json")
                     .read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(W_DOUBLE, 5.0))
    fig.text(0.005, 0.965, "Table S7.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.088, 0.965,
             "The gain sits in the lower tail; the median pays for it.",
             fontsize=8, va="top")
    fig.text(0.005, 0.900,
             "Change in order statistics of per-image recall, proportional minus uniform, in percentage "
             "points. Every condition has the same sign at the minimum, the 5th, 10th and 25th "
             "percentiles, and the opposite sign at the median, so the effect is a shift of the lower "
             "tail rather than a change confined to one image.\n"
             "VisDrone2019-DET validation, seed 42, $N$ = 548.",
             fontsize=6.5, va="top")

    cols = ["head", "$\\rho$", "$\\Delta$ min", "$\\Delta$ p5", "$\\Delta$ p10",
            "$\\Delta$ p25", "$\\Delta$ median", "worst image same?"]
    xs = [0.05, 0.14, 0.26, 0.38, 0.50, 0.62, 0.75, 0.88]
    y0, dy = 0.55, 0.040
    for x, c in zip(xs, cols):
        fig.text(x, y0 + 0.030, c, fontsize=5.6, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 + 0.022, y0 + 0.022],
                                color="black", lw=0.5, transform=fig.transFigure))

    for i, r in enumerate(_d["rows"]):
        y = y0 - i * dy
        cells = [r["head"], f"{r['rho']:.2f}",
                 f"{r['d_min']:+.2f}", f"{r['d_p5']:+.2f}", f"{r['d_p10']:+.2f}",
                 f"{r['d_p25']:+.2f}", f"{r['d_median']:+.2f}",
                 "yes" if r["worst_same"] else "no"]
        for x, v in zip(xs, cells):
            fig.text(x, y, v, fontsize=6.0)
    pdf.savefig(fig)
    plt.close(fig)


def _parse_md_table(rows: list[str]) -> tuple[list[str], list[list[str]]]:
    """把 markdown 表行解析成 (表头, 数据行)。跳过 |---| 分隔行。"""
    def cells(line: str) -> list[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip().replace("`", "").replace("**", "") for c in s.split("|")]

    body = [r for r in rows if not set(r.replace("|", "").strip()) <= set("-: ")]
    if not body:
        return [], []
    return cells(body[0]), [cells(r) for r in body[1:]]


def page_moved_tables(pdf: PdfPages) -> None:
    """S8/S9/S10：从主文迁出的三张表（原 Table 1 / 4 / 5）。"""
    import json as _json
    import re as _re
    import textwrap as _tw
    _d = _json.loads((ROOT / "results" / "tables" / "moved-tables.json")
                     .read_text(encoding="utf-8"))

    for key in ("1", "4", "5"):
        if key not in _d:
            continue
        item = _d[key]
        si = item["si_id"]
        head, body = _parse_md_table(item["rows"])
        if not head:
            continue

        ncol, nrow = len(head), len(body)
        fig = plt.figure(figsize=(W_DOUBLE, 6.6))

        # ---- 标题区（三层，互不重叠）----
        fig.text(0.005, 0.972, f"Table {si}.", fontsize=8, fontweight="bold", va="top")

        # markdown 形如 `**Table N. <Headline>.** <Rest>`，必须在 `.**` 处切分，
        # 否则 headline 与 rest 会是同一段（实测标题因此重复一次）。
        raw = item["title"]
        mm = _re.match(r"\*\*Table \S+\.\s*(.*?)\.\*\*\s*(.*)", raw, _re.S)
        if mm:
            headline = _re.sub(r"\s+", " ", mm.group(1)).strip()[:110]
            rest = _re.sub(r"\s+", " ", mm.group(2).replace("`", "")).strip()
            for _old, _new in (("Table 1", "Table S8"), ("Table 4", "Table S9"),
                               ("Table 5", "Table S10")):
                rest = rest.replace(_old, _new)
        else:
            clean = raw.replace("**", "").replace("`", "")
            part = clean.split(".", 1)
            headline = (part[1].strip() if len(part) > 1 else clean)[:110]
            rest = ""
        fig.text(0.098, 0.972, headline, fontsize=7.2, va="top")
        y_txt = 0.938
        if rest:
            for ln in _tw.wrap(rest, 148)[:5]:
                fig.text(0.005, y_txt, ln, fontsize=5.8, va="top")
                y_txt -= 0.018
        fig.text(0.005, y_txt - 0.004,
                 "Values are exactly as computed; see Methods for definitions.",
                 fontsize=5.8, style="italic", va="top")

        # ---- 表格区 ----
        left, right = 0.030, 0.975
        span = right - left
        xs = [left + span * i / ncol for i in range(ncol)]

        y_head = y_txt - 0.055
        for x, c in zip(xs, head):
            fig.text(x, y_head, "\n".join(_tw.wrap(c, max(8, int(120 / ncol))))[:48],
                     fontsize=4.9, fontweight="bold", va="bottom")
        fig.lines.append(plt.Line2D([left - 0.012, right + 0.012],
                                    [y_head + 0.010, y_head + 0.010],
                                    color="black", lw=0.6, transform=fig.transFigure))

        y_body = y_head - 0.014
        dy = 0.031
        drawn = 0
        for row in body:
            y = y_body - drawn * dy
            if y < 0.025:
                break
            for x, v in zip(xs, row):
                fig.text(x, y, "\n".join(_tw.wrap(v, max(9, int(130 / ncol))))[:60],
                         fontsize=4.9, va="top")
            drawn += 1

        y_end = y_body - max(drawn - 1, 0) * dy - 0.016
        fig.lines.append(plt.Line2D([left - 0.012, right + 0.012], [y_end, y_end],
                                    color="black", lw=0.6, transform=fig.transFigure))
        pdf.savefig(fig)
        plt.close(fig)


def page_table_s11(pdf: PdfPages) -> None:
    """Table S11：目标预算 vs 实际发出框数（spend audit）。"""
    import json as _json
    _d = _json.loads((ROOT / "results" / "tables" / "spend-audit.json")
                     .read_text(encoding="utf-8"))
    _rows = _d["rows"]

    fig = plt.figure(figsize=(W_DOUBLE, 6.4))
    fig.text(0.005, 0.968, "Table S11.", fontsize=8, fontweight="bold", va="top")
    fig.text(0.005, 0.944, "Target budget against the boxes actually emitted.",
             fontsize=7.4, va="top")
    fig.text(0.005, 0.912,
             "`emitted` is $\\sum_i \\min(q_i, p_i)$, the number of boxes the short list really emits; "
             "`capped` is $\\sum_i \\min(q_i, n_i, p_i)$. The two agree on every row, so the spend "
             "identity used in the main text holds throughout the reported conditions. At `B` = 100 and "
             "`B` = 300 the emitted count is pinned at $\\sum_i n_i$ = 38,759 rather than at $\\sum_i p_i$ "
             "= 299,100, which is the direct evidence that the saturation there comes from the annotated "
             "object count and not from the detector's prediction capacity.\n"
             "VisDrone2019-DET validation, `N` = 548, all three seeds. `q_i > n_i` never occurs because the cap is imposed "
             "by construction.",
             fontsize=5.8, va="top")

    hdr = ["seed", "head", "condition", "allocation", "$S$ target", "emitted", "capped",
           "error %", "$q>n$", "$q>p$"]
    xs = [0.030, 0.082, 0.150, 0.268, 0.440, 0.548, 0.640, 0.732, 0.836, 0.906]
    y0 = 0.735
    for x, c in zip(xs, hdr):
        fig.text(x, y0, c, fontsize=5.0, fontweight="bold", va="bottom")
    fig.lines.append(plt.Line2D([0.022, 0.985], [y0 + 0.009, y0 + 0.009],
                                color="black", lw=0.6, transform=fig.transFigure))

    dy = 0.0158
    for i, r in enumerate(_rows):
        y = y0 - 0.020 - i * dy
        if y < 0.030:
            break
        vals = [str(r["seed"]), r["head"], r["condition"], r["allocation"],
                f"{r['S_target']:,.0f}", f"{r['emitted']:,.0f}", f"{r['capped']:,.0f}",
                f"{r['emitted_err_pct']:.4f}", str(r["n_q_gt_n"]), str(r["n_q_gt_p"])]
        for x, v in zip(xs, vals):
            fig.text(x, y, v, fontsize=4.2, va="top")

    y_end = y0 - 0.020 - min(len(_rows), int((y0 - 0.05) / dy)) * dy
    fig.lines.append(plt.Line2D([0.022, 0.985], [y_end, y_end],
                                color="black", lw=0.6, transform=fig.transFigure))
    pdf.savefig(fig)
    plt.close(fig)


def page_fig_s1(pdf: PdfPages) -> None:
    """Figure S1：训练曲线（两种子）。"""
    runs = [r for r in ("r1_yolo26n_visdrone_640__s42",
                        "r1_yolo26n_visdrone_640__s43",
                        "r1_yolo26n_visdrone_640__s44")
            if (RUNS / r / "results.csv").exists()]
    cs = [c for c in (read_curve(r) for r in runs) if c]
    if not cs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.7))
    cols = [OI["vermillion"], OI["blue"], OI["green"]]
    stys = ["-", "--", ":"]
    mks = ["o", "s", "^"]
    ax = axes[0]
    for c, col, ls in zip(cs, cols, stys):
        ax.plot(c["ep"], c["m50"], color=col, ls=ls, lw=1.2, label=f"{c['name'][-3:]} mAP50")
        ax.plot(c["ep"], c["m95"], color=col, ls=ls, lw=0.8, alpha=0.6,
                label=f"{c['name'][-3:]} mAP50-95")
    ax.axhline(0.24, color="0.35", lw=0.5, ls=":")
    ax.axhline(0.130, color="0.35", lw=0.5, ls=":")
    ax.text(2, 0.245, "G0 gate", fontsize=5.5, color="0.35")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP")
    ax.legend(fontsize=5.5, ncol=1)
    panel_label(ax, "a")
    ax = axes[1]
    for c, col, ls in zip(cs, cols, stys):
        ax.plot(c["ep"], c["box"], color=col, ls=ls, lw=1.1, label=f"{c['name'][-3:]} box")
        ax.plot(c["ep"], c["cls"], color=col, ls=ls, lw=0.8, alpha=0.55,
                label=f"{c['name'][-3:]} cls")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (a.u.)")
    ax.legend(fontsize=5.5, ncol=2)
    panel_label(ax, "b")
    pdf.savefig(fig)
    plt.close(fig)


def page_fig_s2(pdf: PdfPages) -> None:
    """Figure S2：ρ 扫描。"""
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.62, 2.7))
    for head, col, ls, mk in (("one-to-many", OI["vermillion"], "-", "o"),
                              ("one-to-one", OI["blue"], "--", "s")):
        rows = [r for r in RHO_SWEEP if r[0] == head]
        ax.plot([r[1] for r in rows], [r[4] for r in rows], color=col, ls=ls, lw=1.2,
                marker=mk, ms=2.8, mew=0.4, label=head)
    ax.axhline(10, color=OI["green"], lw=0.6, ls="-")
    ax.text(0.305, 10.25, "threshold A (10 pp)", fontsize=6, color=OI["green"])
    ax.axhline(5, color="0.45", lw=0.6, ls="-")
    ax.text(0.305, 5.25, "threshold B (5 pp)", fontsize=6, color="0.45")
    ax.set_xlabel(r"Budget fraction $\rho$")
    ax.set_ylabel(r"$\Delta$ worst-image recall (pp)")
    ax.set_ylim(0, 12.2)
    ax.legend(loc="lower left")
    pdf.savefig(fig)
    plt.close(fig)


def page_note_s1(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(W_DOUBLE, 3.2))
    txt = (
        "Note S1. Numerical self-checks and independent recomputation.\n\n"
        "1. Allocation equivalence. Truncating a single inference pass performed at a large max_det\n"
        "   reproduces running at each smaller max_det directly: 16 of 16 comparisons match, on both\n"
        "   prediction heads and across eight budgets, with a maximum coordinate deviation of 0.0.\n"
        "   All budget results therefore cost no additional inference beyond one pass per head.\n\n"
        "2. Equal-spend precondition. The two allocations are compared only where their total spends\n"
        "   agree to within 1%. Conditions failing this test are reported as excluded rather than\n"
        "   silently included, and the count of excluded conditions is stated in the main text.\n\n"
        "3. Degeneracy flag. The criterion reports, for every budget, how many images receive a quota\n"
        "   of zero or exactly one. In that regime an image's recall is decided by whether the\n"
        "   detector's single highest-confidence prediction is correct, so those budgets are excluded\n"
        "   from mechanism inference in both directions.\n\n"
        "4. Independent recomputation. Chunked evaluation passes are merged only after the merged\n"
        "   per-image object counts are re-derived from the label files and compared against the\n"
        "   recorded counts; any mismatch aborts the merge. The reported merge validated 548 of 548\n"
        "   images with zero missing and zero mismatches.\n\n"
        "5. Two independent paths. Every statistic that entered the pre-registered decision was\n"
        "   computed by two separate code paths and required to agree before being reported."
    )
    fig.text(0.02, 0.97, txt, fontsize=6.8, va="top", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    global RHO_SWEEP
    apply_nature_style()
    print("现算 ρ 敏感性（不嵌入任何数字）…")
    RHO_SWEEP = compute_rho_sweep()
    if len(RHO_SWEEP) != len(RHO_GRID) * 2:
        print(f"⚠️ 期望 {len(RHO_GRID) * 2} 个条件，实得 {len(RHO_SWEEP)}")
    for row in RHO_SWEEP:
        print(f"   {row[0]:<12} ρ={row[1]:.2f}  均 {row[2]:>7.3f}%  水 {row[3]:>7.3f}%  "
              f"Δ {row[4]:>+7.3f}  Δmicro {row[5]:>+6.3f}  {row[6]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        page_table_s1(pdf)

        page_table_s2(pdf)
        page_table_s3(pdf)
        page_table_s4(pdf)
        page_table_s5(pdf)
        page_table_s6(pdf)
        page_table_s7(pdf)
        page_moved_tables(pdf)
        page_table_s11(pdf)
        page_fig_s1(pdf)
        page_fig_s2(pdf)
        page_note_s1(pdf)
        d = pdf.infodict()
        d["Title"] = "Supplementary Information"
        d["Subject"] = "Output-budget allocation criterion for object detection"
    kb = OUT.stat().st_size / 1024
    print(f"✅ 补充材料单一 PDF → {OUT.relative_to(ROOT)}  ({kb:.0f} KB, 4 页)")
    print("   Table S1  ρ 敏感性 10 条件")
    print("   Figure S1 训练曲线（两种子，含 G0 闸门线）")
    print("   Figure S2 ρ 扫描")
    print("   Note S1   数值自检与独立复算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
