"""
scripts/build_latex_submission.py —— 生成**符合 Springer Nature 官方模板**的 LaTeX 投稿件

═══════════════════════════════════════════════════════════════════════════
官方原文（逐字，已落盘 `.probe/_srep_submission_guidelines.txt`）
═══════════════════════════════════════════════════════════════════════════
  "For LaTeX submissions we encourage authors to visit the Springer Nature LaTeX Author
   Support page and use the **journal article LaTeX template** available there when preparing
   a submission. Please ensure before submission that the **complete .tex file compiles
   successfully with no errors or warnings**."
  "**Please do not use \\input{...} to include other tex files. Submit your LaTeX manuscript
   as one .tex document.**"
  "All additional figures and files should be attached separately and **not embedded** in the
   TeX document itself."

═══════════════════════════════════════════════════════════════════════════
结构与接口**取自官方模板 `sn-article.tex`**（已下载到 `latex/_template_ref/`）
═══════════════════════════════════════════════════════════════════════════
  · `\\documentclass[sn-mathphys,Numbered]{sn-jnl}`   ← 模板默认（编号制，物理/数学类）
  · 标题页：`\\title[short]{full}` / `\\author*[1]{\\fnm{}\\,\\sur{}}\\email{}` /
            `\\affil*[1]{\\orgdiv{}\\orgname{}\\orgaddress{\\city{}\\country{}}}`
            （`*` 标记**通讯作者**，与官方 submission guidelines 一致）
  · 前置：`\\abstract{}`、`\\keywords{}`、`\\maketitle`
  · 后置：`\\backmatter` + `\\bmhead{...}`（Supplementary information / Acknowledgments /
            Data availability / Author contributions / Competing interests / Funding / Ethics）
  · 文献：`\\bibliography{references}`，正文用 `\\cite{}` / `\\citep{}`

用法：
    python scripts/build_latex_submission.py
    python scripts/build_latex_submission.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latex_spans import spans_to_latex  # noqa: E402  （span 处理器，单遍扫描）

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex"
DOC_A = ROOT / "docs" / "manuscript-abstract-intro-discussion.md"
DOC_B = ROOT / "docs" / "manuscript-results-methods.md"
LEDGER = ROOT / "results" / "tables" / "reference-ledger.json"
FIGDIR = ROOT / "results" / "figures"

# ── 作者与单位（P62：官方要求标题页含单位与联系信息、通讯作者加星号）────────
TITLE = ("Reallocating a detector's output budget is positional: "
         "it replaces the worst image, and only when that image is quota-limited")
TITLE_SHORT = "Reallocating a detector's output budget is positional"
AUTHORS = [
    # (given, surname, affil-number, email, is_corresponding)
    ("Yuxuan", "Xia", "1", "xiayuxuan@mail.szpu.edu.cn", False),
    ("Ziming", "Zeng", "1", "zzm1983@szpu.edu.cn", True),
]
AFFIL = ("", "Shenzhen Polytechnic University", "Shenzhen", "China")
def _kw() -> str:
    """★★ P109：**关键词单一来源**。

    此前本文件有自己的 KEYWORDS 副本，与 build_submission_pdf.py 不一致
    （我只改了后者，导致 LaTeX 版缺 budget reallocation）。
    现从单一来源取，并对 `_` 做 LaTeX 转义：未转义的 `_` 在文本模式中
    会启动下标 → `max_det` 渲染成 `maxdet`（下划线消失，实测如此）。
    """
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location("_bp_kw", ROOT / "scripts" / "build_submission_pdf.py")
    _m = _iu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    return _m.KEYWORDS.replace("_", r"\_")

KEYWORDS = None  # 占位；真值在下方被 _kw() 覆盖
# ── Unicode → LaTeX ──────────────────────────────────────────────────────
UNI = {
    "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "−": r"$-$",
    "≈": r"$\approx$", "→": r"$\to$", "∈": r"$\in$", "∑": r"$\sum$",
    "Σ": r"$\Sigma$", "Δ": r"$\Delta$", "ρ": r"$\rho$", "α": r"$\alpha$",
    "µ": r"$\mu$", "·": r"$\cdot$", "–": "--", "—": "---",
    "∅": r"$\emptyset$", "∞": r"$\infty$", "≠": r"$\neq$",
    "ᵢ": r"$_i$", "ⱼ": r"$_j$", "²": r"$^2$", "³": r"$^3$", "⁴": r"$^4$",
    "⁻": r"$^-$", "⌈": r"$\lceil$", "⌉": r"$\rceil$", "⌊": r"$\lfloor$",
    "⌋": r"$\rfloor$", "▪": r"$\blacksquare$", "✓": r"\checkmark{}",
}
# 其余非 ASCII（如中文）若出现在正文里，直接报错——投稿件必须全英文
LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc_unicode(s: str) -> str:
    """先把 Unicode 数学符号换成 LaTeX，再报告残留的非 ASCII。"""
    for k, v in UNI.items():
        s = s.replace(k, v)
    left = sorted({c for c in s if ord(c) > 127})
    return s, left


def protect_math(s: str) -> tuple[str, list[str]]:
    """把**要保留为 LaTeX 命令**的片段先占位，避免后续转义破坏它们。

    ★★ 第 54 轮 P66：初版只保护 `$...$` 数学段，**没保护我自己生成的 LaTeX 命令**，
      于是 `esc_text` 把 `\\cite{key}` 转义成 `\\textbackslash{}cite\\{key\\}`
      → **`.bib` 生成 0 条**（因为 .tex 里根本没有 `\\cite`）。
      凡是"先生成命令、后统一转义"的流水线，都必须把**已生成的命令**一并占位。
    """
    parts: list[str] = []

    def _grab(m):
        parts.append(m.group(0))
        return f"\x00MATH{len(parts) - 1}\x00"

    # ① 数学段
    s = re.sub(r"\$[^$]*\$", _grab, s)
    # ② 我自己可能已经生成的 LaTeX 命令（含可选的星号与花括号参数）
    for cmd in ("cite", "citep", "citet", "ref", "eqref", "label", "textbf",
                "textit", "texttt", "url", "includegraphics", "checkmark"):
        s = re.sub(re.escape("\\" + cmd) + r"\*?(?:\{[^{}]*\})*", _grab, s)
    return s, parts


def restore_math(s: str, parts: list[str]) -> str:
    for i, p in enumerate(parts):
        s = s.replace(f"\x00MATH{i}\x00", p)
    return s


def esc_text(s: str) -> str:
    """转义 LaTeX 特殊字符（数学段已被占位保护）。"""
    out = []
    for ch in s:
        if ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def md_inline(s: str) -> str:
    """行内 markdown → LaTeX。**实际实现见 `scripts/latex_spans.py`。**

    ★★★ 第 54 轮 P66：这里曾是一个"正则占位符"实现
    （`\\x00MATHn\\x00` 保护数学段 → 转义全文 → 还原），在真实文本上**三处互相破坏**，
    我用编译日志才定位到：
      · `\\textbf{ MATH0 24.20}` —— 占位符 token 被吞，字面 "MATH0" 进了 PDF；
      · `\\textbackslash{}$\\textbackslash{}Delta\\$` —— `\\Delta` 在"保护"之前就被转义；
      · `95\\textbackslash{}\\%` —— `%` 二次转义。
    现改为 `latex_spans.spans_to_latex()`：**单遍扫描、按 span 类型分别处理、不用占位符**，
    并有 10 例自测（`python scripts/latex_spans.py`）。
    """
    return spans_to_latex(s)


def cite_replace(s: str) -> str:
    """`[key]` / `[key; key2]` → `\\cite{key,key2}`（编号制）。"""
    def _sub(m):
        parts = [p.strip() for p in re.split(r"[;,]\s*", m.group(1)) if p.strip()]
        keys = [p for p in parts
                if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", p)
                and (re.search(r"\d{4}", p) or p.startswith("ultralytics"))]
        if not keys or len(keys) != len(parts):
            return None
        return r"\cite{" + ",".join(keys) + "}"

    def _wrap(m):
        r = _sub(m)
        return r if r else m.group(0)

    return re.sub(r"\[([^\[\]]{2,80})\]", _wrap, s)


def sections(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    out, cur, buf = [], "(前言)", []
    for ln in lines:
        if ln.startswith("# ") and not ln.startswith("## "):
            out.append((cur, "\n".join(buf)))
            cur, buf = ln[2:].strip(), []
        else:
            buf.append(ln)
    out.append((cur, "\n".join(buf)))
    return out


def strip_meta(body: str) -> str:
    """去掉引用块与内部注记。

    ★★ 第 224 轮：`*(...)*` 生产注记**可跨多行**。旧实现只跳过以 `*(` 开头的
      单行，于是 Figure legends 节里那条四行注记只丢了首行，后三行变成正文，
      末尾 `)*` 的星号又被强调处理吃掉 —— PDF 里印出断句
      "names the sample size and defines the symbols…"，结尾还多一个 `)`。
      改为**块级**丢弃：先按 `*( ... )*` 整体删除，再逐行清引用块。
    """
    # ① 块级删除 *( ... )*（非贪婪、允许跨行）
    body = re.sub(r"\*\(.*?\)\*", "", body, flags=re.S)
    keep = []
    for ln in body.split("\n"):
        st = ln.strip()
        if st.startswith((">", "*(")) or re.fullmatch(r"-{3,}", st):
            continue
        keep.append(ln)
    return "\n".join(keep)


def md_table_to_latex(rows: list[str], caption: str, label: str) -> str:
    """markdown 表格 → booktabs 表格。

    ★★ P66：初版用 `\\begin{table}` + `l` + `r`×6 的固定列宽，
      结果**5 张表全部 Overfull hbox（84–144 pt 过宽）** —— 编译警告。
      修法：`table*`（跨满版心）+ `\\small` + `\\setlength{\\tabcolsep}{4pt}`，
      并让第一列可自动换行（`p{...}`）。
    """
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]
    header = cells(rows[0])
    body = [cells(r) for r in rows[2:]] if len(rows) > 2 else []
    ncol = len(header)

    def fmt(c):
        c = cite_replace(c)
        c = md_inline(c)
        # ★ P66：长 token（`\texttt{ceiling(100)}`、`$-$2.84 / $-$2.84`）不可断行
        #   → Overfull。允许在 `(`、`)`、`/`、`,` 之后断行。
        #   注意：`\textbf{...}` 内部**无法**用 `\allowbreak` 断（整个 textbf 是一个盒子），
        #   故同时配合表格级 `\sloppy` 放宽断行阈值。
        for ch in ("(", ")", "/", ","):
            c = c.replace(ch, ch + "\\allowbreak ")
        return c

    # ★ 最终采用 **`tabularx`** + `X` 列。
    #   前四版用 `tabular*` + 手算宽度（`w = 0.97/0.86/0.80 / ncol`）**不收敛**：
    #   `\extracolsep{\fill}` 把富余给了列间距而非单元格，残余 73–137 pt。
    #   `tabularx` 会**真正解算**列宽（先按自然宽排一遍，再按 X 列比例分配），
    #   配合 `\sloppy` 与 `\allowbreak` 可把 Overfull 压到可忽略。
    colspec = "@{}" + "R" * ncol + "@{}"

    out = [r"\begin{table}[t]",
           r"\centering",
           r"\small",
           r"\sloppy",
           r"\setlength{\tabcolsep}{3pt}",
           r"\caption{" + fmt(caption) + "}",
           r"\label{" + label + "}",
           r"\begin{tabularx}{\textwidth}{" + colspec + "}",
           r"\toprule"]
    out.append(" & ".join(fmt(c) for c in header) + r" \\")
    out.append(r"\midrule")
    for r in body:
        while len(r) < ncol:
            r.append("")
        out.append(" & ".join(fmt(c) for c in r[:ncol]) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 sn-jnl 格式的 LaTeX 投稿件")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # ── 1. 读手稿并切节 ───────────────────────────────────────────────
    secs: dict[str, str] = {}
    for doc in (DOC_A, DOC_B):
        for name, body in sections(doc):
            secs[name] = strip_meta(body)

    order = ["Abstract", "Introduction", "Results", "Discussion", "Methods",
             "Figure legends", "Data and code availability"]
    for k in order:
        if k not in secs:
            print(f"⚠️ 手稿里没有节 '{k}'")

    # ── 2. 正文转换 ──────────────────────────────────────────────────
    def conv_blocks(body: str, *, tables: bool = True) -> str:
        """把节正文转成 LaTeX：合并段内换行、处理表格、转义。"""
        lines = body.split("\n")
        out: list[str] = []
        buf: list[str] = []
        i = 0
        tbl_no = 0

        def flush():
            nonlocal buf
            if not buf:
                return
            para = " ".join(x.strip() for x in buf)
            buf = []
            para = cite_replace(para)          # [key] → \cite{key}
            # ★ md_inline 现在**自带转义**（先转义、后套命令），故这里不再重复转义。
            para = md_inline(para)
            left = sorted({c for c in para if ord(c) > 127})
            if left:
                raise SystemExit(f"❌ 正文残留非 ASCII 字符（投稿件须全英文）: {left}")
            out.append(para)
            out.append("")

        while i < len(lines):
            ln = lines[i]
            st = ln.strip()
            # 表格
            if st.startswith("|"):
                flush()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append(lines[i].strip())
                    i += 1
                # 表格前的说明行（以 **Table N. 开头）常在上一段，这里用通用标题
                tbl_no += 1
                out.append(md_table_to_latex(rows, f"Table {tbl_no}.", f"tab{tbl_no}"))
                continue
            if st.startswith("### "):
                flush()
                h = st[4:]
                h, _ = esc_unicode(md_inline(h))
                out += [r"\subsection*{" + h + "}", ""]
                i += 1
                continue
            if st.startswith("## "):
                flush()
                h = st[3:]
                h, _ = esc_unicode(md_inline(h))
                out += [r"\subsection{" + h + "}", ""]
                i += 1
                continue
            if not st:
                flush()
                i += 1
                continue
            buf.append(ln)
            i += 1
        flush()
        return "\n".join(out)

    # Abstract 要单独放进 \abstract{}；Introduction 起是正文
    abstract_txt = " ".join(l.strip() for l in secs["Abstract"].split("\n") if l.strip())
    abstract_txt = cite_replace(abstract_txt)
    abstract_txt = md_inline(abstract_txt)
    abstract_txt, left = esc_unicode(abstract_txt)
    if left:
        raise SystemExit(f"❌ 摘要残留非 ASCII: {left}")
    abstract_txt, _p = protect_math(abstract_txt)
    abstract_txt = esc_text(abstract_txt)
    abstract_txt = restore_math(abstract_txt, _p)

    # ── 3. 组装 .tex ─────────────────────────────────────────────────
    L: list[str] = []
    L += [r"% !TeX program = pdflatex",
          r"% LaTeX submission prepared with the Springer Nature journal article template (sn-jnl).",
          r"% Structure and commands follow the official template's sn-article.tex.",
          r"\documentclass[sn-mathphys,Numbered]{sn-jnl}",
          # ★★ 第 54 轮 P66：**前导区必须与官方模板一致**。
          #   初版只加了 graphicx/booktabs/amsmath/textcomp/geometry，
          #   **漏了 xcolor** → `\ArtType`（用 `\colorbox{artcatboxgray}`）未定义
          #   → **编译致命失败**（官方要求"必须编译无错"）。
          #   现严格照 sn-article.tex 的包列表补齐。
          r"\usepackage{graphicx}", r"\usepackage{multirow}",
          r"\usepackage{amsmath,amssymb,amsfonts}", r"\usepackage{amsthm}",
          r"\usepackage{mathrsfs}", r"\usepackage[title]{appendix}",
          r"\usepackage{xcolor}", r"\usepackage{textcomp}",
          r"\usepackage{manyfoot}", r"\usepackage{booktabs}",
          r"\usepackage{tabularx}", r"\usepackage{ragged2e}", r"\usepackage{geometry}", r"",
          # 表格用 X 列（自动解算宽度）；`\RaggedRight` + 连字允许长词断开，
          # 消掉最后几个 Overfull（`CrowdHuman` / `VisDrone2019-` 等长词撑破首列）。
          r"\newcolumntype{R}{>{\RaggedRight\arraybackslash\hspace{0pt}}X}", r"",
          r"\graphicspath{{figures/}}", r"",
          # 官方模板在 \begin{document} 前的这两行，我也必须照抄
          r"\raggedbottom",
          r"%%\unnumbered% uncomment this for unnumbered level heads",
          r"", r"\begin{document}", r""]
    L += [r"\title[" + TITLE_SHORT + r"]{" + TITLE + r"}", r""]
    for given, sur, aff, mail, corr in AUTHORS:
        star = "*" if corr else ""
        L.append(r"\author" + star + r"[" + aff + r"]{\fnm{" + given +
                 r"} \sur{" + sur + r"}}\email{" + mail + r"}")
    L += [r"", r"\affil*[1]{\orgname{Shenzhen Polytechnic University},"
          r"\orgaddress{\city{Shenzhen}, \country{China}}}", r""]
    L += [r"\abstract{" + abstract_txt + r"}", r"",
          r"\keywords{" + _kw() + r"}", r"", r"\maketitle", r""]

    for sec in ("Introduction", "Results"):
        L += [r"\section{" + sec + r"}", conv_blocks(secs[sec])]
    L += [r"\section{Discussion}", conv_blocks(secs["Discussion"])]
    L += [r"\section{Methods}", conv_blocks(secs["Methods"])]

    # ── 4. 后置（\backmatter + \bmhead，官方模板写法）───────────────
    L += [r"\backmatter", ""]
    L += [r"\bmhead{Supplementary information}", "",
          r"The supplementary information accompanying this manuscript contains Table~S1 "
          r"(the $\rho$ sweep over ten conditions), Figure~S1 (per-epoch training curves for both "
          r"completed seeds), Figure~S2 (the gain plotted against $\rho$), and Note~S1 (numerical "
          r"self-checks against independent recomputation).", ""]
    L += [r"\bmhead{Data availability}", "", conv_blocks(secs["Data and code availability"])]
    L += [r"\bmhead{Author contributions}", "",
          r"Y.X. designed the study, implemented the training and evaluation pipeline, produced all "
          r"per-image prediction records and the numerical results reported here, and drafted the "
          r"manuscript. Z.Z. supervised the work, ensured compliance with academic standards, "
          r"provided scientific advice throughout, and acted as an internal reviewer by simulating "
          r"referee critique. Both authors analysed and interpreted the results, revised the "
          r"manuscript critically, and approved the submitted version.", ""]
    L += [r"\bmhead{Competing interests}", "",
          r"The authors declare no competing interests.", ""]
    L += [r"\bmhead{Funding}", "",
          r"This research received no specific grant from any funding agency in the public, "
          r"commercial, or not-for-profit sectors.", ""]
    L += [r"\bmhead{Ethics approval}", "",
          r"No human or animal subjects were involved, and no personally identifiable information "
          r"was processed. CrowdHuman images were not downloaded; only its annotation file was used, "
          r"to compute a label distribution.", ""]

    # ── 5. 图 + 图注 ────────────────────────────────────────────────────
    # ★★ 第 224 轮：**原实现一行 \includegraphics 都没有**，注释写着"此处用
    #    \includegraphics 引用"却从未输出 —— 编译出的 PDF 只有图注、没有图。
    #    现在逐图输出 figure 环境，并把它对应的图注紧邻其后。
    FIG_STEMS = [
        "fig1-density-distributions", "fig2-delta-vs-budget", "fig3-criterion",
        "fig4-quintile-cost", "fig5-box-area-vs-count", "fig6-recall-quantiles",
    ]
    legends_body = conv_blocks(secs["Figure legends"])
    # 按 `\textbf{Figure N.` 切段，使每个图与自己的图注相邻
    parts = re.split(r"(?=\\textbf\{Figure\s+\d+\.)", legends_body)
    parts = [x for x in (q.strip() for q in parts) if x]
    legend_by_n: dict[int, str] = {}
    preamble_parts: list[str] = []
    for q in parts:
        m = re.match(r"\\textbf\{Figure\s+(\d+)\.", q)
        if m:
            legend_by_n[int(m.group(1))] = q
        else:
            preamble_parts.append(q)

    L += [r"\bmhead{Figure legends}", ""]
    for pre in preamble_parts:
        L += [pre, ""]
    for n, stem in enumerate(FIG_STEMS, 1):
        L += [r"\begin{figure}[htbp]", r"\centering",
              rf"\includegraphics[width=\textwidth]{{{stem}}}", r"\end{figure}", ""]
        if n in legend_by_n:
            L += [legend_by_n[n], ""]
    L += [r"\bibliography{references}", r""]
    # ★★ P66：**必须显式闭合 document**。我在重写主循环时漏了它，
    #   编译报 `job aborted, no legal \end found` —— 而那只是症状，
    #   真正原因是生成器少写了一行。
    L += [r"\end{document}", r""]

    tex = "\n".join(L) + "\n"
    if not a.check:
        (OUT / "manuscript.tex").write_text(tex, encoding="utf-8")

    # ── 6. .bib ─────────────────────────────────────────────────────
    # ★★ BibTeX 类型**取自台账的显式字段 `bibtex_type`**，不从字段形式反推。
    #    （初版按"有 doi→@article / 有 arXiv id→@misc"推断，
    #      于是 ECCV / CVPR / SIGIR 三篇**会议论文被写成 @article**。）
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    used = set(re.findall(r"\\cite\{([^}]+)\}", tex))
    used_keys = {k.strip() for grp in used for k in grp.split(",")}

    def bib_escape(s: str) -> str:
        """BibTeX 里 `&`、`%`、`$`、`#`、`_` 需转义；`{}` 用于保护大小写。"""
        s = str(s or "")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                     ("#", r"\#"), ("_", r"\_")):
            s = s.replace(a, b)
        # 去掉 markdown 强调残留
        s = re.sub(r"\*+", "", s)
        return s

    bib: list[str] = [
        "% References for the accompanying manuscript.",
        "% Generated from results/tables/reference-ledger.json by",
        "%   scripts/build_latex_submission.py",
        "% Entry types are taken from the ledger field `bibtex_type` (explicit, not inferred).",
        "",
    ]
    n_bib = 0
    unverified: list[str] = []
    for e in led:
        k = e.get("key", "")
        if k.startswith("[") or k not in used_keys:
            continue
        btype = e.get("bibtex_type")
        if not btype:
            print(f"⚠️ {k}: 台账缺 `bibtex_type`，跳过（不能靠推断）")
            continue
        au = e.get("authors") or []
        if isinstance(au, list):
            astr = " and ".join(au)
        else:
            astr = str(au)
        fields = [f"  author  = {{{bib_escape(astr)}}}",
                  f"  title   = {{{bib_escape(e.get('title', ''))}}}",
                  f"  year    = {{{e.get('year', '')}}}"]
        vf = e.get("bibtex_venue_field")
        vv = e.get("bibtex_venue_value") or e.get("venue") or ""
        if vf:
            fields.append(f"  {vf:<7} = {{{bib_escape(vv)}}}")
        doi = e.get("doi") or ""
        if doi:
            fields.append(f"  doi     = {{{doi}}}")
            fields.append(f"  url     = {{https://doi.org/{doi}}}")
        ident = str(e.get("id") or "")
        if ident.startswith("arXiv:"):
            aid = ident.replace("arXiv:", "")
            fields += [f"  eprint  = {{{aid}}}", "  archivePrefix = {arXiv}",
                       "  primaryClass  = {cs.CV}"]
        elif ident.startswith("ISBN"):
            fields.append(f"  isbn    = {{{ident.replace('ISBN', '').strip()}}}")
        elif ident.startswith("http"):
            fields.append(f"  howpublished = {{{ident}}}")
        elif ident.startswith("JSTOR:"):
            fields.append(f"  jstor   = {{{ident.replace('JSTOR:', '')}}}")
        # 未能在本机联网核验的条目：**显式标注**，不假装已核验
        if not doi and not ident.startswith(("arXiv:", "http", "JSTOR")):
            fields.append("  note    = {ISBN as recorded by the authors; "
                          "not resolvable from this machine}")
            unverified.append(k)
        bib.append(f"@{btype}{{{k},")
        bib.append(",\n".join(fields))
        bib.append("}\n")
        n_bib += 1

    if not a.check:
        (OUT / "manuscript.tex").write_text(tex, encoding="utf-8")
        (OUT / "references.bib").write_text("\n".join(bib), encoding="utf-8")

    print(f"✅ latex/manuscript.tex   {len(tex) / 1024:.0f} KB")
    print(f"✅ latex/references.bib   {n_bib} 条")
    print(f"   正文引用键 {len(used_keys)} 个；"
          f"{'全部有条目' if used_keys <= {e.get('key') for e in led} else '⚠️ 有缺'}")
    if unverified:
        print(f"   ⚠️ 未联网核验的条目（已在 bib 里显式标注）：{unverified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
