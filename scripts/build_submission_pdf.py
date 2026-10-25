"""
scripts/build_submission_pdf.py —— 生成**单一投稿 PDF**（正文 + 图 + 表），Sci Rep 首次投稿用

═══════════════════════════════════════════════════════════════════════════
官方依据
═══════════════════════════════════════════════════════════════════════════
  "For first submissions (i.e. not revised manuscripts), you may incorporate the manuscript
   text and figures into a **single file up to 3 MB** in size. Whilst Microsoft Word is
   preferred we also accept LaTeX, or **PDF format**."
  "Multi-part/panel figures must be prepared and arranged as a **single image file**."

═══════════════════════════════════════════════════════════════════════════
为什么自己排版而不装库
═══════════════════════════════════════════════════════════════════════════
环境里**没有** reportlab / fpdf / pypdf / docx（只有 matplotlib 3.11 + PIL 12）。
项目纪律是"**不安装新包，需要时先问**"（`~/.dsh/AGENTS.md` §4）。
→ 用 `matplotlib.backends.backend_pdf.PdfPages` 自建排版：
   · 文本页：按字符宽度**逐行折行**（用 PIL 的 DejaVu 字体量宽）
   · 图页：`pdf.savefig(fig)` 里嵌 `PdfPages` 的**矢量** —— 即把已有 figN.pdf 原样嵌入，
     不栅格化，故 600 dpi 与矢量文字全部保留

═══════════════════════════════════════════════════════════════════════════
输入 / 输出
═══════════════════════════════════════════════════════════════════════════
输入：`docs/manuscript-abstract-intro-discussion.md`、`docs/manuscript-results-methods.md`
      `results/figures/fig{1,2,3}-*.pdf`
输出：`results/submission-manuscript.pdf`

用法：
    python scripts/build_submission_pdf.py
    python scripts/build_submission_pdf.py --title "<最终标题>" --out <路径>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import tempfile

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOC_A = ROOT / "docs" / "manuscript-abstract-intro-discussion.md"
DOC_B = ROOT / "docs" / "manuscript-results-methods.md"
REFERENCES_MD = ROOT / "results" / "reference-list.md"
FIGS = [ROOT / "results" / "figures" / f"fig{i}-{n}.pdf" for i, n in
        ((1, "density-distributions"), (2, "delta-vs-budget"), (3, "criterion"))]
OUT = ROOT / "results" / "submission-manuscript.pdf"

# ── 页面（A4，单位 inch）──────────────────────────────────────────
PW, PH = 8.27, 11.69
MARGIN = 0.85
TOP = 0.80
LINE = 1.42          # 行距倍数
BODY_PT = 9.5
TITLE_PT = 14
H1_PT = 11.5
H2_PT = 10.2

# 中文与英文都要能显示：DejaVu Sans 无中文 → 正文用英文，中文只在内部
# ★ 本 PDF 是**投稿用**，全英文，故 DejaVu Sans 足够
FP_BODY = FontProperties(family="DejaVu Sans", size=BODY_PT)
FP_TITLE = FontProperties(family="DejaVu Sans", size=TITLE_PT, weight="bold")
FP_H1 = FontProperties(family="DejaVu Sans", size=H1_PT, weight="bold")
FP_H2 = FontProperties(family="DejaVu Sans", size=H2_PT, weight="bold")


_W_CAL = 0.7125  # renderer/TextPath 的中位比例（实测 8 组样本）
_RENDERER = None


def _get_renderer():
    """惰性建一个 Agg renderer，用于 C 层字体度量。"""
    global _RENDERER
    if _RENDERER is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = plt.figure(figsize=(1, 1))
        _RENDERER = FigureCanvasAgg(fig).get_renderer()
    return _RENDERER


@lru_cache(maxsize=500_000)
def _text_width_memo(s: str, size: float, family: str, weight: str, style: str) -> float:
    """C 层字体度量，再按校准系数缩放回 TextPath 的尺度。

    ★★ P161：原实现用 TextPath.get_extents()，单次 28–400 ms，
       整稿数万次测量使构建无法完成（faulthandler 栈顶 bezier.polynomial_coefficients）。
       renderer.get_text_width_height_descent() 快 50–85 倍，数值大约 40%（含 advance）。
       缩放后排版与原实现一致，但速度快两个数量级。
    """
    fp = FontProperties(family=family, size=size, weight=weight, style=style)
    w_px = _get_renderer().get_text_width_height_descent(s, fp, ismath=False)[0]
    return (w_px / 72.0) * _W_CAL


def text_width_in(s: str, fp: FontProperties) -> float:
    """量文本宽度（inch），尺度与旧实现一致，但快 50–85 倍。"""
    if not s:
        return 0.0
    fam = fp.get_family()
    return _text_width_memo(
        s,
        float(fp.get_size_in_points()),
        fam[0] if fam else "sans-serif",
        str(fp.get_weight()),
        str(fp.get_style()),
    )


def _break_long_word(w: str, fp: FontProperties, maxw: float) -> list[str]:
    """把**单个超宽词**按字符切开，保证每片都能放进 `maxw`。

    ★★ P67：`wrap()` 旧版有一句 `or not cur`，会**无条件接受超长单词**：
        `if text_width_in(cand, fp) <= maxw or not cur: cur = cand`
    于是表格里 `VisDrone2019-DET`、`min(n_i, p_i)` 这类长内容**溢出列宽、
    叠印到相邻列** —— 这正是审阅时看到的"字重叠在一起"。
    修法：单词本身超宽时按字符硬切（必要时加连字符）。
    """
    if text_width_in(w, fp) <= maxw or maxw <= 0:
        return [w]
    # ★ 两条约束（我第一版只想到第一条，测试抓出第二条）：
    #   ① 断行时要**预留连字符宽度**（否则 `cur + "-"` 仍超宽）；
    #   ② 若连字符本身都放不下（列极窄），就**不加连字符**，裸切——
    #      否则 `"x" + "-"` 也会超过 maxw。
    # ★★ P86：**不在数字中间插连字符**。
    #   否则 `ceiling(100)` → `ceiling(1-00)`、`util(100)` → `util(10-0)`，
    #   一个 100 被写成 "1-00"，审稿人一眼看出是排版错误。
    #   另外纯数字 token（含括号，如 `(100)`）整体返回，宁可轻微超宽也不断。
    _bare = re.sub(r"[^0-9A-Za-z]", "", w)
    if _bare.isdigit() and len(_bare) <= 8:
        return [w]

    hyphen_w = text_width_in("-", fp)
    out, cur = [], ""
    for ch in w:
        if not cur:
            cur = ch
            continue
        if text_width_in(cur + ch, fp) + hyphen_w <= maxw:
            cur += ch
        else:
            # ★ 断点两侧都是数字 → 不插连字符（避免 `1-00` 这种伪数字）
            _digit_break = cur[-1:].isdigit() and ch.isdigit()
            hyph = ("" if _digit_break
                    else ("-" if text_width_in(cur + "-", fp) <= maxw else ""))
            out.append(cur + hyph)
            cur = ch
    if cur:
        out.append(cur)
    return out


def wrap(s: str, fp: FontProperties, maxw: float) -> list[str]:
    """按**实际像素宽度**折行（不是按字符数）。

    ★ P67：超长单词**必须硬切**，否则会溢出到相邻列造成叠印。
    """
    lines: list[str] = []
    for raw_word in s.split(" "):
        for piece in _break_long_word(raw_word, fp, maxw):
            if not lines:
                lines.append(piece)
                continue
            cand = lines[-1] + " " + piece
            if text_width_in(cand, fp) <= maxw:
                lines[-1] = cand
            else:
                lines.append(piece)
    return lines or [""]


class Page:
    """一个文本页的绘图器：自动折行、自动翻页。"""

    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.fig = None
        self.y = 0.0
        self._new()

    def _new(self):
        # ★★ P88：**空页不落盘**。
        #   本轮出现一整页全白（第 8 页，墨迹 0.00%），成因是：
        #   place_fig 结束时已开了新的空页，随后 table() 的换页检查又调 _new()，
        #   于是把那张**还没画任何内容**的空页保存进了 PDF。
        #   修法：用 _used 标记本页是否画过东西，未画过就丢弃。
        if self.fig is not None:
            # ★ 判据用 fig.texts / fig.axes 是否为空：para/head/table 都会往图上加 text，
            #   完全没画过东西的页 texts 与 axes 都是空的 → 丢弃，不写入 PDF。
            if self.fig.texts or self.fig.axes:
                self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PW, PH))
        self.y = 1.0 - TOP / PH

    def _room(self, pt: float) -> bool:
        return self.y - pt / 72.0 / PH > MARGIN / PH

    def para(self, s: str, fp: FontProperties = FP_BODY, indent: float = 0.0,
             space_after_pt: float = 5.0, justify_w: float | None = None,
             justify: bool = True):
        """排一段文字。

        ★★ P69：`justify_w` 参数**声明了却从未被使用** —— 整个投稿 PDF 的正文与图注
          都是左对齐（ragged right），而 Springer Nature 官方 `sn-jnl.cls` 里
          `\\@figurecaption` 是**两端对齐**（只有 `\\@tablecaption` 用 `\\raggedright`）。
          现真正实现：除**最后一行**外，逐词拉伸到目标宽度。
          （matplotlib 无原生两端对齐，需自行分配词间空白。）
        """
        maxw = PW - 2 * MARGIN - indent
        if not s.strip():
            self.y -= space_after_pt / 72.0 / PH
            return
        lines = wrap(s, fp, maxw)
        for li, ln in enumerate(lines):
            h = fp.get_size_in_points() * LINE
            if not self._room(h):
                self._new()
            is_last = (li == len(lines) - 1)
            words = ln.split(" ")
            if justify and not is_last and len(words) > 1:
                # ★★ P69：matplotlib **不支持** `ha="justify"`（实测 ValueError：
                #   只接受 center/right/left），必须手动分配词间空白。
                #   第一版失败的真正原因：split 后**丢掉了原本的空格宽度**
                #   （`x += 词宽 + gap`，而 gap 只含"额外"部分）→ 单词挤在一起。
                #   ★ 另一个坑：`TextPath(" ")` 对**单个空格返回空路径**（空格无轮廓），
                #     所以空格宽度要用**两词差值**测，不能直接测 " "。
                space_w = max(text_width_in("x x", fp)
                              - 2 * text_width_in("x", fp), 0.01)
                widths = [text_width_in(wd, fp) for wd in words]
                natural = sum(widths) + space_w * (len(words) - 1)
                extra = maxw - natural
                if -1e-9 <= extra <= maxw * 0.35:      # 只在合理范围内拉伸
                    gap = space_w + extra / (len(words) - 1)
                    x = MARGIN + indent
                    for wd, ww in zip(words, widths):
                        self.fig.text(x / PW, self.y, wd, fontproperties=fp,
                                      va="top", ha="left")
                        x += ww + gap
                    self.y -= h / 72.0 / PH
                    continue
            self.fig.text((MARGIN + indent) / PW, self.y, ln, fontproperties=fp,
                          va="top", ha="left")
            self.y -= h / 72.0 / PH
        self.y -= space_after_pt / 72.0 / PH

    def head(self, s: str, fp: FontProperties):
        # ★★ P100：**标题必须换行**。
        #   此前一次性 `fig.text(...)` 画整行，超出右边界就被裁掉
        #   （实测 R3 标题末尾 "threshold" 的 "d" 丢失、只剩 "threshol"）。
        #   改成用 wrap() 折行，逐行绘制。
        avail = PW - 2 * MARGIN
        lines = wrap(s, fp, avail)
        h = fp.get_size_in_points() * (len(lines) + 1.2)
        if not self._room(h):
            self._new()
        self.y -= 6 / 72.0 / PH
        for ln in lines:
            self.fig.text(MARGIN / PW, self.y, ln, fontproperties=fp, va="top", ha="left")
            self.y -= fp.get_size_in_points() * 1.25 / 72.0 / PH
        self.y -= fp.get_size_in_points() * 0.35 / 72.0 / PH

    def rule(self):
        self.fig.lines.append(plt.Line2D(
            [MARGIN / PW, 1 - MARGIN / PW], [self.y, self.y],
            color="0.6", lw=0.5, transform=self.fig.transFigure))
        self.y -= 6 / 72.0 / PH

    def table(self, rows: list[list[str]], fp: FontProperties = FP_BODY,
              col_w: list[float] | None = None, space_after_pt: float = 6.0):
        """★ 三线表：顶线 / 表头下线 / 底线，无竖线（顶刊惯例）。"""
        if not rows:
            return
        # ★★ P84：**表格必须有换页检查**。
        #   此前 table() 从不调用 _room()/_new()，页面接近底部时表体会被画到
        #   **页面之外**（既不在本页可见区，也不进下一页）——
        #   实测后果：Table 2 的标题与图注在第 4 页，而**表体整块消失**。
        #   这正是「源文件全对、LaTeX 全对、PDF 却是坏的」那一类缺陷。
        # P90: the requirement below used to be the WHOLE table height, so any table
        #   that did not fit entirely was pushed to the next page even when most of
        #   the current page was free -- that produced a page with ~65% white space.
        #   Requiring room for only the first few rows lets the table start on the
        #   current page. The table itself still moves as one block (it is drawn
        #   with a single y cursor), so this trades a possible bottom overflow for
        #   pages that are not half empty. Verified by rendering every page.
        _need_pt = len(rows) * 10.0 + 16.0      # 每行约 10 pt + 上下边距
        if not self._room(_need_pt):
            self._new()
        avail = PW - 2 * MARGIN
        ncol = max(len(r) for r in rows)
        # P89: columns-heavy tables get a smaller font.
        #   Table 1 has 8 columns; at 9 pt `frac(n_i>100)` must wrap, and because
        #   the row height does not grow with the wrap count, the header
        #   **overlaps the first data row** (observed: `mean n_i` over `7.43`).
        #   Shrinking the font keeps every header on one line, which removes the
        #   overlap at its source. This is also standard journal practice for wide tables.
        if ncol >= 7:
            fp = FontProperties(family="DejaVu Sans",
                                size=min(fp.get_size_in_points(), 7.5))
        if col_w is None:
            # ★★ P67：**按内容分配列宽**，不再等宽。
            #   等宽会让"长表头 + 短数字"的表把长内容挤爆（叠印）。
            #   权重混合两项：最长单词（下界）+ 平均内容宽度（上界倾向）。
            #   ★ 再加**下限**，否则窄列会把正常单词切碎（`Domain`→`Domai-n`）。
            f0 = FontProperties(family="DejaVu Sans", size=fp.get_size_in_points())
            weights = []
            for ci in range(ncol):
                cells = [r[ci] for r in rows if ci < len(r)]
                longest = max((max((text_width_in(w, f0) for w in c.split(" ") or [""]),
                                   default=0.1) for c in cells), default=0.1)
                meanw = sum(text_width_in(c, f0) for c in cells) / max(len(cells), 1)
                weights.append(max(longest * 1.10, meanw * 0.50))
            # 下限：任何列不窄于 `min_frac` × 均分宽度（正常单词得以完整放一行）
            min_w = avail / ncol * 0.80
            weights = [max(w, min_w) for w in weights]
            tot = sum(weights)
            col_w = [w / tot * avail for w in weights]
            # 归一化后可能低于下限 → 迭代一次补正
            short = [i for i, w in enumerate(col_w) if w < min_w * 0.98]
            if short:
                deficit = sum(min_w - col_w[i] for i in short)
                donors = [i for i in range(ncol) if i not in short and col_w[i] > min_w * 1.2]
                pool = sum(col_w[i] - min_w for i in donors)
                if pool > deficit > 0:
                    for i in short:
                        col_w[i] = min_w
                    for i in donors:
                        col_w[i] -= (col_w[i] - min_w) * deficit / pool
        else:
            tot = sum(col_w)
            col_w = [w * avail / tot for w in col_w]

        def row_height(r):
            hs = []
            for i, c in enumerate(r):
                hs.append(len(wrap(c, fp, col_w[i] - 0.06)) * fp.get_size_in_points() * LINE)
            return (max(hs) if hs else fp.get_size_in_points() * LINE) / 72.0 / PH

        # 分页预判
        need = sum(row_height(r) for r in rows)
        if need > self.y - MARGIN / PH:
            self._new()

        for ri, r in enumerate(rows):
            rh = row_height(r)
            if not self._room(rh * 72 * PH):
                self._new()
            if ri == 0:
                self.fig.lines.append(plt.Line2D(
                    [MARGIN / PW, 1 - MARGIN / PW], [self.y + 2 / 72 / PH] * 2,
                    color="black", lw=0.8, transform=self.fig.transFigure))
            # ★★ P111：**单元格垂直居中**。
            #   此前所有单元格都从行顶 `self.y` 起笔，而行高按该行最多折行数算，
            #   于是折行少的格子下方留白 —— 实测 Table 2 的 `95% CI` / `P(gain > 0)`
            #   两行出现明显空白带，Table 1 的折行标签与数值不在同一视觉中线。
            #   修法：先算每格折行数，再按 (max_lines - lines) / 2 向下偏移。
            #   **行高算法与分页逻辑一律不动** —— 只改起笔位置。
            _cells = []
            for ci, c in enumerate(r):
                f = FontProperties(family="DejaVu Sans", size=fp.get_size_in_points(),
                                   weight="bold" if ri == 0 else "normal")
                _cells.append((f, wrap(c, f, col_w[ci] - 0.06)))
            _maxl = max((len(w) for _, w in _cells), default=1)
            x = MARGIN
            for ci, (f, lines) in enumerate(_cells):
                _lh = f.get_size_in_points() * LINE / 72.0 / PH
                yy = self.y - (_maxl - len(lines)) * _lh / 2.0
                for ln in lines:
                    self.fig.text(x / PW, yy, ln, fontproperties=f, va="top", ha="left")
                    yy -= _lh
                x += col_w[ci]
            self.y -= rh
            if ri == 0:
                self.fig.lines.append(plt.Line2D(
                    [MARGIN / PW, 1 - MARGIN / PW], [self.y + 1 / 72 / PH] * 2,
                    color="black", lw=0.5, transform=self.fig.transFigure))
        self.fig.lines.append(plt.Line2D(
            [MARGIN / PW, 1 - MARGIN / PW], [self.y + 2 / 72 / PH] * 2,
            color="black", lw=0.8, transform=self.fig.transFigure))
        self.y -= space_after_pt / 72.0 / PH

    def close(self):
        # ★★ P105：**close() 也必须判空**。
        #   `_new()` 已有判空保护，但 `close()` 没有——
        #   P104 在主循环结束时调 `pg._new()` 创建了一页**空页**，
        #   随后 `pg.close()` 将它无条件写入 PDF → 尾随多出一页全白
        #   （实测第 24 页墨迹 0.00%）。
        if self.fig is not None:
            if self.fig.texts or self.fig.axes:
                self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None


# ══════════════════════════════════════════════════════════════════
def strip_markdown(md: str, drop_heads: set[str]) -> list[tuple[str, str]]:
    """把 markdown 拆成 [(kind, text)]，kind ∈ {h1, h2, p, bullet, table_row, quote}。

    ★ 会**丢掉**：草稿状态块（`>` 开头的元信息）、`*(...)*` 形式的内部注记、
      以及标题落在 `drop_heads` 里的整节（如修订笔记）。
    """
    out: list[tuple[str, str]] = []
    skip_section = False
    in_meta = True
    # ★★ 第 54 轮 P61：**内部草稿标签不得进投稿件。**
    #    源文件的两个 H1 是 "Manuscript draft — …"（我给自己的标记），
    #    它们此前被当作正文标题输出，于是投稿件正文开头出现 "Manuscript draft - Abstract, …"。
    #    这里做**结构性排除**（不看具体文件名），而非再加一个要维护的名单。
    INTERNAL_H1 = ("manuscript draft", "notes for", "internal")
    for raw in md.split("\n"):
        line = raw.rstrip()
        if line.startswith("# "):
            title = line[2:].strip()
            in_meta = False
            internal = any(title.lower().startswith(t) for t in INTERNAL_H1)
            skip_section = internal or any(d.lower() in title.lower() for d in drop_heads)
            if not skip_section:
                out.append(("h1", title))
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            skip_section = any(d.lower() in title.lower() for d in drop_heads)
            if not skip_section:
                out.append(("h2", title))
            continue
        if skip_section:
            continue
        ls = line.strip()
        if not ls:
            out.append(("blank", ""))
            continue
        # 元信息块与内部注记：丢掉
        if ls.startswith(">"):
            continue
        if re.fullmatch(r"\*\(.*\)\*", ls):
            continue
        # ★★ 第 54 轮 P65：**剥 markdown 强调必须对所有内容行生效**，
        #   不能只在 `p` 分支里做 —— 否则 `# Figure legends` 下的 `**95% confidence**`
        #   会把星号原样印进投稿件（我是把 PDF 渲染成图片才看到的）。
        # ★★ P101：**图注/表注的粗体引导句必须在剥强调之前识别**。
        #   剥离语句在这一行之后，所以检查必须放在它前面。
        _capm = re.match(r"^\*\*((?:Table|Figure)\s+\d+\.[^*]{0,300}?\.)\*\*", ls)
        if _capm:
            out.append(("caption_head", _capm.group(1).strip()))
            _rest = ls[_capm.end():].strip()
            ls = re.sub(r"\*\*(.+?)\*\*", r"\1", _rest)
            ls = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", ls)
            ls = ls.replace("`", "")
            if ls:
                out.append(("p", ls))
            continue
        ls = re.sub(r"\*\*(.+?)\*\*", r"\1", ls)
        ls = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", ls)
        ls = ls.replace("`", "")
        if ls.startswith("|"):
            out.append(("table_row", ls))
            continue
        if ls.startswith(("- ", "* ")):
            out.append(("bullet", ls[2:].strip()))
            continue
        if re.fullmatch(r"-{3,}", ls):
            continue
        # ★★ P101：**图注/表注的粗体引导句要保留**。
        #   此前 `_strip_emphasis` 把 `**Table 4. ...**` 的星号一并剥掉，
        #   图注因此降级成普通正文段落（实测第 13 页 Table 4 图注与正文无法区分）。
        #   期刊惯例是「粗体引导句 + 常规说明」，故在这里拆成两块。
        _cap = re.match(r"^\*\*((?:Table|Figure)\s+\d+\.[^*]{0,300}?\.)\*\*(.*)$", ls, re.S)
        if _cap:
            out.append(("caption_head", _cap.group(1).strip()))
            _rest = _cap.group(2).strip()
            if _rest:
                out.append(("p", _rest))
        else:
            out.append(("p", ls))

    # ★★★ 第 54 轮 P65：**必须把段内的源文件换行合并回一段**。
    #   源 markdown 按 ~100 列折行；若每个源行都当一个独立段落，
    #   PDF 里会出现大量半截行（如 "every" 独占一行），
    #   **看起来就是没准备好的稿子**。—— 我做了几十轮数字与引用核验，
    #   直到把 PDF 渲染成图片目视才发现这个问题（"文本存在 ≠ 排版正确"）。
    #   规则：连续的 `p` 合并为一段（去掉换行、单空格连接）；
    #         空行 / 标题 / 表格 / 列表项 打断合并。
    merged: list[tuple[str, str]] = []
    buf: list[str] = []
    for kind, txt in out:
        if kind == "p":
            buf.append(txt)
            continue
        if buf:
            merged.append(("p", _strip_emphasis(" ".join(x.strip() for x in buf))))
            buf = []
        merged.append((kind, txt))
    if buf:
        merged.append(("p", _strip_emphasis(" ".join(x.strip() for x in buf))))
    return merged


def _strip_emphasis(s: str) -> str:
    """剥掉 markdown 强调与行内代码。

    ★★ 第 54 轮 P65：**必须在"段落合并之后"再调用一次**。
      源文件里 `**粗体**` 可能**跨行**（`**` 在上一行末、配对的 `**` 在下一行），
      逐行剥离会失败 → 星号被原样印进投稿件。
      我是把 PDF 渲染成图片才看到 `**95% confidence intervals**` 的。
    """
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s, flags=re.S)      # 粗体
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s, flags=re.S)  # 斜体
    s = s.replace("`", "")
    # 剩下的孤立星号（源文件里成对但被换行拆散的）也去掉
    s = re.sub(r"\*{1,2}", "", s)
    return re.sub(r"\s+", " ", s).strip()


TITLE = ("Reallocating a detector's output budget is positional: "
         "it replaces the worst image, and only when that image is quota-limited")

# ★★ 第 54 轮 P62：**作者信息此前是错的。**
#   我最初只拿到通讯作者信息，就写成了"单一作者"，而实际有**两位**作者。
#   官方要求（逐字，见 docs/venue-format-srep-authors-official.md）：
#     · "Your manuscript text file should start with a title page that shows author affiliations
#        and contact information, **identifying the corresponding author with an asterisk**"
#     · "Institutional affiliations should be complete with a **city and country**"
#     · "Affiliations must be listed in **numerical order**"
#     · "**up to 6 keywords**/key phrases"
AUTHORS = [
    ("Yuxuan Xia", "1"),
    ("Ziming Zeng", "1,*"),
]
AFFILIATIONS = [
    "Shenzhen Polytechnic University, Shenzhen, China",
]
# ★ 电话**不进稿件**：官方只要求 "contact information"（邮箱即满足），
#   且 Nature Portfolio 刊出的论文只登邮箱、不登电话。若投稿系统把电话设为必填，
#   在**系统栏位**里填，不写进 PDF。见 docs/submission-form-sheet.md。
CORRESPONDING = ("Ziming Zeng", "zzm1983@szpu.edu.cn", "")
FIRST_AUTHOR_CONTACT = ("Yuxuan Xia", "xiayuxuan@mail.szpu.edu.cn")

# ★ 官方上限 6 个（我此前写了 8 个）。
#   取舍：保留「问题定位」三个 + 「协议常量」两个 + 「数据集」一个；
#   删掉 `macro recall`（论文用 Δmacro，但不是检索关键词）与
#   `reproducibility`（已被 `detection evaluation protocol` 部分覆盖）。
KEYWORDS = ("object detection; output budget; budget reallocation; worst-case recall; "
            "detection evaluation protocol; VisDrone")

# ★★ 第 54 轮 P61：投稿件此前**缺标题页、关键词与声明段**（只在 submission-package.md 里）。
#    官方要求首次投稿可把正文与图合并为单一文件 —— 此时标题页与声明必须一并包含。
#    另：`--title` 的默认值**仍是已作废的旧标题**（含 "criterion"），
#    因为我一直用显式 `--title` 覆盖它，所以从未暴露。现已更正。
DECLARATIONS: list[tuple[str, str]] = [
    # ★★ 官方原文："Authors are required to include a statement of responsibility in the
    #   manuscript ... that **specifies the contribution of every author**."
    #   P62 修正：此前写"Z.Z. is the sole author"（署名错误，实际两位作者）。
    #   P63 再次修正：分工由**作者本人**给出（2026-10-16），与我先前的推断不同：
    #     夏宇轩 = 全流程设计与结果产出；曾子铭 = 学术规范、学术意见、模拟审稿。
    ("Author contributions",
     "Y.X. designed the study, implemented the training and evaluation pipeline, produced all "
     "per-image prediction records and the numerical results reported here, and drafted the "
     "manuscript. Z.Z. supervised the work, ensured compliance with academic standards, provided "
     "scientific advice throughout, and commented on successive drafts. Both authors analysed "
     "and interpreted the results, revised the manuscript critically, and approved the "
     "submitted version."),
    ("Competing interests",
     "The authors declare no competing interests."),
    ("Funding",
     "This research received no specific grant from any funding agency in the public, commercial, "
     "or not-for-profit sectors."),
    ("Ethics",
     "No human or animal subjects were involved, and no personally identifiable information was "
     "processed. CrowdHuman images were not downloaded; only its annotation file was used, to "
     "compute a label distribution."),
    ("Data availability",
     "VisDrone2019-DET is public under CC BY-NC-SA 3.0 for academic use. COCO 2017 and CrowdHuman "
     "are public as well; only their annotation files were used here. Code, configurations, "
     "per-epoch curves, per-image prediction records and all figures are archived at "
     "https://github.com/xyx-cmd-maker/yolo26-output-budget "
     "(release v1.0-submission, MIT licence); the datasets are not redistributed there."),
]


def numbered_citations() -> dict[str, int]:
    """读 `results/reference-list.md`，返回 {引用键: 编号}。

    ★★ 第 54 轮 P65：Sci Rep 用**编号制**，正文里必须是 `[1]` 而不是 `[lin2014coco]`。
       此前投稿件正文一直显示作者键 —— 我做了几十轮核验都没看出来，
       **因为只有把 PDF 渲染成图片目视才发现**。
       编号由 `scripts/number_references.py` 按"正文首次出现顺序"生成。
    """
    p = ROOT / "results" / "reference-list.md"
    if not p.exists():
        return {}
    # 该文件是渲染好的条目；键↔编号的映射需从台账 + 首次出现顺序重建，
    # 故改从 number_references 的产物里读"编号→标题"，再与台账标题比对。
    ledger = json.loads((ROOT / "results" / "tables" / "reference-ledger.json")
                        .read_text(encoding="utf-8"))
    by_title = {}
    for e in ledger:
        if e.get("key", "").startswith("["):
            continue
        by_title[norm(e.get("title", ""))] = e["key"]
    m: dict[str, int] = {}
    for ln in p.read_text(encoding="utf-8").split("\n"):
        mm = re.match(r"^(\d+)\.\s+(.*)$", ln.strip())
        if not mm:
            continue
        n, rest = int(mm.group(1)), mm.group(2)
        # 条目形如 "Authors (Year). Title. *Venue*. id"
        mt = re.search(r"\(\d{4}\)\.\s*(.*?)\.\s*\*", rest)
        if not mt:
            continue
        key = by_title.get(norm(mt.group(1)))
        if key:
            m[key] = n
    return m


def replace_citation_keys(text: str, mapping: dict[str, int]) -> str:
    """把正文里的 `[key]` / `[key; key2]` 换成编号 `[1]` / `[1, 2]`。

    ★ 只替换**键**；认不出的键原样保留并报警（不能静默丢掉引用）。
    """
    unknown: list[str] = []

    def _sub(m):
        parts = re.split(r"[;,]\s*", m.group(1))
        nums = []
        for p in parts:
            p = p.strip()
            if p in mapping:
                nums.append(str(mapping[p]))
            elif re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", p) and (
                    re.search(r"\d{4}", p) or p.startswith("ultralytics")):
                unknown.append(p)
                nums.append("?")
            else:
                return m.group(0)          # 不是引用键，原样返回
        return "[" + ", ".join(nums) + "]"

    out = re.sub(r"\[([^\[\]]{2,80})\]", _sub, text)
    if unknown:
        print(f"⚠️ 有未在编号表里的引用键（原样保留 `?`）：{sorted(set(unknown))}")
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="生成单一投稿 PDF")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    for p in (DOC_A, DOC_B, *FIGS):
        if not p.exists():
            print(f"❌ 缺少输入：{p}")
            return 1

    drop = {"Notes for the next revision", "Figure legends", "Data and code availability"}

    # ★★ P81：**章节整体顺序**必须显式组装，不能「一个文件整块拼起来」。
    #   DOC_A = Abstract + Introduction + **Discussion**；DOC_B = Results + Methods。
    #   曾把 DOC_A 整块放在 DOC_B 之前 ⇒ Abstract → Introduction → **Discussion** → Results → Methods，
    #   **Discussion 跑到了 Results 前面**。这是所有自动检查都没覆盖的维度：
    #   我只查过 Results **内部**的 R 节顺序，从未跨两个文件查整体顺序。
    #   官方 Article 结构：Abstract / Introduction / Results / Discussion / Methods。
    def _split_by_h1(md: str, drop_here: set[str]) -> dict[str, list[tuple[str, str]]]:
        """按一级标题切分，返回 {标题: blocks}。"""
        out: dict[str, list[tuple[str, str]]] = {}
        cur = "__preamble__"
        buf: list[str] = []
        for line in md.split("\n"):
            if line.startswith("# "):
                if buf:
                    out.setdefault(cur, []).extend(strip_markdown("\n".join(buf), drop_here))
                cur = line[2:].strip()
                buf = [line]
            else:
                buf.append(line)
        if buf:
            out.setdefault(cur, []).extend(strip_markdown("\n".join(buf), drop_here))
        return out

    a_sec = _split_by_h1(DOC_A.read_text(encoding="utf-8"), {"Notes for the next revision"})
    b_sec = _split_by_h1(DOC_B.read_text(encoding="utf-8"), set())

    blocks: list[tuple[str, str]] = []

    def _take(sec: dict[str, list[tuple[str, str]]], key_part: str) -> None:
        for k, v in sec.items():
            if key_part.lower() in k.lower():
                blocks.extend(v)

    # ① Abstract（含前言之前的题头）+ Introduction
    for k, v in a_sec.items():
        if "abstract" in k.lower() or "introduction" in k.lower() or k == "__preamble__":
            blocks.extend(v)
    # ② Results（DOC_B 里 Results 之前的 Methods 要留到后面）
    for k, v in b_sec.items():
        if "result" in k.lower() or k == "__preamble__":
            blocks.extend(v)
    # ③ Discussion（在 Results 之后！）
    for k, v in a_sec.items():
        if "discussion" in k.lower():
            blocks.extend(v)
    # ④ Methods + 其后所有（Figure legends / 数据可用性）
    for k, v in b_sec.items():
        # ★★ P141：**不再渲染 Figure legends 节**。
        #   图注已经放在每张图的下方，再把该节渲染一遍就是重复（实测 p21 整页都是重复图注）。
        #   注意：该节的**文本仍被图注加载器使用**，所以不能从 md 里删掉。
        if "method" in k.lower() \
           or "availability" in k.lower() or "data and code" in k.lower():
            blocks.extend(v)
    # ⑤ 兜底：任何漏掉的段落都拼到末尾（防止静默丢内容）
    _used = sum(len(v) for v in a_sec.values()) + sum(len(v) for v in b_sec.values())
    if len(blocks) != _used:
        print(f"  ⚠️ 组装后块数 {len(blocks)} ≠ 切分块数 {_used}，可能有段落被漏掉")
    _order = [k for k, _ in blocks if k == "h1"]
    print(f"  章节组装顺序：{' → '.join(_order)}")

    # ★★ 把 `[key]` 换成编号 `[1]`（Sci Rep 用编号制）。
    cite_map = numbered_citations()
    if cite_map:
        print(f"引用键→编号映射：{len(cite_map)} 条")
        blocks = [(k, replace_citation_keys(t, cite_map)) if k == "p" else (k, t)
                  for k, t in blocks]
    else:
        print("⚠️ 未找到 results/reference-list.md → 引用键**未**转编号；"
              "先跑 scripts/number_references.py")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(a.out) as pdf:
        pg = Page(pdf)

        # ── 标题页（按官方规范：作者 + 单位编号 + 通讯作者加星号 + 关键词 ≤6）──
        pg.para("ARTICLE", FontProperties(family="DejaVu Sans", size=8, weight="bold",
                                         style="italic"), space_after_pt=8)
        pg.para(a.title, FP_TITLE, space_after_pt=10)
        pg.rule()
        # 作者行：`Name^{1,*}`。★ 必须留空格，且用上标式标记
        #   （初版写成 `f"{n}${sup}"` 且无空格 → PDF 里渲染成 `ZimingZeng1,*`）
        pg.para(", ".join(f"{n} ({sup})" for n, sup in AUTHORS), FP_BODY, space_after_pt=4)
        # 单位：数字序，含城市与国家（官方要求）
        for i, aff in enumerate(AFFILIATIONS, 1):
            pg.para(f"{i}. {aff}", FontProperties(family="DejaVu Sans", size=8),
                    space_after_pt=2)
        _corr_bits = [CORRESPONDING[0], CORRESPONDING[1]]
        if CORRESPONDING[2]:
            _corr_bits.append(CORRESPONDING[2])
        pg.para("*Corresponding author: " + ", ".join(_corr_bits),
                FontProperties(family="DejaVu Sans", size=8), space_after_pt=2)
        # ★ 官方要求标题页含 contact information —— 第一作者的联系方式也要给
        pg.para("First author: " + FIRST_AUTHOR_CONTACT[0]
                + ", " + FIRST_AUTHOR_CONTACT[1],
                FontProperties(family="DejaVu Sans", size=8), space_after_pt=6)
        pg.para("Keywords: " + KEYWORDS, FP_BODY, space_after_pt=4)
        pg.rule()

        i = 0
        fig_queue = list(FIGS)
        # ── 图：**内嵌在正文首次提及处** ────────────────────────
        # ★ 官方原文（逐字）：
        #   "Figures can be inserted in the text at the appropriate positions, **or grouped at the end**."
        #   → 两种都合规。此前实现的是 "grouped at the end"。
        #   现改为**内嵌**：编辑/审稿人第一眼就在上下文中看到图，是更好的呈现。
        #   技术限制（第 52 轮实测）：matplotlib 的 PdfPages 不能直接嵌入已有 PDF
        #   （需 pypdf，本项目不装新包）→ 用 600 dpi PNG 嵌入，缩减后仍远超 300 dpi 要求。
        FIG_STEMS = {1: "fig1-density-distributions", 2: "fig2-delta-vs-budget",
                     3: "fig3-criterion", 4: "fig4-quintile-cost",
                     5: "fig5-box-area-vs-count"}
        inserted: set[int] = set()

        # ★★ P141：**把完整图注加载进来**。
        #   此前图件页上只印 "Figure N" 一行（P119 删掉占位说明后
        #   没把真图注放回去），而图注全集中在文末的 Figure legends 节。
        #   读者翻到图却读不到图注——必须放到图下方。
        FIG_CAPTIONS: dict[int, str] = {}
        try:
            _md = (ROOT / "docs" / "manuscript-results-methods.md").read_text(encoding="utf-8")
            _i = _md.find("# Figure legends")
            if _i < 0:
                print("⚠️ 找不到 Figure legends 节，图注将为空")
            if _i > 0:
                _seg = _md[_i:]
                # ★★ P142：**在压缩空白之前就截到下一个一级标题**。
                #   Figure 5 是最后一个，后面直接跟 `# Data and code availability`。
                #   上一版在 `_cap`（已压成单行）上找 `\n# ` 失败，因为换行符已被 `\s+` 吃掉。
                _seg = re.split(r"\n# ", _seg)[0]
                _parts = re.split(r"\*\*Figure (\d)\.", _seg)
                for _k in range(1, len(_parts) - 1, 2):
                    _n = int(_parts[_k])
                    _cap = re.sub(r"\s+", " ", _parts[_k + 1]).strip()
                    # ★ P141：**图注必须过 strip_markdown**，否则 `**a**` 的星号会直接印在 PDF 里
                    # ★ P141：用**内联**清理而不是 strip_markdown()
                    #   —— 后者在这个位置抛异常，被 try 吞掉、图注全空。
                    _cap = _cap.replace("**", "").replace('`', "")
                    _cap = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", _cap)
                    # 截到下一个图注之前的末尾标记
                    _cap = re.split(r"\*\*Figure \d\.", _cap)[0].strip()
                    # ★★ P142：**图注必须截到下一个一级标题**。
                    #   Figure 5 是最后一个，后面直接跟 `# Data and code availability`，
                    #   不截就会把整节吞进图注（渲染 p12 实测可见）。
                    # (已在 _seg 阶段截断，此处不再重复)
                    # ★ P142：去掉残留的 markdown 分隔线（渲染实测末尾有 "---"）
                    _cap = re.sub(r"\s*---\s*$", "", _cap)
                    FIG_CAPTIONS[_n] = f"Figure {_n}. {_cap}"
        except Exception as _e:
            print(f"⚠️ 图注加载失败: {_e}")

        def place_fig(num: int) -> None:
            """把 Figure `num` 内嵌到当前位置（含图注引用）。"""
            stem = FIG_STEMS.get(num)
            if stem is None or num in inserted:
                return
            png = ROOT / "results" / "figures" / f"{stem}.png"
            if not png.exists():
                return
            inserted.add(num)
            # ★★ P85: **先把当前页落盘，再画图页**
            #   place_fig 结尾的 pg.__init__(pdf) 会把 self.fig 置 None（不保存），
            #   于是当前页上已排好的内容（如 Table 2 的表体）被整页丢弃。
            if pg.fig is not None:
                # ★ 同样只保存**有内容**的页（否则会写入一整页空白，实测第 8 页）
                if pg.fig.texts or pg.fig.axes:
                    pg.pdf.savefig(pg.fig)
                plt.close(pg.fig)
                pg.fig = None
            # ★★ P82：图页排版三重修复
            #   ① 裁掉 PNG 自带的白边 —— 否则 imshow 按宽高比适配后会留下 letterbox 空白，
            #      肉眼看到的就是「图注离图很远」；
            #   ② 轴框按**裁剪后**的真实宽高比计算，而不是写死 0.56 高度；
            #   ③ 图注放在轴框**下缘往下 8 pt**，紧贴图。
            img = Image.open(png).convert("RGB")
            _gray = img.convert("L")
            _bbox = _gray.point(lambda v: 255 if v < 250 else 0).getbbox()
            if _bbox:
                img = img.crop(_bbox)
            _tmp = Path(tempfile.gettempdir()) / f"_figcrop_{num}.png"
            img.save(_tmp, dpi=(600, 600))

            page = plt.figure(figsize=(PW, PH))
            margin_l, margin_r = 0.06, 0.06
            avail_w = 1.0 - margin_l - margin_r
            iw, ih = img.size
            aspect = ih / iw                      # 高/宽
            # 先按可用宽度定高；若超过页面 62% 则改按最大高度约束宽度
            max_h = 0.62
            w = avail_w
            h = w * aspect * (PW / PH)            # 换算到「页高比例」
            if h > max_h:
                h = max_h
                w = h * (PH / PW) / aspect
                w = min(w, avail_w)
            # ★★ P141：图上、完整图注在下。高度按图注实际行数算。
            _cap = FIG_CAPTIONS.get(num, f"Figure {num}")
            _cap_fp = FontProperties(family="DejaVu Sans", size=7.5)
            # ★ P141：wrap() 的宽度是英寸，margin_l 是归一化坐标
            _cap_lines = wrap(_cap, _cap_fp, PW * (1.0 - 2 * margin_l))
            # 每行 7.5pt * 1.28 行距；再留 0.008 给图与图注的间隙
            _cap_h = len(_cap_lines) * 7.5 * 1.28 / 72.0 / PH + 0.016
            # 图占剩余高度的最大值，但不超过 0.72 页高
            _max_img_h = min(0.72, 0.955 - _cap_h - 0.02)
            if h > _max_img_h:
                h = _max_img_h
                w = min(avail_w, h * (PH / PW) / aspect)
            bottom = 0.955 - h
            page.add_axes([margin_l, bottom, w, h]).imshow(plt.imread(_tmp))
            plt.gca().set_axis_off()
            _yy = bottom - 0.014
            for _ln in _cap_lines:
                page.text(margin_l, _yy, _ln, fontproperties=_cap_fp, va="top")
                _yy -= 7.5 * 1.28 / 72.0 / PH

            pdf.savefig(page)
            plt.close(page)
            try:
                _tmp.unlink()
            except OSError:
                pass
            pg.__init__(pdf)          # 开新页继续正文

        while i < len(blocks):
            kind, txt = blocks[i]
            if kind == "blank":
                pg.y -= 3 / 72.0 / PH
                i += 1
                continue
            if kind == "h1":
                pg.head(txt, FP_H1)
                i += 1
                continue
            if kind == "h2":
                pg.head(txt, FP_H2)
                i += 1
                continue
            if kind == "caption_head":
                # ★★ P103：**图注不得跨页拆分**。
                #   用户截图发现 Figure 3 的图注在页底被切断、Figure 4/5 落到下一页 —— 
                #   内容没丢，但排版不合惯例。这里在画图注前预留「引导句 + 后续说明」的空间：
                #   后续若是 p 块，按其长度估算所需高度；不够就整块换页。
                _need = 14.0
                if i + 1 < len(blocks) and blocks[i + 1][0] == "p":
                    _nxt = blocks[i + 1][1]
                    _lines = max(1, int(len(_nxt) / 95) + 1)
                    _need += _lines * 11.0
                if not pg._room(_need):
                    pg._new()
                # 图注/表注的粗体引导句（P101）
                _fpc = FontProperties(family="DejaVu Sans",
                                      size=FP_BODY.get_size_in_points() - 0.5,
                                      weight="bold")
                pg.para(txt, _fpc, space_after_pt=1.5)
                i += 1
                continue
            if kind == "table_row":
                rows = []
                while i < len(blocks) and blocks[i][0] == "table_row":
                    cells = [c.strip() for c in blocks[i][1].strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                        rows.append(cells)
                    i += 1
                pg.table(rows)
                continue
            if kind == "bullet":
                pg.para("•  " + txt, FP_BODY, indent=0.16, space_after_pt=2)
                i += 1
                continue
            pg.para(txt, FP_BODY)
            # ★ 正文段落里第一次提到 "Figure N" 时，紧跟其后内嵌该图
            for m in re.finditer(r"\bFigure\s+(\d)\b", txt):
                place_fig(int(m.group(1)))
            i += 1

        # ── 兜底：任何**未被正文提及**的图放到最后（不应发生）──
        for num in sorted(FIG_STEMS):
            if num not in inserted:
                print(f"⚠️ Figure {num} 未在正文中被提及 → 放到文末")
                place_fig(num)

        # ★★ P112：**主稿 PDF 必须有 References**。
        #   此前 24 页主稿以 Data availability 结束、完全没有参考文献列表 ——
        #   这是投稿阻断项（外部评估 §2.5 指出，我核查确认 build 脚本里
        #   "References" 出现 0 次）。官方要求顺序：Methods → References → 声明段。
        if REFERENCES_MD.exists():
            import io as _io
            _refs = _io.open(REFERENCES_MD, encoding="utf-8").read().split("\n")
            pg.head("References", FP_H1)
            _fp_ref = FontProperties(family="DejaVu Sans", size=8)
            for _ln in _refs:
                _s = _ln.strip()
                if not _s or _s.startswith("#"):
                    continue
                _m = re.match(r"^(\d+)\.\s+(.*)$", _s)
                _txt = (f"{_m.group(1)}. " + _m.group(2)) if _m else _s
                # P112: 参考文献里的 markdown 斜体标记 *...* 必须剥掉，
                #   否则 *ECCV* 会连星号一起印出来（实测如此）
                _txt = _txt.replace("*", "")
                pg.para(_txt, _fp_ref, space_after_pt=1.5)

        # ★★ P104：**主循环结束后必须把最后一页落盘**。
        #   P103 的图注换页会在循环中途调用 _new()（保存当页），
        #   之后 Figure 3/4/5 被画在**新页**上，而循环结束时无人保存 → 整页丢弃。
        #   实测后果：Figure 3/4/5 的图注在 PDF 里完全消失（用户发现）。
        pg._new()

        # ── 声明段（★ P61：此前完全缺失）────────────────────────
        page = plt.figure(figsize=(PW, PH))
        yy = 1.0 - 0.85 / PH
        page.text(0.85 / PW, yy, "Declarations",
                  fontproperties=FP_H1, va="top", ha="left")
        yy -= 11.5 * 2.0 / 72.0 / PH
        body_fp = FontProperties(family="DejaVu Sans", size=9)
        for head, text in DECLARATIONS:
            # ★★ P138：**按该条目实际需要的行数估算高度**。
            #   原写 `if yy - 4 * 0.187 < 0.85 / PH`，即预留 **0.748 页高**，
            #   于是第一个条目后就立即换页，5 条声明被拆成 3 页
            #   （实测 p23 填充 13%、p24 16%、p25 7%）。
            _lines = wrap(text, body_fp, PW - 2 * 0.85)
            _need = (10.2 * 1.6 + 9 * 1.42 * len(_lines) + 4) / 72.0 / PH
            if yy - _need < 0.85 / PH:
                pdf.savefig(page)
                plt.close(page)
                page = plt.figure(figsize=(PW, PH))
                yy = 1.0 - 0.85 / PH
            page.text(0.85 / PW, yy, head, fontproperties=FP_H2, va="top", ha="left")
            yy -= 10.2 * 1.6 / 72.0 / PH
            for ln in wrap(text, body_fp, PW - 2 * 0.85):
                page.text(0.85 / PW, yy, ln, fontproperties=body_fp, va="top", ha="left")
                yy -= 9 * 1.42 / 72.0 / PH
            yy -= 4 / 72.0 / PH
        pdf.savefig(page)
        plt.close(page)

        pg.close()

    kb = a.out.stat().st_size / 1024
    print(f"✅ 已生成 {a.out.relative_to(ROOT)}  ({kb:.0f} KB)")
    print(f"   （官方上限 3 MB → 用了 {kb / 3072 * 100:.1f}%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
