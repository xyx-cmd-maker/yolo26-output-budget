"""LaTeX span 处理：单遍扫描，按类型分段，**不用占位符**。

═══════════════════════════════════════════════════════════════════════════
为什么重写（P66）
═══════════════════════════════════════════════════════════════════════════
初版用"正则占位符"（`\\x00MATHn\\x00`）保护数学段与代码段，再转义全文，最后还原。
这套方案在真实文本上**三处互相破坏**，我用编译日志才定位到：

  · `\\textbf{ MATH0 24.20}` —— **占位符 token 被 `\\s+` 归一化与正则吞掉**，
    还原时找不到，字面 "MATH0" 进了 PDF；
  · `\\textbackslash{}$\\textbackslash{}Delta\\$` —— **`\\Delta` 在"保护"之前就被转义**，
    因为保护正则只在特定调用点生效，而 `esc_text` 是全局的；
  · `95\\textbackslash{}\\%` —— `%` 被二次转义。

**教训：只要"保护—转义—还原"三步共用同一字符串，就必然出现互相破坏。**
正确做法是**单遍扫描**：先切出 span（数学 / 代码 / 纯文本），
**每类 span 用各自的规则处理**，再拼回去。没有占位符，就没有还原失败。
"""

from __future__ import annotations

import re

UNI = {
    "≤": r"\leq", "≥": r"\geq", "×": r"\times", "−": "-",
    "≈": r"\approx", "→": r"\to", "∈": r"\in", "∑": r"\sum",
    "Σ": r"\Sigma", "Δ": r"\Delta", "ρ": r"\rho", "α": r"\alpha",
    "µ": r"\mu", "·": r"\cdot", "–": "--", "—": "---",
    "∅": r"\emptyset", "∞": r"\infty", "≠": r"\neq",
    "ᵢ": r"_i", "ⱼ": r"_j", "²": r"^2", "³": r"^3", "⁴": r"^4",
    "⁻": r"^-", "⌈": r"\lceil{}", "⌉": r"\rceil{}", "⌊": r"\lfloor{}",
    "⌋": r"\rfloor{}", "✓": r"\checkmark{}",
}
# 只在**纯文本**里转义
PLAIN_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _esc_plain(s: str) -> str:
    return "".join(PLAIN_ESC.get(c, c) for c in s)


def _uni_to_math(s: str) -> tuple[str, list[str]]:
    """纯文本里的 Unicode 符号 → 数学片段；返回 (文本, 待插入的数学片段列表)。"""
    parts: list[str] = []

    def _sub(m):
        parts.append(UNI[m.group(0)])
        return f"\x01{len(parts) - 1}\x01"

    pat = "|".join(re.escape(k) for k in sorted(UNI, key=len, reverse=True))
    return re.sub(pat, _sub, s), parts


# 已经生成好的 LaTeX 命令：**必须逐字保留**，不可被转义。
#   ★ 这是 span 处理器的第四种 span。初版漏了它，于是 `\cite{key}` 被转成
#     `\textbackslash{}cite\{key\}` → **`.bib` 从 17 条掉到 3 条**。
KNOWN_CMDS = ("cite", "citep", "citet", "ref", "eqref", "label", "url",
              "includegraphics", "checkmark", "ldots", "dots", "times", "cdot",
              "leq", "geq", "neq", "approx", "to", "in", "sum", "Sigma", "Delta",
              "rho", "alpha", "mu", "emptyset", "infty", "lceil", "rceil",
              "lfloor", "rfloor", "textbackslash", "textasciitilde",
              "textasciicircum",
              # ★ 转义序列本身也要认，否则**幂等性被破坏**：
              #   源文档里的 `` `max\_dets` `` 已经含 `\_`，若把 `\` 当字面量再转义 `_`，
              #   就得到 `max\\_dets`。把 `\_`/`\%`/`\&` 视为"已完成转义"即可幂等。
              "_", "%", "&", "$", "#", "{", "}")
_CMD_RE = re.compile(r"\\(" + "|".join(KNOWN_CMDS) + r")\*?(?:\{[^{}]*\})*")


# ══════════════════════════════════════════════════════════════════
# P110：反引号片段分类 —— 数学 vs 代码
# ══════════════════════════════════════════════════════════════════
# 原来所有反引号内容一律转 `\texttt{}`。对 `max_det`（真代码）正确，
# 但 `n_i` 是**数学变量**、`util(B) = Σ_i min(B, n_i) / (B·N)` 是**公式**，
# 包成等宽代码是错的排版。用户提问"能不能写角标"引出此修复。
_MATH_CODE_EXACT = {
    "max_det", "max_dets", "maxDets", "ceiling", "extra.ignore",
    "science-top-journal-writing", "academic-humanizer", "densescout2026",
}
_MATH_PATH_HINTS = ("results/", "runs/", "src/", "docs/")
_MATH_SIGNALS = ("Σ", "≥", "≤", "·", "⌈", "⌉", "ρ",
                 "Δ", "=", "frac(", "P(", "TP_")
_MATH_FUNC_CALL = re.compile(r"\b(?:min|max)\s*\(")
_MATH_SUB = re.compile(r"[A-Za-z]_[A-Za-z0-9](?![A-Za-z0-9])")
_MATH_SINGLE = re.compile(r"^[A-Za-z]$")


def _is_math_span(code: str) -> bool:
    """这个反引号片段应当排成数学，而不是等宽代码？"""
    c = code.strip()
    if not c:
        return False
    if any(h in c for h in _MATH_PATH_HINTS):
        return False
    if c in _MATH_CODE_EXACT:
        return False
    if re.search(r"max[_]?dets?\b", c, re.I):
        return False
    if _MATH_SINGLE.match(c):
        return True
    if _MATH_SUB.search(c):
        return True
    if _MATH_FUNC_CALL.search(c):
        return True
    if any(sig in c for sig in _MATH_SIGNALS):
        return True
    return False


_MC_UNI = {
    "≤": r"\leq{}", "≥": r"\geq{}", "×": r"\times{}",
    "−": "-", "≈": r"\approx{}", "→": r"\to{}", "∈": r"\in{}",
    "∑": r"\sum", "Σ": r"\Sigma", "Δ": r"\Delta",
    "ρ": r"\rho", "α": r"\alpha", "µ": r"\mu",
    "·": r"\cdot{}", "≠": r"\neq{}",
    "⌈": r"\lceil{}", "⌉": r"\rceil{}",
    "⌊": r"\lfloor{}", "⌋": r"\rfloor{}",
    "ᵢ": "_i", "ⱼ": "_j", "²": "^2", "³": "^3",
}


def _span_to_math(code: str) -> str:
    """把一个数学片段转成 LaTeX 数学模式内容（**不含**首尾 $）。"""
    s = code.strip()
    # ① 先处理需要特殊结构的
    s = s.replace("Σ_i", r"\sum_i").replace("Σᵢ", r"\sum_i")
    s = re.sub(r"Δ\s*(macro|micro)", r"\\Delta_{\\mathrm{\1}}", s)
    # ② Unicode→命令（\cdot{} 形式，避免 \cdotN 这类未定义命令）
    for k, v in _MC_UNI.items():
        s = s.replace(k, v)
    # ③ 名字类 → \mathrm
    for w in ("frac", "util", "gain", "spend", "micro", "macro"):
        s = re.sub(rf"\b{w}\b(?!\}})", rf"\\mathrm{{{w}}}", s)
    s = re.sub(r"\bTP(?=[_/ \)])", r"\\mathrm{TP}", s)
    # ④ 算子名（含下标形式 min_i）
    s = re.sub(r"(?<![\\A-Za-z])\bmin(?=[_\(\s])", r"\\min", s)
    s = re.sub(r"(?<![\\A-Za-z])\bmax(?=[_\(\s])", r"\\max", s)
    s = s.replace(r"\mathrm{\\mathrm{", r"\mathrm{")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def spans_to_latex(text: str) -> str:
    """把一行/一段 markdown 转成 LaTeX：单遍扫描，按 span 处理。

    span 类型（四类）：
      · `$...$`   —— 已是数学，**逐字保留**
      · `` `...` `` —— 代码，内部做 Unicode→数学 与 文本转义，再包 `\\texttt{}`
      · `\\cmd{...}` —— **已生成的 LaTeX 命令，逐字保留**
      · 其余       —— 纯文本：Unicode→行内数学，转义特殊字符
    强调（`**` / `*`）在**全部 span 处理完之后**套用，
      这样 `\\textbf{}` 的 `}` 永不会被 `%` 注释掉。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "$":                       # 数学段：找到配对的 $
            j = text.find("$", i + 1)
            if j < 0:
                out.append(_esc_plain(c))
                i += 1
                continue
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if c == "`":                       # 代码段
            j = text.find("`", i + 1)
            if j < 0:
                out.append(_esc_plain(c))
                i += 1
                continue
            code = text[i + 1:j]
            if _is_math_span(code):
                # ★ P110：数学变量/公式 -> $...$（真下标），而非等宽代码
                out.append("$" + _span_to_math(code) + "$")
                i = j + 1
                continue
            seg, math = _uni_to_math(code)
            seg = _esc_plain(seg)
            seg = _restore_math_placeholders(seg, math)
            if "\x01" in seg:
                raise SystemExit(f"❌ 占位符残留: {seg!r}")
            out.append(r"\texttt{" + seg + "}")
            i = j + 1
            continue
        if c == "\\":                      # ★ 已生成的 LaTeX 命令：逐字保留
            mc = _CMD_RE.match(text, i)
            if mc:
                out.append(mc.group(0))
                i = mc.end()
                continue
            out.append(_esc_plain(c))
            i += 1
            continue
        # 纯文本：收集到下一个 $ / ` / \
        j = i
        while j < n and text[j] not in "$`\\":
            j += 1
        if j == i:                          # 单独的 `\`（非命令）
            out.append(_esc_plain(text[i]))
            i += 1
            continue
        seg, math = _uni_to_math(text[i:j])
        seg = _esc_plain(seg)
        seg = _restore_math_placeholders(seg, math)
        out.append(seg)
        i = j
    s = "".join(out)
    # 强调：此时所有特殊字符都已转义，`}` 安全
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s, flags=re.S)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", s, flags=re.S)
    s = re.sub(r"\*{1,2}", "", s)
    return s



def _restore_math_placeholders(seg: str, math: list[str]) -> str:
    r"""把占位符**单遍**换回 `$...$`。

    ★★ P71：**不能顺序替换**（`for k, m in enumerate(math): seg.replace(...)`）——
      先生成的替换结果会「吃掉」后面的占位符。
      实例：手稿的 `` `⌈ρ·n_i⌉ = 0` `` 触发 `\lceil` 与 `\rceil` **两个**占位符；
      k=0 替换后，字符串里的占位符 1 被拆开，k=1 再也匹配不上 →
      **控制字符泄漏进 .tex** → pdflatex 报 `Unicode character ^^A (U+0001)`。
      单遍正则一次到位，并在最后**断言无残留**（fail loud，不静默产出坏 .tex）。

    ★ 另一个坑：正则必须匹配**真正的控制字符**（即 `chr(1)`），
      不能写成 `"\\x01"` —— 那是「字面反斜杠 + x01」，永远匹配不到。
    """
    def _sub(m):
        k = int(m.group(1))
        if k >= len(math):
            raise SystemExit(f"\u274c 占位符索引越界: {k} / {len(math)}")
        return "$" + math[k] + "$"

    out = re.sub(chr(1) + "(\\d+)" + chr(1), _sub, seg)
    if chr(1) in out:
        raise SystemExit(f"\u274c 占位符未清空: {out!r}")
    return out


if __name__ == "__main__":
    TESTS = [
        (r"the gap is $\le$ 1.41 points", None),
        (r"**95% confidence** interval", None),
        (r"`max_dets = 300` and `B = 25`", None),
        (r"$\Delta$**micro** is 1.41", None),
        (r"min(B, n_i)/(B·N) with `min(B, n_i)`", None),
        (r"**+8.33 points** on the one-to-many head", None),
        (r"$\rho$ = 0.5, `max_dets = 300`", None),
        (r"quintile's change is $> 0$ for all", None),
        (r"*Seed 42 fully reported; seed 43 added.*", None),
        (r"`util(B) = \Sigma_i min(B, n_i) / (B*N)`", None),
    ]
    print("=" * 92)
    print("span 处理器自测")
    print("=" * 92)
    for raw, _ in TESTS:
        got = spans_to_latex(raw)
        bad = []
        if re.search(r"(?<!\\)%", got):
            bad.append("裸 %")
        if "\\textbackslash" in got:
            bad.append("textbackslash")
        if got.count("{") != got.count("}"):
            bad.append(f"花括号不配对 {got.count('{')}/{got.count('}')}")
        if "\x00" in got or "\x01" in got:
            bad.append("占位符残留")
        flag = "✅" if not bad else "❌ " + ",".join(bad)
        print(f"  {flag}  {raw}")
        print(f"        → {got}")
    print()
    print("  说明：以上 10 例覆盖 数学/代码/强调/Unicode/百分号 的组合。")
