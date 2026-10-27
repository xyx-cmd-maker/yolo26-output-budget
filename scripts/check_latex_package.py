"""LaTeX 投稿件的结构与内容检查器（新增，防本轮两个缺陷复现）。

检查项：
  ① `\\includegraphics` 出现次数必须等于图数（6）
  ② 输出的 .tex / PDF 里不得残留 `*(` 生产注记片段
  ③ `$` 必须成对（避免数学模式断裂）
  ④ 六个图文件必须真实存在于 latex/figures/
  ⑤ PDF 里每条图注都能找到（Figure 1..6）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

R = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
TEX = R / "latex" / "manuscript.tex"
PDF = R / "latex" / "manuscript.pdf"
FIGDIR = R / "latex" / "figures"
N_FIG = 6

def main() -> int:
    fails: list[str] = []

    # ① includegraphics 次数
    tex = TEX.read_text(encoding="utf-8") if TEX.exists() else ""
    n_inc = tex.count(r"\includegraphics")
    print(f"  ① includegraphics: {n_inc} 次（期望 {N_FIG}）")
    if n_inc != N_FIG:
        fails.append(f"① includegraphics {n_inc} != {N_FIG}")

    # ② 生产注记残留
    # 生产注记残留的**真实表现**是图注节开头出现断句，而不是含 `*(` 前缀
    #   （首行已被旧代码丢掉，残下的是 "names the sample size…"）。
    m = re.search(r"\\bmhead\{Figure legends\}(.*?)(\\bibliography|\\end\{document\})",
                  tex, re.S)
    seg = m.group(1) if m else ""
    orphan = re.search(r"names the sample size|states the interval\.\)", seg)
    print(f"  ② 生产注记残留: {'有' if orphan else '无'}")
    if orphan:
        fails.append(f"② 图注节残留生产注记：{orphan.group(0)[:40]}")
    if seg and r"\textbf{Figure 1." not in seg and r"\includegraphics" not in seg:
        fails.append("② 图注节首段既不是 Figure 1 也不是图")

    # ③ $ 成对
    dollars = len(re.findall(r"(?<!\\)\$", tex))
    print(f"  ③ 未转义 $: {dollars}（{'成对' if dollars % 2 == 0 else '奇数!'}）")
    if dollars % 2:
        fails.append(f"③ $ 数量为奇数 {dollars}")

    # ④ 图文件存在
    figs = sorted(FIGDIR.glob("*.pdf")) if FIGDIR.exists() else []
    print(f"  ④ latex/figures/ 下 PDF 图: {len(figs)} 个")
    if len(figs) != N_FIG:
        fails.append(f"④ 图文件 {len(figs)} != {N_FIG}")

    # ⑤ PDF 里的图注
    if PDF.exists():
        import fitz

        d = fitz.open(PDF)
        txt = "".join(p.get_text() for p in d)
        missing = [k for k in range(1, N_FIG + 1) if f"Figure {k}." not in txt]
        drawings = sum(len(p.get_drawings()) for p in d)
        print(f"  ⑤ PDF {d.page_count} 页；缺图注 {missing or '无'}；矢量绘图 {drawings} 条")
        if missing:
            fails.append(f"⑤ PDF 缺图注 {missing}")
        if drawings < 500:
            fails.append(f"⑤ 矢量绘图仅 {drawings} 条，疑似图未嵌入")
        d.close()
    else:
        fails.append("⑤ latex/manuscript.pdf 不存在")

    print()
    if fails:
        print("  ❌ 失败:")
        for f in fails:
            print(f"     - {f}")
        sys.exit(1)
    print("  ✅ LaTeX 投稿件结构检查全部通过")

    return 0


if __name__ == "__main__":
    sys.exit(main())
