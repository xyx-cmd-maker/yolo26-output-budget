"""SI 条目完整性：正文引用的每个 S 条目都必须在 SI 里带编号出现。

第 8 轮教训（P177）：Figure S1/S2 曾经是**没有编号与图注的裸图**，
而 Discussion 引用了它们各 2 次。20 项自动检查当时全绿 —— 因为没人查这一层。

用法：python scripts/check_si_items.py
退出码：0 = 全部对得上；1 = 有缺失
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "submission-manuscript.pdf"
SUP = ROOT / "results" / "supplementary-information.pdf"


def main() -> int:
    if not MAIN.exists() or not SUP.exists():
        print("  缺 PDF，先重建")
        return 1
    d = fitz.open(MAIN)
    main_txt = "".join(pg.get_text() for pg in d)
    d.close()
    d = fitz.open(SUP)
    sup_txt = "".join(pg.get_text() for pg in d)
    d.close()

    # Sci Rep 七个补充材料类别（官方逐字）：
    #   Table, Figure, Video, Audio, Note, Data, Discussion, Equations, Methods
    CATS = ["Table", "Figure", "Video", "Audio", "Note", "Data", "Discussion",
            "Equations", "Methods"]
    cited = {}
    for cat in CATS:
        if cat == "Figure":
            pat = r"Supplementary (?:Figure|Fig\.?) (S\d+)"
        else:
            pat = rf"Supplementary {cat} (S\d+)"
        cited[cat] = sorted(set(re.findall(pat, main_txt)))
    cited_tab = cited["Table"]
    cited_fig = sorted(set(cited["Figure"]) | set(
        re.findall(r"Supplementary Fig\. ?(S\d+)", main_txt)))
    cited_note = cited["Note"]
    print("=" * 78)
    print("SI 条目完整性")
    print("=" * 78)
    print(f"  正文引用表: {cited_tab}")
    print(f"  正文引用图: {cited_fig}")
    print(f"  正文引用 note: {cited_note}")

    fails = []
    for t in cited_tab:
        if f"Table {t}" not in sup_txt:
            fails.append(f"Table {t} 被引用但 SI 无编号")
    for f in cited_fig:
        if f"Figure {f}" not in sup_txt:
            fails.append(f"Figure {f} 被引用但 SI 无编号")
    for n in cited_note:
        if f"Note {n}" not in sup_txt:
            fails.append(f"Note {n} 被引用但 SI 无编号")
    # ★ 其它类别：被引用就必须在 SI 里有同类别同编号的条目
    for cat in ("Video", "Audio", "Data", "Discussion", "Equations", "Methods"):
        for n in cited.get(cat, []):
            if f"{cat} {n}" not in sup_txt:
                fails.append(f"{cat} {n} 被引用但 SI 无此条目")

    # 反向：SI 里的每个表/图都必须有简短标题（编号后紧跟文字）
    print()
    n_tab = len(re.findall(r"Table S\d+\.", sup_txt))
    n_fig = len(re.findall(r"Figure S\d+\.", sup_txt))
    print(f"  SI 内有标题的表 {n_tab} 个、图 {n_fig} 个")
    if n_tab < len(cited_tab):
        fails.append(f"SI 表标题数 {n_tab} < 正文引用数 {len(cited_tab)}")
    if n_fig < len(cited_fig):
        fails.append(f"SI 图标题数 {n_fig} < 正文引用数 {len(cited_fig)}")

    print()
    if fails:
        print(f"  [FAIL] {len(fails)} 项:")
        for f in fails:
            print(f"     {f}")
        return 1
    print("  [OK] 正文引用与 SI 条目完全对应")
    return 0


if __name__ == "__main__":
    sys.exit(main())
