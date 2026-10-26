"""补充表引用必须带 "Supplementary"（Sci Rep 官方要求）。

官方原文："Be sure to include the word 'Supplementary' each time one is mentioned."

第 9 轮发现：`Table S7` 只以裸写出现一次、`Table S6` 也有一次裸写 ——
读者/审稿人按 "Supplementary Table S7" 检索会找不到。

用法：python scripts/check_supplementary_wording.py
退出码：0 = 全部合规；1 = 有裸写
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "docs" / "manuscript-results-methods.md",
         ROOT / "docs" / "manuscript-abstract-intro-discussion.md"]

# 中文笔记块（`# Notes for the next revision` 之后）不计入
CUT = "# Notes for the next revision"


def main() -> int:
    fails = []
    for f in FILES:
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8")
        k = t.find(CUT)
        body = t[:k] if k > 0 else t
        for m in re.finditer(r"Table (S\d+)\b", body):
            pre = body[max(0, m.start() - 30):m.start()]
            if "Supplementary" not in pre:
                fails.append(f"{f.name}: `Table {m.group(1)}` 缺 Supplementary")
        for m in re.finditer(r"Figure (S\d+)\b", body):
            pre = body[max(0, m.start() - 30):m.start()]
            if "Supplementary" not in pre and "Fig." not in pre:
                fails.append(f"{f.name}: `Figure {m.group(1)}` 缺 Supplementary")

    print("=" * 78)
    print("补充材料引用措辞（每次提及都要带 Supplementary）")
    print("=" * 78)
    if fails:
        print(f"  [FAIL] {len(fails)} 处裸写:")
        for x in fails:
            print(f"     {x}")
        return 1
    print("  [OK] 全部补充表/图引用都带 Supplementary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
