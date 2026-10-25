#!/usr/bin/env python
"""scripts/check_caption_claims.py —— 图注断言 vs 实现的登记册检查（P129）

═══ 为什么需要 ═══════════════════════════════════════════════════════════
图注是一份**独立于代码的叙述**，它会与实现脱节。实测已抓到：Figure 2 的 crosses
标的是 `N1==False`（不可花费预算），而图注写成"低配额"。15 项自动检查都不查这个。

═══ 机制 ═════════════════════════════════════════════════════════════════
`results/tables/figure-caption-claims.json` 登记每一条图注断言，并指向一个
**真实存在的**文件与其中必须出现的字符串。检查器逐条核对：
  1. 源文件是否存在
  2. 源文件是否真的含有 `must_contain`
  3. 数字类断言的值是否仍出现在对应图注里

用法：python scripts/check_caption_claims.py
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results" / "tables" / "figure-caption-claims.json"
DOC = ROOT / "docs" / "manuscript-results-methods.md"


def caption_text(doc: str, n: int) -> str:
    m = re.search(r"\*\*Figure " + str(n) + r"\.", doc)
    if not m:
        return ""
    seg = doc[m.start():]
    end = seg.find("**Figure " + str(n + 1) + ".")
    if end < 0:
        end = seg.find("\n## ")
    return re.sub(r"\s+", " ", seg[:end if end > 0 else 1200])


def main() -> int:
    if not REG.exists():
        print(f"  ❌ 缺登记册 {REG.relative_to(ROOT)}")
        return 1
    reg = json.loads(REG.read_text(encoding="utf-8"))
    doc = io.open(DOC, encoding="utf-8").read()

    print("=" * 80)
    print("图注断言 vs 实现 的一致性检查")
    print("=" * 80)

    bad = 0
    for e in reg["entries"]:
        fig = e["figure"]
        src = ROOT / e["file"]
        cap = caption_text(doc, int(fig))
        problems = []

        # ① 源文件存在
        if not src.exists():
            problems.append(f"源文件不存在 {e['file']}")
        else:
            body = src.read_text(encoding="utf-8", errors="replace")
            # ② 源文件确实含有该符号
            if e["must_contain"] not in body:
                problems.append(f"源文件里找不到 {e['must_contain']!r}")

        # ③ 数字类断言的值仍出现在图注里
        if e["kind"] == "number" and cap:
            m = re.match(r".*?([\d][\d,\.]*)", e["claim"])
            if m:
                token = m.group(1).replace(",", "")
                core = token.split(".")[0]
                if core and core not in cap.replace(",", ""):
                    problems.append(f"图注里已找不到数值 {core}")

        if problems:
            bad += 1
            print(f"\n  ❌ Figure {fig} · {e['kind']}")
            print(f"     断言: {e['claim']}")
            for p in problems:
                print(f"     · {p}")
        else:
            print(f"  ✅ Figure {fig} · {e['kind']:<9} {e['claim'][:58]}")

    print()
    if bad:
        print(f"  ❌ {bad} 条图注断言与实现/数据脱节 —— 必须改图注或改代码")
        return 1
    print(f"  ✅ 全部 {len(reg['entries'])} 条图注断言都能追溯到真实存在的代码符号或数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
