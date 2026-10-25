#!/usr/bin/env python
"""
scripts/check_boundary_condition.py —— **无条件声称检查器**

═══ 为什么需要（P79）═══════════════════════════════════════════════════
test-dev 的实测结果推翻了"重分配提升最差图召回"这个**无条件**主张（Δworst = +0.000）。
A+ 改写把它降级为**有条件**主张：只在最差图"配额受限"时成立。

但这类"降级"极易在后续编辑中被稀释：某处又写成 "reallocation raises worst-image recall"，
不带条件限定，读者就会以为这是普遍结论。**这与种子数是同一类问题**：
改了一个口径，漏了某些载体。所以做成检查器。

规则：
  ① 任何"提升最差图召回"的**正面主张**，所在句子必须含**条件限定词**
     （validation / quota-limited / bounded / only when / condition / replaces）
  ② 禁止出现**无条件**的强主张短语
  ③ 允许"机制"类陈述无条件（守恒、Δmacro 崩塌、分级代价——这些在两个数据集都成立）

用法：python scripts/check_boundary_condition.py
退出码：0 = 合规；1 = 有未限定的强主张
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "manuscript-results-methods.md",
        ROOT / "docs" / "manuscript-abstract-intro-discussion.md",
        ROOT / "docs" / "submission-package.md"]

# 限定词：句子含其一，才允许出现"提升最差图召回"类主张
QUALIFIERS = re.compile(
    r"validation|quota-limited|bounded|only when|only if|condition|"
    r"replaces|replacement|positional|vanish|depends on|"
    r"test-dev|does not generalise|not unconditio|not a general|not on the held-out|should be read|constrain",
    re.I,
)

# 正面主张的动词短语（出现即要求同句有 qualifier）
CLAIM = re.compile(
    r"(raises?|improves?|increases?|gains?|boost)\s+"
    r"(the\s+)?(worst[- ]image|worst[- ]case|minimum)\s+(recall)?",
    re.I,
)

# ★ 无条件强主张黑名单（出现即错，必须改写）
BANNED = [
    (r"reallocation\s+(always\s+)?improves\b", "reallocation always improves"),
    (r"we\s+show\s+that\s+reallocating\s+improves\b", "we show that reallocating improves"),
    (r"universally\s+improves", "universally improves"),
    (r"improves\s+worst[- ]image\s+recall\s+in\s+general", "improves worst-image recall in general"),
]


def sentence_of(text: str, pos: int) -> str:
    """取 pos 所在的句子。

    ★ 句界 = `.`/`!`/`?` + 空白，**或表格行边界**（`|` 开头的行）。
      否则句子会从表格里开始（表格单元格没有句号），导致报错行号指向错误的行。
    """
    # 向左找最近的句界或表格行
    left = max(text.rfind(". ", 0, pos), text.rfind("! ", 0, pos), text.rfind("? ", 0, pos))
    # 若 pos 之前有表格行，则取更近者
    bar = text.rfind("\n|", 0, pos)
    if bar > left:
        left = text.find("\n", bar + 1)
    s = max(left + (1 if left >= 0 else 0), 0)

    m = re.search(r"[.!?](\s|$)", text[pos:])
    e = pos + (m.start() + 1 if m else 200)
    out = re.sub(r"\s+", " ", text[s:e]).strip().lstrip("|").strip()
    return out[:260]


def main() -> int:
    problems: list[str] = []
    n_claims = 0
    n_ok = 0

    for f in DOCS:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        # 跳过 Figure legends / Notes 段（图注与修订记录不是主张）
        for cut in ("# Figure legends", "# Notes for the next revision"):
            k = text.find(cut)
            if k >= 0:
                text = text[:k]

        for pat, label in BANNED:
            for m in re.finditer(pat, text, re.I):
                ln = text[:m.start()].count("\n") + 1
                problems.append(f"{f.name}:L{ln} 无条件强主张：{label}")

        # ★ 跳过 Markdown 表格：表格里的是**列标签**，不是主张。
        #   注意：一句里可能跨行匹配，所以要检查匹配点**所在行及其前后一行**。
        _lns = text.split("\n")
        for m in CLAIM.finditer(text):
            ln0 = text[:m.start()].count("\n")
            if any(_lns[k].lstrip().startswith("|")
                   for k in (ln0 - 1, ln0, ln0 + 1) if 0 <= k < len(_lns)):
                continue
            n_claims += 1
            sent = sentence_of(text, m.start())
            # ★ 提问不是主张：封面信常用「does reallocating ... raise worst-image recall?」
            #   来提出研究问题，这种句子不需要条件限定。
            if "?" in sent or re.search(r"\bdoes\b|\bwhether\b|question", sent, re.I):
                continue
            if QUALIFIERS.search(sent):
                n_ok += 1
            else:
                ln = text[:m.start()].count("\n") + 1
                problems.append(f"{f.name}:L{ln} 主张缺条件限定：{sent[:150]}")

    print("=" * 100)
    print("「提升最差图召回」类主张的条件限定检查")
    print("=" * 100)
    print(f"  检出正面主张 {n_claims} 处，其中 {n_ok} 处带条件限定")
    print()
    if problems:
        print(f"  ❌ 发现 {len(problems)} 处问题：")
        for x in problems:
            print(f"     {x}")
        print("\n  ★ 处置：给该句补条件（validation / quota-limited / replaces / vanishes on test-dev），")
        print("    或改成机制类陈述（守恒 / Δmacro 崩塌 / 分级代价——这些无条件成立）。")
        return 1
    print("  ✅ 全部主张都带条件限定，或属于无条件成立的机制陈述")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
