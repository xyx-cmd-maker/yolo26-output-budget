#!/usr/bin/env python
"""
scripts/check_seed_consistency.py —— **种子数声明必须与数据一致**

═══ 为什么需要（P77/P78）═══════════════════════════════════════════════
同一个错误连续犯了**三次**：
  ① 正文叙述仍写 "Two training seeds; a third is in progress"，而摘要已写 "three"
  ② Table 2 与 Table 5 的每格仍是 `s42 / s43`，而正文写 "across the three seeds"
  ③ Table 4 的表头仍是 `(s42 / s43)`，而正文写 "on all three seeds"
每次都是"改了口径却漏了某些载体"。**靠记忆去查必然漏**，所以做成检查器。

本检查做四件事：
  ① 从 `manuscript-numbers.json` 读出**实际**种子数（headline 的顶层键）
  ② 扫描手稿里所有"种子数声明"（two/three seeds、a third/fourth、seed 42/43/44 …）
  ③ 扫描所有表格表头里的种子标注（`(s42 / s43)`、`(seed 42 / 43)` …）
  ④ 任何载体声明的种子数 ≠ 实际种子数 → 报错并指出行号

用法：python scripts/check_seed_consistency.py
退出码：0 = 一致；1 = 有矛盾
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "manuscript-results-methods.md",
        ROOT / "docs" / "manuscript-abstract-intro-discussion.md",
        ROOT / "docs" / "submission-package.md"]

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def actual_seeds() -> int:
    d = json.loads((ROOT / "results" / "tables" / "manuscript-numbers.json").read_text(encoding="utf-8"))
    return len(d["headline"])


# ═══ 已知的**合法**用法白名单（不是种子数声明）═════════════════════════
# 每条写明为什么合法，避免下次又要重新判断一遍。
WHITELIST = [
    # 「four seed-by-head conditions」数的是**条件**（3 种子 × 2 头中的 4 个），不是种子数
    (r"four\s+seed-by-head", "数条件，不是种子"),
    # 「re-running one seed alone」说的是**重跑单个种子**这件事，不是声明总种子数
    (r"re-running\s+one\s+seed\s+alone", "说的是重跑单种子，不是声明总数"),
    # 「single seed」在局限性里指"单个种子的比较"，同理
    (r"single\s+seed", "单种子比较，不是声明总数"),
    # 「从 single seed 改为 two seeds」是**修订记录**（Notes 段），不是正文声明
    (r"Limitations\s+从", "修订记录"),
]


def _whitelisted(line: str) -> str | None:
    for pat, why in WHITELIST:
        if re.search(pat, line, re.I):
            return why
    return None


def main() -> int:
    n_actual = actual_seeds()
    print("=" * 100)
    print(f"实际种子数（manuscript-numbers.json 的 headline 键数）= {n_actual}")
    print("=" * 100)

    problems: list[str] = []

    # ① 散文里的数字声明
    DECL = re.compile(r"\b(one|two|three|four|five)\s+(?:independent\s+)?(?:training\s+)?seeds?\b", re.I)
    # ② "a third/fourth is [in progress|training|complete]"
    ORD = re.compile(r"\ba\s+(third|fourth|fifth)\b[^.\n]{0,60}?(progress|training|complete|reported)", re.I)
    # ③ 表头里的种子标注
    HDR = re.compile(r"\((?:seed\s*)?42\s*/\s*43(?:\s*/\s*44)?\)|\(s42\s*/\s*43(?:\s*/\s*44)?\)")

    for f in DOCS:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            why = _whitelisted(line)
            if why:
                continue  # 合法用法，跳过

            # ①
            for m in DECL.finditer(line):
                n = WORD_NUM[m.group(1).lower()]
                if n != n_actual:
                    problems.append(f"{f.name}:L{i} 声明 {m.group(1)} seeds，实际 {n_actual}"
                                    f"  →  {line.strip()[:90]}")
            # ②
            for m in ORD.finditer(line):
                problems.append(f"{f.name}:L{i} 提到「a {m.group(1)} {m.group(2)}」，"
                                f"实际已完整 {n_actual} 个种子  →  {line.strip()[:90]}")
            # ③
            for m in HDR.finditer(line):
                got = len(re.findall(r"4[234]", m.group(0)))
                if got != n_actual:
                    problems.append(f"{f.name}:L{i} 表头只标了 {got} 个种子，实际 {n_actual}"
                                    f"  →  {line.strip()[:90]}")

    print()
    if problems:
        print(f"  ❌ 发现 {len(problems)} 处种子数声明与数据不一致：")
        for x in problems:
            print(f"     {x}")
        print("\n  ★ 修法：改**声明**去匹配数据，不要改数据去匹配声明；")
        print("    表格补种子时用 manuscript-numbers.json / seed-replication.json，不要手打。")
        return 1
    print(f"  ✅ 全部载体（散文 / 表头）声明的种子数都等于实际的 {n_actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
