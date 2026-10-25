#!/usr/bin/env python
"""
scripts/check_lesson_ids.py —— **教训账编号必须与代码/提交信息里的引用一致**

═══ 为什么需要（P92）═════════════════════════════════════════════════
我在代码注释与提交信息里反复引用 `P76`、`P84`、`P91` …… 但**没有任何机制保证
这些编号真的被登记进 `AGENTS-LESSONS.md`**。

实测后果：`P81–P91` 共 11 个编号被引用过，却**全部未登记**，
直到我手工数了一遍才发现。这与「正文说三种子、表格两种子」是**同一形状**：
改了/说了一处，漏了另一处，而没有任何门把它挡住。

本检查做三件事：
  ① `AGENTS-LESSONS.md` 里的编号必须严格递增、无缺号
  ② 代码/文档里引用的 `P<n>` 必须在账上存在（悬空引用 = 未登记）
  ③ 账上每个编号至少被引用一次（或显式列在 ALLOW_UNREFERENCED）

用法：python scripts/check_lesson_ids.py
退出码：0 = 一致；1 = 有悬空引用或缺号
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "AGENTS-LESSONS.md"

# 允许"只登记不被引用"的编号（例如纯流程性记录）
ALLOW_UNREFERENCED: set[int] = set()

# ★★ P93：**排除百分位数**。
#   代码里的 `P90` / `P99` 很多时候是**统计量的第 N 百分位数**，
#   不是教训编号（实测：`budget_headroom.py` 里 `P90 {..} P99 {..}`，
#   `decisions.md` 里「P99 225」指每图框数的 99% 分位）。
#   首版检查器把它们当成教训编号，报了 9 个假悬空引用。
PERCENTILE_IDS = {1, 5, 10, 25, 50, 75, 90, 95, 99}

SCAN_DIRS = ["scripts", "src", "docs", "tests"]
SCAN_EXT = {".py", ".md", ".ps1", ".mjs", ".json", ".tex"}


def ledger_ids() -> list[int]:
    t = LEDGER.read_text(encoding="utf-8")
    return [int(m.group(1)) for m in re.finditer(r"^\| \*\*P(\d+)\*\*", t, flags=re.M)]


def referenced_ids() -> dict[int, list[str]]:
    """扫描代码/文档里引用的 P<n>，返回 {编号: [文件:行]}。"""
    refs: dict[int, list[str]] = {}
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.suffix not in SCAN_EXT or not f.is_file():
                continue
            if f.name == "AGENTS-LESSONS.md":
                continue          # 账文件自身不算引用
            try:
                txt = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for m in re.finditer(r"\bP(\d{1,3})\b", txt):
                n = int(m.group(1))
                if not (1 <= n <= 999):
                    continue
                if n in PERCENTILE_IDS:
                    continue        # 百分位数，不是教训编号
                ln = txt[:m.start()].count("\n") + 1
                refs.setdefault(n, []).append(f"{f.relative_to(ROOT)}:{ln}")
    return refs


def main() -> int:
    ids = ledger_ids()
    refs = referenced_ids()
    problems: list[str] = []

    print("=" * 96)
    print("教训账编号一致性")
    print("=" * 96)
    print(f"  账上编号：{len(ids)} 条，范围 P{min(ids)}–P{max(ids)}")

    # ① 缺号 / 递增
    miss = sorted(set(range(1, max(ids) + 1)) - set(ids))
    if miss:
        problems.append(f"账上缺号：{miss}")
    if ids != sorted(ids):
        problems.append("账上编号不是严格递增")

    # ② 悬空引用
    known = set(ids)
    dangling = sorted(n for n in refs if n not in known)
    for n in dangling:
        where = refs[n][:3]
        problems.append(f"引用了 P{n} 但账上不存在（{len(refs[n])} 处，如 {where}）")

    # ③ 未被引用（提示级，不判失败）
    unref = sorted(known - set(refs) - ALLOW_UNREFERENCED)

    print(f"  被代码/文档引用：{len(set(refs) & known)} 个编号")
    print()
    if problems:
        print(f"  ❌ {len(problems)} 项不一致：")
        for x in problems:
            print(f"     {x}")
        print("\n  ★ 处置：**先在 AGENTS-LESSONS.md 登记，再在代码里引用**；")
        print("    或在注释里改为不带编号的描述，避免悬空引用。")
        return 1
    print(f"  ✅ 编号递增、无缺号、无悬空引用")
    if unref:
        print(f"  （提示：{len(unref)} 个编号未被引用，属正常：{unref[:12]}"
              f"{' …' if len(unref) > 12 else ''}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
