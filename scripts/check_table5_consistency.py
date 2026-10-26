#!/usr/bin/env python
"""scripts/check_table5_consistency.py —— 手稿 Table 5 与生成脚本的一致性检查（P124）

Table 5 的布尔列曾长期是**手工填的**，与数据脱节而 14 项自动检查全部放行。
本检查把手稿的四个布尔列与 table5-structural.json 逐格比对，任何不一致即失败。

用法：python scripts/check_table5_consistency.py
退出码：0 = 一致；1 = 有差异
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "tables" / "table5-structural.json"
DOC = ROOT / "docs" / "manuscript-results-methods.md"


def norm(s: str) -> str:
    return s.strip().strip("*").strip().lower()


def main() -> int:
    if not SRC.exists():
        print(f"  ❌ 缺 {SRC.relative_to(ROOT)}（先跑 build_table5_structural.py）")
        return 1
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = {r["B"]: r for r in data["heads"]["o2m"]["rows"]}

    # ★ 第 10 版：Table 5 已迁入补充材料作为 Supplementary Table S10
    #   （Sci Rep "Display items are limited to 8"，主文原 6 图 + 5 表 = 11 项超限）。
    #   表体原文在搬迁时已存入 moved-tables.json，故改从那里读取。
    moved_f = ROOT / "results" / "tables" / "moved-tables.json"
    blk = None
    if moved_f.exists():
        mv = json.loads(moved_f.read_text(encoding="utf-8"))
        if "5" in mv and mv["5"].get("rows"):
            blk = "\n".join(mv["5"]["rows"])
    if blk is None:
        md = io.open(DOC, encoding="utf-8").read()
        i = md.find("| `B` | identical vectors")
        if i < 0:
            print("  ❌ 既不在 moved-tables.json，也不在主稿里找到该表")
            return 1
        blk = md[i:md.find("\n\n", i)]

    print("=" * 78)
    print("Supplementary Table S10（原 Table 5）一致性：表体 vs 生成脚本")
    print("=" * 78)
    print(f"  {'B':>4}{'identical':>16}{'cap(u/p)':>18}{'degenerate':>14}")
    print("  " + "-" * 56)

    bad = 0
    seen = 0
    for line in blk.split("\n")[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        try:
            B = int(norm(cells[0]))
        except ValueError:
            continue
        r = rows.get(B)
        if r is None:
            continue
        seen += 1
        # P126：表注与正文说的是「the same image is the worst
        #   under both allocations」（identity），故按 `worst_same_image` 核对，
        #   不是按 at-cap。两者当前数据下等价，但定义必须对得上。
        same = ("yes" if r["worst_same_image"] else "no")
        want = (
            "yes" if r["identical_vectors"] else "no",
            same + " / " + same,
            "yes" if r["degenerate"] else "no",
        )
        got = (norm(cells[1]), norm(cells[2]).replace(" ", ""), norm(cells[3]))
        want_n = (want[0], want[1].replace(" ", ""), want[2])
        ok = got == want_n
        if not ok:
            bad += 1
        mark = "✅" if ok else "❌"
        print(f"  {B:>4}{mark}{got[0]:>14}{got[1]:>18}{got[2]:>14}"
              + ("" if ok else f"   应为 {want_n}"))

    print()
    if seen != len(rows):
        print(f"  ❌ 手稿里只找到 {seen} 行，生成脚本有 {len(rows)} 行")
        return 1
    if bad:
        print(f"  ❌ {bad} 处不一致 —— **手稿的布尔列与数据脱节**，必须手工列改为脚本生成")
        return 1
    # P126：identity 与 at-cap 两种定义必须当前等价（否则表注描述会失真）
    disagree = [r["B"] for r in rows.values() if not r.get("definitions_agree", True)]
    if disagree:
        print(f"  ⚠️ B={disagree} 处 identity 与 at-cap 两种定义不再等价："
              "\n     表注写的是 identity，若 at-cap 不同则必须改表注")
        return 1
    print(f"  ✅ 全部 {seen} 行四列一致；"
          f"identity 与 at-cap 两种定义当前等价")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
