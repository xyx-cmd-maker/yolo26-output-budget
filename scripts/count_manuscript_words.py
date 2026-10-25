"""
scripts/count_manuscript_words.py —— 手稿字数的**唯一权威计数**（按 Sci Rep 官方口径）

═══════════════════════════════════════════════════════════════════════════
官方口径（原文，见 docs/venue-format-verification-srep-official.md）
═══════════════════════════════════════════════════════════════════════════
  "the main text should be no more than 4,500 words, **not including the abstract,
   Methods, references, figure legends, and tables**"（以官方核实文档为准）

→ **计入**：Introduction + Results 正文（含子标题、图注引用句、表标题句）
→ **不计**：Abstract、Methods、References、**Figure legends**、**表格内容**、数据可用性声明

═══════════════════════════════════════════════════════════════════════════
为什么需要这个脚本（P47 教训）
═══════════════════════════════════════════════════════════════════════════
我曾在一个内联 python -c 里用 `chr(10)+'# '` 做"查找换行后的 #"，
但在**单行字符串**上永远匹配不到 → section 边界取成了"文件末尾"
→ 每个 section 都吃掉了后面所有内容 → 字数完全失真（先报 3,496，后报 4,434）。
本脚本用**按行遍历**，不再依赖跨行正则。

用法：
    python scripts/count_manuscript_words.py
    python scripts/count_manuscript_words.py --limit 4500
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = (
    ("Abstract/Intro/Discussion", ROOT / "docs" / "manuscript-abstract-intro-discussion.md"),
    ("Results/Methods", ROOT / "docs" / "manuscript-results-methods.md"),
)

# 顶层节（`# X`）中**不计入**的
SKIP_H1 = {"methods", "figure legends", "data and code availability", "references",
           "notes for the next revision"}
# 计入的节
COUNT_H1 = {"introduction", "results", "discussion"}

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def is_meta(line: str) -> bool:
    """引用块、内部注记、分隔线、markdown 标题行本身都不计。"""
    s = line.strip()
    return (not s) or s.startswith((">", "*(*", "---", "#")) or re.fullmatch(r"\*\(.*\)\*", s)


def count_section(lines: list[str]) -> tuple[int, int]:
    """返回 (含表格词数, 去表格词数)。"""
    prose, with_tab = 0, 0
    for line in lines:
        if is_meta(line):
            continue
        s = line.strip()
        if s.startswith("|"):                      # 表格内容 → 官方口径不计
            with_tab += len(WORD.findall(s))
            continue
        n = len(WORD.findall(s))
        prose += n
        with_tab += n
    return with_tab, prose


def main() -> int:
    ap = argparse.ArgumentParser(description="手稿字数（Sci Rep 口径）")
    ap.add_argument("--limit", type=int, default=4500)
    a = ap.parse_args()

    total_prose = 0
    print("=" * 96)
    print("手稿字数 —— Sci Rep 官方口径（不计 Abstract / Methods / legends / 表格 / 数据声明）")
    print("=" * 96)

    for tag, path in FILES:
        if not path.exists():
            print(f"  ❌ 缺 {path}")
            return 1
        lines = path.read_text(encoding="utf-8").split("\n")
        # 按 '# ' 切顶层节
        secs: list[tuple[str, list[str]]] = []
        cur_name, cur = "(前言)", []
        for line in lines:
            if line.startswith("# ") and not line.startswith("## "):
                secs.append((cur_name, cur))
                cur_name, cur = line[2:].strip(), []
            else:
                cur.append(line)
        secs.append((cur_name, cur))

        print(f"\n### {tag}  ({path.name})")
        for name, body in secs:
            if not body and name == "(前言)":
                continue
            with_tab, prose = count_section(body)
            low = name.lower()
            counted = any(k in low for k in COUNT_H1) and not any(k in low for k in SKIP_H1)
            flag = "计入" if counted else "不计"
            print(f"  [{flag}] {prose:>5} 词(去表) / {with_tab:>5} 词(含表)   {name[:52]}")
            if counted:
                total_prose += prose

    print("\n" + "=" * 96)
    print(f"★ 计入上限的部分 = {total_prose} 词   上限 {a.limit}   余量 {a.limit - total_prose}")
    print("=" * 96)
    if total_prose > a.limit:
        print(f"  ❌ **超限 {total_prose - a.limit} 词**，必须压缩")
        return 2
    print(f"  ✅ 合规，余量 {a.limit - total_prose} 词")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
