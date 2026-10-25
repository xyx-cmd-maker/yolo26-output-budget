#!/usr/bin/env python
"""
scripts/check_section_order.py —— **章节编号必须与出现顺序一致**

═══ 为什么需要这个检查（P75）═══════════════════════════════════════════
用户发现 Results 的顺序是 `R0, R1, R2, R3, R3b, R5, R5b, R4` —— **R5 在 R4 之前**。
更糟的是：`docs/submission-package.md` 里**我自己把这条写成了"✅ 已重排"**，
即把这个缺陷当成成果记录了下来，还打了勾。

任何编辑或审稿人看到 `R5` 出现在 `R4` 之前，都会认为稿件没有准备好。
**编号是给读者的路标；路标与实际顺序不符，比没有路标更糟。**

本检查做三件事：
  ① `## R<n>` 的出现顺序必须非递减（含 `R3b` 视为 3.5）
  ② 表格编号 `Table N` 的出现顺序必须非递减
  ③ 图编号 `Figure N` 的出现顺序必须非递减

用法：
    python scripts/check_section_order.py
退出码：0 = 合规；1 = 有乱序
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "docs" / "manuscript-results-methods.md",
         ROOT / "docs" / "manuscript-abstract-intro-discussion.md"]


def sec_key(tag: str) -> float:
    """`R3` → 3.0；`R5b` → 5.5（b 视为 .5，保证 R5b 在 R5 之后、R6 之前）。"""
    m = re.fullmatch(r"R(\d+)(b?)", tag)
    if not m:
        return float("nan")
    return int(m.group(1)) + (0.5 if m.group(2) else 0.0)


def check(path: Path) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"缺文件 {path.name}"]
    text = path.read_text(encoding="utf-8")

    # ① 章节
    heads = [l[3:].split(".")[0].strip()
             for l in text.split("\n") if l.startswith("## R")]
    if heads:
        keys = [sec_key(h) for h in heads]
        if any(k != k for k in keys):          # NaN
            problems.append(f"{path.name}: 无法解析的章节号 {heads}")
        elif keys != sorted(keys):
            bad = [(heads[i], heads[i + 1]) for i in range(len(keys) - 1)
                   if keys[i] > keys[i + 1]]
            problems.append(f"{path.name}: 章节乱序 {heads}（逆序对：{bad}）")

    # ② 表
    tabs, seen = [], set()
    for m in re.finditer(r"\*\*Table (\d+)\.", text):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            tabs.append(n)
    if tabs != sorted(tabs):
        problems.append(f"{path.name}: 表号乱序 {tabs}")

    # ③ 图（只在正文引用，不含 Figure legends 节）
    body = text.split("# Figure legends")[0]
    figs, seen_f = [], set()
    for m in re.finditer(r"\bFigure (\d)\b", body):
        n = int(m.group(1))
        if n not in seen_f:
            seen_f.add(n)
            figs.append(n)
    if figs != sorted(figs):
        problems.append(f"{path.name}: 图号乱序 {figs}")

    return problems


# ═══ 跨文件整体章节顺序（P83）════════════════════════════════════════
# 官方 Article 结构：Abstract → Introduction → Results → Discussion → Methods
EXPECTED_DOC_ORDER = ["abstract", "introduction", "result", "discussion", "method"]


def check_doc_order() -> list[str]:
    """检查 build_submission_pdf.py 组装后的**整体**章节顺序。

    ★ 为什么单文件检查不够：Results 与 Discussion 分处两个 md 文件，
      单文件内部顺序全对，但**拼起来**可能把 Discussion 排在 Results 之前。
      这正是 2026-10-17 实际发生的缺陷。
    """
    bp = ROOT / "scripts" / "build_submission_pdf.py"
    if not bp.exists():
        return []
    src = bp.read_text(encoding="utf-8")
    # 从组装代码里按出现顺序抓取 key_part 字符串
    # ★ 一行里可能用 or 连多个候选（如 "abstract" or "introduction"），必须全部抓。
    #   此前只抓 `if "X" in k.lower()` 的第一个 X，导致 introduction 被漏掉。
    ks = []
    for line in src.split("\n"):
        if "in k.lower()" in line:
            ks += re.findall(r'"([a-z ]+)"', line)
    found = [k.strip() for k in ks if k.strip()]
    probs: list[str] = []
    if not found:
        return ["check_doc_order: 未从 build_submission_pdf.py 解析出章节组装顺序"]
    # 归一到族
    def fam(k: str) -> str | None:
        for e in EXPECTED_DOC_ORDER:
            if e in k:
                return e
        return None
    fams = [f for f in (fam(k) for k in found) if f]
    seen = []
    for f in fams:
        if f not in seen:
            seen.append(f)
    # 必需族都得出现，且顺序必须是期望顺序的子序列
    miss = [e for e in EXPECTED_DOC_ORDER if e not in seen]
    if miss:
        probs.append(f"组装顺序缺少章节：{miss}（检出 {seen}）")
    idx = [EXPECTED_DOC_ORDER.index(x) for x in seen if x in EXPECTED_DOC_ORDER]
    if idx != sorted(idx):
        probs.append(f"整体章节乱序：{seen}（期望 {EXPECTED_DOC_ORDER}）")
    return probs


def main() -> int:
    print("=" * 96)
    print("章节/表/图 编号 vs 出现顺序")
    print("=" * 96)
    all_p = []
    print("  — 跨文件整体章节顺序（Abstract→Introduction→Results→Discussion→Methods）")
    doc_p = check_doc_order()
    all_p += doc_p
    print(f"     {'✅ 顺序正确' if not doc_p else '❌ 有乱序'}")
    print()
    for f in FILES:
        p = check(f)
        all_p += p
        # 打印顺序
        if f.exists():
            t = f.read_text(encoding="utf-8")
            h = [l[3:].split(".")[0].strip() for l in t.split("\n") if l.startswith("## R")]
            if h:
                print(f"  {f.name}")
                print(f"     章节顺序: {' -> '.join(h)}")
    print()
    if all_p:
        print("  ❌ 发现乱序：")
        for x in all_p:
            print(f"     {x}")
        print("\n  ★ 修法：优先**移动章节**（不改编号），因为改编号会波及交叉引用。")
        return 1
    print("  ✅ 章节、表、图的编号顺序全部与出现顺序一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
