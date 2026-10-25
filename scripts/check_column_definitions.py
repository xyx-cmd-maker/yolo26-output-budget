#!/usr/bin/env python
"""scripts/check_column_definitions.py —— 表头口径检查（P130）

═══ 为什么需要 ═══════════════════════════════════════════════════════════
Table 1 的 `Images` 列实际是"有标注的图数"（COCO val2017 总 5000，4952 有标注），
但列名只写 `Images`，审稿人一定会问。我靠人工发现并改成 `Images (with gt)`。
这类"列名口径未说明"的缺陷同样可以自动化。

═══ 机制 ═════════════════════════════════════════════════════════════════
对每个表的表头，抽出两类 token：
  · 括号限定词，例如 `(with gt)`、`(pt)`、`(uniform / prop.)`
  · 反引号符号，例如 `n_i`、`B`、`util`
每个 token 必须能在**该表自己的图注**或**正文 prose**里找到定义性出现。
找不到 ⇒ 该列口径未说明 ⇒ 检查失败。

用法：python scripts/check_column_definitions.py
"""
from __future__ import annotations

import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "manuscript-results-methods.md",
        ROOT / "docs" / "manuscript-abstract-intro-discussion.md"]

# 表头 token 的"同义扩展"：列名缩写在正文里常以全称出现
ALIASES = {
    "pt": ["points"],
    "with gt": ["at least one annotated object", "images that carry at least one annotated"],
    ">100": ["> 100", "exceed a budget of 100"],
    "uniform / prop.": ["uniform", "proportional"],
    "spend / s": ["spend", "S`"],
}


def prose_only(text: str) -> str:
    """去掉表格行与 markdown 结构，只留叙述文字。"""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|") or s.startswith("---"):
            continue
        out.append(line)
    return "\n".join(out)


def tables(doc: str):
    """返回 [(n, caption, header)]。"""
    res = []
    for n in range(1, 6):
        m = re.search(r"\*\*Table " + str(n) + r"\.", doc)
        if not m:
            continue
        seg = doc[m.start():]
        nxt = seg.find("**Table " + str(n + 1) + ".")
        if nxt < 0:
            nxt = seg.find("\n## ")
        block = seg[:nxt if nxt > 0 else 1500]
        lines = block.split("\n")
        hdr, hdr_i = None, -1
        for i, l in enumerate(lines):
            if l.strip().startswith("|"):
                hdr, hdr_i = l.strip(), i
                break
        cap = "\n".join(lines[:hdr_i]) if hdr_i > 0 else ""
        res.append((n, re.sub(r"\s+", " ", cap), hdr))
    return res


def main() -> int:
    md = io.open(DOCS[0], encoding="utf-8").read()
    prose = prose_only(io.open(DOCS[0], encoding="utf-8").read())
    prose += "\n" + prose_only(io.open(DOCS[1], encoding="utf-8").read())
    prose_l = prose.lower()

    print("=" * 82)
    print("表头口径检查：每个 token 必须在表注或正文里有定义性出现")
    print("=" * 82)

    bad = 0
    total = 0
    for n, cap, hdr in tables(md):
        if not hdr:
            continue
        quals = re.findall(r"\(([^)]*)\)", hdr)
        syms = [s for s in re.findall(r"`([^`]+)`", hdr) if not s.startswith("frac")]
        toks = [(q, "qualifier") for q in quals] + [(s, "symbol") for s in syms]
        seen = set()
        for tok, kind in toks:
            key = tok.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            total += 1
            cands = [key] + [a.lower() for a in ALIASES.get(key, [])]
            in_cap = any(c in cap.lower() for c in cands)
            in_prose = any(c in prose_l for c in cands)
            ok = in_cap or in_prose
            where = "图注" if in_cap else ("正文" if in_prose else "——")
            if not ok:
                bad += 1
            print(f"  {'✅' if ok else '❌'} Table {n} {kind:<9} {tok!r:<22} 定义见于: {where}")

    print()
    if bad:
        print(f"  ❌ {bad}/{total} 个表头 token 在表注与正文里都找不到定义")
        print("     ⇒ 列名口径未说明。改法：在列名里写出限定词，或在表注/Methods 里给出定义。")
        return 1
    print(f"  ✅ 全部 {total} 个表头 token 都在表注或正文里有定义性出现")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
