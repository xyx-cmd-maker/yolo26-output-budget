"""
scripts/number_references.py —— 按**正文首次出现顺序**给参考文献编号（Sci Rep 用编号制）

═══════════════════════════════════════════════════════════════════════════
为什么需要
═══════════════════════════════════════════════════════════════════════════
`build_reference_list.py` 输出的是**字母序占位**，并在表头注明"编号需按正文首次出现重排"。
投编号制期刊时，编号必须由**正文首次引用顺序**决定 —— 否则排版后会出现
"文中先出现 [7] 再出现 [1]"。

═══════════════════════════════════════════════════════════════════════════
扫描顺序（按投稿件的实际阅读顺序）
═══════════════════════════════════════════════════════════════════════════
  ① 标题页（不引用）
  ② Abstract（Sci Rep 要求 Abstract **不引用文献** → 若出现键则报错）
  ③ Introduction
  ④ Results（R0 → R1 → … → R5b → R4）
  ⑤ Discussion
  ⑥ Methods
  ⑦ Figure legends
  ⑧ Data and code availability

用法：
    python scripts/number_references.py
    python scripts/number_references.py --check     # 只检查，不写文件
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "tables" / "reference-ledger.json"
MD = ROOT / "results" / "reference-list.md"
DOC_A = ROOT / "docs" / "manuscript-abstract-intro-discussion.md"
DOC_B = ROOT / "docs" / "manuscript-results-methods.md"

# ★★ 第 54 轮 P64：引用键正则有**两个**坑，我都踩过：
#   ① 初版 `\[([a-zA-Z][a-zA-Z0-9_]*?)\]` 要求括号内**只有**一个键，
#      于是 `[wang2026recceiling; buckley2004reliability]` 与 `[bertsekas1992data; hardy1934inequalities]`
#      **整对漏掉** → 重排后只有 13 条，实际 16 条。
#   ② 只按 `# 节` 切分会漏掉**第一个子标题之前**的正文（Results 的引言段）。
#   现改为：先取**全文**、再按节序号定位，并用"括号内可含多个键"的正则。
KEY_RE = re.compile(r"\[([^\[\]]{2,80})\]")


def keys_in(text: str) -> list[str]:
    """从一段文本里抽出所有引用键（支持 `[a; b]` 与 `[a; b; c]`）。"""
    out: list[str] = []
    for m in KEY_RE.finditer(text):
        for part in re.split(r"[;,]\s*", m.group(1)):
            p = part.strip()
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", p) and (
                    re.search(r"\d{4}", p) or p.startswith("ultralytics")):
                out.append(p)
    return out


def sections(path: Path) -> list[tuple[str, str]]:
    """按 `# X` 切节，返回 [(节名, 正文)]。

    ★ 关键：**第一个 `# ` 之前的正文归入 '(前言)'**，且**第一个子标题之前的正文**
      属于该节（不能丢）。初版按 `# ` 切分后只在 ORDER 里查节名，
      于是 Results 的引言段（R0 之前）被整段跳过。
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    out, cur_name, cur = [], "(前言)", []
    for ln in lines:
        if ln.startswith("# ") and not ln.startswith("## "):
            out.append((cur_name, "\n".join(cur)))
            cur_name, cur = ln[2:].strip(), []
        else:
            cur.append(ln)
    out.append((cur_name, "\n".join(cur)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="按首次出现顺序编号参考文献")
    ap.add_argument("--check", action="store_true", help="只检查，不写文件")
    a = ap.parse_args()

    # ★ 阅读顺序：DOC_A 含 Abstract/Introduction/Discussion，DOC_B 含 Results/Methods/…
    #   但排版顺序是 Abstract → Introduction → Results → Discussion → Methods → …
    #   故必须显式指定节序，不能按文件顺序。
    ORDER = ["Abstract", "Introduction", "Results", "Discussion", "Methods",
             "Author contributions", "Figure legends", "Data and code availability"]

    sec_map: dict[str, list[tuple[str, str]]] = {}
    for doc in (DOC_A, DOC_B):
        for name, body in sections(doc):
            sec_map.setdefault(name, []).append((name, body))

    print("=" * 100)
    print("① 按阅读顺序扫描正文，记录每个键的首次出现")
    print("=" * 100)
    first: dict[str, int] = {}
    seq: list[str] = []
    for sec in ORDER:
        if sec not in sec_map:
            continue
        for name, body in sec_map[sec]:
            # 去掉引用块与内部注记，避免把说明文字里的键当成正文引用
            clean = "\n".join(l for l in body.split("\n")
                              if not l.strip().startswith((">", "*(")))
            keys = keys_in(clean)
            # ★★ P64 的关键 bug：**不能 `sorted(set(keys))`** ——
            #   `sorted()` 把"首次出现顺序"变成了**字母序**，
            #   于是编号整体错位（`bertsekas1992data` 被编成 #1，
            #   而真正最先引用的是第 40 行的 `lin2014coco`）。
            #   必须**按出现顺序去重**。
            uniq = list(dict.fromkeys(keys))
            if not keys:
                continue
            print(f"\n  【{sec}】出现 {len(keys)} 处 / {len(uniq)} 个不同键")
            for k in uniq:
                if k not in first:
                    first[k] = len(seq) + 1
                    seq.append(k)
                    print(f"      #{first[k]:>2}  {k}   ← 首次出现")
                else:
                    print(f"      (#{first[k]:>2}) {k}")

    print("\n" + "=" * 100)
    print("② 检查")
    print("=" * 100)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    have = {e["key"] for e in ledger if not e["key"].startswith("[")}

    # Abstract 不应含引用（Sci Rep：Abstract 非结构化且通常无引用）
    abs_keys: list[str] = []
    for _, body in sec_map.get("Abstract", []):
        abs_keys += keys_in(body)
    print(f"  Abstract 中的引用键 = {abs_keys if abs_keys else '无 ✅'}")

    missing = [k for k in seq if k not in have]
    unused = sorted(have - set(seq))
    print(f"  正文引用 {len(seq)} 个键；台账有 {len(have)} 个")
    if missing:
        print(f"  ❌ 正文引用但台账没有：{missing}")
    else:
        print("  ✅ 正文引用的每个键都在台账里")
    if unused:
        print(f"  ⚠️ 台账有但正文未引用（保留备用，不进参考文献表）：{unused}")

    print("\n" + "=" * 100)
    print("③ 按首次出现顺序输出参考文献表")
    print("=" * 100)
    by_key = {e["key"]: e for e in ledger}

    def render(e: dict, n: int) -> str:
        au = e.get("authors")
        if isinstance(au, list):
            if len(au) == 1:
                astr = au[0]
            elif len(au) == 2:
                astr = f"{au[0]} & {au[1]}"
            else:
                astr = ", ".join(au[:-1]) + f", & {au[-1]}"
        else:
            astr = str(au)
        astr = re.sub(r"\bet al$", "et al.", astr)
        ident = e.get("doi") or e.get("id") or ""
        venue = e.get("venue") or ""
        if not venue:
            venue = (e.get("venue_status") or "").split(";")[0].split("(")[0].strip()
        line = f"{n}. {astr} ({e.get('year')}). {e.get('title')}. *{venue}*. {ident}"
        return re.sub(r"\s+", " ", line).strip()

    lines = ["# Reference list (numbered by first appearance in the text)", ""]
    for i, k in enumerate(seq, 1):
        line = render(by_key[k], i)
        lines.append(line)
        print(f"  {line[:150]}")

    if a.check:
        print("\n  （--check：未写文件）")
        return 0
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ 已写出 {MD.relative_to(ROOT)}（{len(seq)} 条，按首次出现编号）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
