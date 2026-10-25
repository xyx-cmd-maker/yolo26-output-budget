"""
scripts/build_reference_list.py —— 从台账生成**参考文献表**，并审计其可生成性

═══════════════════════════════════════════════════════════════════════════
为什么需要
═══════════════════════════════════════════════════════════════════════════
我一直在维护 `docs/reference-ledger.md` 与 `reference-ledger.json`，
但**从未试着把它们渲染成投稿需要的参考文献表**。
独立审查提醒过：`decisionvalue2026` / `densescout2026` **缺作者名**时"无法生成参考文献表"
—— 我已补齐，但**从未验证现在真的能生成**。

本脚本做两件事：
  ① **审计**：逐条检查台账能否渲染出完整条目（作者/年份/标题/venue/标识符），
     缺任何一项就报错并列出来
  ② **渲染**：按 Sci Rep 的编号制生成 `results/reference-list.md`
     （正文用 `[key]`，此处输出 key → 编号 的映射表，供排版时替换）

用法：
    python scripts/build_reference_list.py
    python scripts/build_reference_list.py --used-only    # 只列正文用到的
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "tables" / "reference-ledger.json"
MD = ROOT / "docs" / "reference-ledger.md"
OUT = ROOT / "results" / "reference-list.md"
MS = (ROOT / "docs" / "manuscript-abstract-intro-discussion.md",
      ROOT / "docs" / "manuscript-results-methods.md")

# 渲染一条所需的最小字段
REQUIRED = ("authors", "year", "title", "venue")
OPTIONAL_ID = ("doi", "id")

# ★★ 第 54 轮 P60：台账把文献元数据放在**一个 `cite` 字符串**里，没有结构化字段
#    （19/20 条如此）。于是"生成参考文献表"只能靠人抄字符串 —— 而"手抄必错"
#    是本项目最老的教训。这里**解析 `cite` 并回写台账**，使此后可机械生成与审计。
SURNAME_PREFIX = ""


def _split_authors(s: str) -> list[str]:
    """按 `,` / `&` 切作者，**但跳过括号内部**（如 `(2nd ed.)`、`(2018)`）。"""
    out, buf, depth = [], "", 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ",&" and depth == 0:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return [re.sub(r"^(and|&)\s+", "", x).strip() for x in out if x.strip()]


def parse_cite(cite: str) -> dict:
    """把 `Authors (Year). Title. Venue.` 解析成结构化字段。

    解析是**尽力而为**；失败的字段留空，由审计报出。
    """
    got: dict = {}
    # 年份：优先取第一个 (19xx|20xx)
    m = re.search(r"\((1[89]\d{2}|20\d{2})\)", cite)
    if m:
        got["year"] = int(m.group(1))
    # 作者 = 第一个括号之前的部分
    head = cite.split("(", 1)[0].strip()
    # ★ 只去掉**一个**尾随句点，不能全去：`et al.` 的句点必须保留
    #   （初版用 `rstrip('.')` 把 `Jocher et al.` 变成了 `Jocher et al`）。
    if head.endswith("."):
        head = head[:-1]
    if head:
        got["authors"] = _split_authors(head)
    # 年份括号之后的部分：Title. Venue.
    if m:
        rest = cite[m.end():].lstrip(". ").strip()
        # 用 ". " 切；若第一段以 URL/DOI 开头则说明无标题
        parts = [p.strip() for p in re.split(r"\.\s+", rest) if p.strip()]
        if parts:
            got["title"] = parts[0].rstrip(".")
        if len(parts) > 1:
            got["venue"] = ". ".join(parts[1:]).rstrip(".")
    # 标识符
    md = re.search(r"(10\.\d{4,5}/[^\s]+)", cite)
    if md:
        got["doi"] = md.group(1)
    ma = re.search(r"(arXiv:\s*\d{4}\.\d{4,5})", cite)
    if ma:
        got["id"] = ma.group(1).replace(" ", "")
    mi = re.search(r"(ISBN\s*[\d\-X]+)", cite)
    if mi and "doi" not in got:
        got["id"] = mi.group(1)
    mj = re.search(r"(JSTOR:\s*\d+)", cite)
    if mj and "doi" not in got:
        got["id"] = mj.group(1)
    mu = re.search(r"(https?://\S+)", cite)
    if mu and "doi" not in got and "id" not in got:
        got["id"] = mu.group(1).rstrip(".")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description="生成并审计参考文献表")
    ap.add_argument("--used-only", action="store_true", help="只输出正文实际引用的条目")
    a = ap.parse_args()

    entries = json.loads(LEDGER.read_text(encoding="utf-8"))
    print("=" * 100)
    print("① 台账可渲染性审计 + 从 `cite` 解析结构化字段")
    print("=" * 100)
    print(f"  台账共 {len(entries)} 条")

    # ★ 对缺结构化字段的条目，从 `cite` 解析并**回写台账**（使此后可机械生成）
    patched = 0
    for e in entries:
        if any(not e.get(f) for f in REQUIRED) and e.get("cite"):
            got = parse_cite(e["cite"])
            for k, v in got.items():
                e.setdefault(k, v)
            if got:
                patched += 1
    if patched:
        LEDGER.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ✅ 从 `cite` 解析并回写 {patched} 条（台账已更新，此后可直接机械生成）")

    # 正文实际用到的 key（★ 修：原正则漏了含数字/下划线的键，如 jocher2026yolo26）
    used: set[str] = set()
    for p in MS:
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\[([a-zA-Z][a-zA-Z0-9_]{2,})\]", txt):
            v = m.group(1)
            if re.search(r"\d{4}", v) or v.startswith("ultralytics"):
                used.add(v)
    print(f"  正文引用（方括号键）= {len(used)} 个")

    broken, renderable = [], []
    for e in entries:
        key = e.get("key", "?")
        if key.startswith("["):
            continue                       # 形如 `[待补]UAVDT` 的占位条目，跳过
        missing = [f for f in REQUIRED if not e.get(f)]
        if not any(e.get(i) for i in OPTIONAL_ID):
            missing.append("doi/id")
        if missing:
            broken.append((key, missing, e))
        else:
            renderable.append(e)

    print(f"\n  可渲染 = {len(renderable)} 条；**缺字段 = {len(broken)} 条**")
    for key, miss, e in broken:
        print(f"    ❌ {key}: 缺 {miss}")
        print(f"       现有 cite = {e.get('cite', '(无)')[:96]}")

    print("\n" + "=" * 100)
    print("② 渲染参考文献表（Sci Rep 用编号制）")
    print("=" * 100)

    order = [e for e in renderable
             if (not a.used_only) or (e["key"] in used)]
    # 按第一作者姓氏 + 年份排序（编号制下顺序由正文首次出现决定；
    #   此处给字母序，并在表里标明"编号需按正文首次出现重排"）
    def sortkey(e):
        au = e["authors"][0] if isinstance(e.get("authors"), list) else str(e.get("cite", ""))
        return (str(au).lower(), str(e.get("year", "")))

    order.sort(key=sortkey)

    lines = ["# Reference list (generated from `docs/reference-ledger.md`)", "",
             f"*{len(order)} entries. Sci Rep uses a numbered style; the numbers below are alphabetical "
             "placeholders and must be renumbered by first appearance in the text.*", ""]
    fails = 0
    for i, e in enumerate(order, 1):
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
        # ★ 缩写形式要补句点：`Jocher et al` → `Jocher et al.`（原始 cite 里本就有句点，
        #   解析时为了让渲染可控而去掉了，这里按语境加回）。
        astr = re.sub(r"\bet al$", "et al.", astr)
        ident = e.get("doi") or e.get("id") or ""
        # ★ 渲染用**干净的 venue**；`venue_status` 是**内部审计字段**
        #   （含 "PEER-REVIEWED: ..." 之类的判断），不应进投稿件。
        venue = e.get("venue") or ""
        if not venue:
            vs = e.get("venue_status") or ""
            venue = vs.split(";")[0].split("(")[0].strip() or vs
        title = e.get("title") or ""
        line = f"{i}. {astr} ({e.get('year')}). {title}. *{venue}*. {ident}".strip()
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
        print(f"  {line[:150]}")
        if not title or not venue:
            fails += 1

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ 已写出 {OUT.relative_to(ROOT)}（{len(order)} 条）")

    print("\n" + "=" * 100)
    print("③ 结论")
    print("=" * 100)
    if broken:
        print(f"  ⚠️ **{len(broken)} 条台账条目缺字段**，无法渲染出完整条目：")
        for key, miss, _ in broken:
            print(f"     {key}: 缺 {miss}")
        print("     → 补齐 `reference-ledger.json` 后重跑本脚本。")
        return 1
    print("  ✅ 台账全部条目可渲染为完整参考文献条目")
    print("  ⚠️ 但**编号仍需按正文首次出现重排** —— 本脚本给的是字母序占位。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
