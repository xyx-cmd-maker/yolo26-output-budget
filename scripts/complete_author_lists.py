"""
scripts/complete_author_lists.py —— 从权威源取回**完整作者列表**，供 BibTeX 使用

★ 为什么必须做：台账里部分条目只有 `Lin et al` 这类简写（因为原先只用于渲染
  "Lin et al. (2014)"），**不足以生成合法 BibTeX**。
  权威源：DOI → Crossref（DataCite 分流）；arXiv id → arXiv API。

输出：写回台账的 `authors`（完整列表）与 `authors_short`（渲染用简写），
      使"渲染"与"BibTeX"各取所需，不再互相迁就。
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "tables" / "reference-ledger.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; author-list-builder)"}


def get(url: str, tries: int = 3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == tries - 1:
                return f"__ERR__ {type(e).__name__}: {e}"
            time.sleep(1.5)
    return "__ERR__"


def from_crossref(doi: str) -> list[str]:
    t = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if t.startswith("__ERR__"):
        return []
    try:
        m = json.loads(t)["message"]
    except Exception:
        return []
    out = []
    for a in m.get("author", []) or []:
        fam, giv = (a.get("family") or "").strip(), (a.get("given") or "").strip()
        if fam:
            out.append(f"{fam}, {giv}" if giv else fam)
    return out


def from_arxiv(aid: str) -> list[str]:
    """arXiv 返回 `First Middle Last`，需统一成 `Last, First Middle`。

    ★ skill §4 硬规则："作者名格式统一（`Last, First` 或 `First Last`，**全文一致**，
      不要混用）"。Crossref 给的是 `Last, First`，故 arXiv 侧必须转成同一形式。
    """
    aid = aid.replace("arXiv:", "").strip()
    t = get(f"http://export.arxiv.org/api/query?id_list={aid}&max_results=1")
    if t.startswith("__ERR__"):
        return []
    out = []
    for name in re.findall(r"<author>\s*<name>([^<]+)</name>", t):
        name = re.sub(r"\s+", " ", name).strip()
        parts = name.split(" ")
        if len(parts) >= 2:
            out.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        else:
            out.append(name)
    return out


def short_form(authors: list[str]) -> list[str]:
    """渲染用简写：1 位→姓；2 位→姓, 姓；≥3 位→首作者姓 et al."""
    if not authors:
        return []
    surs = [a.split(",")[0].strip() for a in authors]
    if len(surs) == 1:
        return surs
    if len(surs) == 2:
        return surs
    return [surs[0] + " et al"]


def main() -> int:
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    n_full = n_fail = 0
    for e in led:
        k = e.get("key", "")
        if k.startswith("["):
            continue
        # 判断"已是完整列表"：有作者、无 et al 简写、且**每位都含逗号**（`Last, First`）。
        #   ★ 这条判据我改了两次才对：
        #     初版只查 "et al" → 哨兵 `['重取']` 被当成完整 → 补全 0 条；
        #     二版加了"无逗号且单词"的例外 → 中文哨兵『重取』**无空格也无逗号**，
        #       仍被误判为完整。**教训：哨兵值必须显式识别，不能靠"像不像真名"来判。**
        SENTINEL = "重取"
        au = e.get("authors") or []
        complete = bool(au) and all(
            (SENTINEL not in str(a)) and ("et al" not in str(a)) and ("," in str(a))
            for a in au
        )
        if complete:
            e["authors_short"] = short_form(au)
            continue
        doi, ident = e.get("doi") or "", str(e.get("id") or "")
        full: list[str] = []
        src = ""
        if doi:
            full, src = from_crossref(doi), f"Crossref {doi}"
        elif ident.startswith("arXiv:"):
            full, src = from_arxiv(ident), f"arXiv {ident}"
        if full:
            e["authors"] = full
            e["authors_short"] = short_form(full)
            e["_authors_source"] = src
            e["_verified_2026_10_16"] = ((e.get("_verified_2026_10_16") or "")
                                         + f" | 作者列表取自 {src}（{len(full)} 位）").strip(" |")
            n_full += 1
            print(f"  ✅ {k}: {len(full)} 位作者  ← {src}")
            print(f"        {', '.join(full[:4])}{' …' if len(full) > 4 else ''}")
            print(f"        渲染简写: {e['authors_short']}")
        else:
            n_fail += 1
            print(f"  ⚠️ {k}: 取不到完整作者（doi={doi!r} id={ident!r}）")

    LEDGER.write_text(json.dumps(led, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  补全 {n_full} 条；失败 {n_fail} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
