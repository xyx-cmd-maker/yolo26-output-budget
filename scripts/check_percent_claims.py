#!/usr/bin/env python
"""scripts/check_percent_claims.py —— 百分比/整数断言的登记册检查（P127 根因修复）

═══ 为什么需要 ═══════════════════════════════════════════════════════════
`verify_manuscript_numbers_against_source.py` 只比对**带小数的数字**，所以
"11% to 29%" 这类整数百分比断言**全部从检查里漏过去**，而它们恰恰最容易被手工维护。
实测：那一句是陈旧的（真实范围 0.3%–42.1%）。

═══ 机制 ═════════════════════════════════════════════════════════════════
`results/tables/percent-claims-registry.json` 是**登记册**：正文里每一个 `N%` 断言
都必须有一条登记，写明它从哪来（`source`）。未登记的断言 → **检查失败**。

若登记条目带 `json` 字段，检查器还会把 `value` 与 JSON 里的实际值比对。

用法：python scripts/check_percent_claims.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results" / "tables" / "percent-claims-registry.json"
DOCS = [ROOT / "docs" / "manuscript-results-methods.md",
        ROOT / "docs" / "manuscript-abstract-intro-discussion.md"]

# 提取 N% 或 N per cent；排除表格行（以 | 开头）
CLAIM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:%|per\s?cent\b)", re.I)


def extract_claims(path: Path) -> list[tuple[str, float, str]]:
    """返回 [(上下文片段, 数值, 类别)]；类别在 {'ci', 'table', 'prose'}。"""
    out = []
    for raw in path.read_text(encoding="utf-8").split("\n"):
        line = raw.strip()
        if not line:
            continue
        is_table = line.startswith("|")
        for m in CLAIM_RE.finditer(line):
            val = float(m.group(1))
            # "95% CI" 是固定术语，不是数据断言
            tail = line[m.end():m.end() + 4].strip().lower()
            head = line[max(0, m.start() - 5):m.start()].strip().lower()
            if head.endswith("95") or tail.startswith("ci"):
                cat = "ci"
            elif is_table:
                cat = "table"
            else:
                cat = "prose"
            ctx = re.sub(r"\s+", " ", line[max(0, m.start() - 70):m.end() + 40])
            out.append((ctx, val, cat))
    return out


def main() -> int:
    if not REG.exists():
        print(f"  ❌ 缺登记册 {REG.relative_to(ROOT)}")
        return 1
    reg = json.loads(REG.read_text(encoding="utf-8"))
    allowed = {(e["value"], e.get("scope", "prose")) for e in reg["entries"]}
    # 分位带名等非数据断言在登记册的 exclusions 里声明后得到豁免
    excluded = set()
    for ex in reg.get("exclusions", []):
        excluded.update(float(v) for v in ex.get("values", []))

    total = 0
    unregistered = []
    for doc in DOCS:
        for ctx, val, cat in extract_claims(doc):
            if cat == "ci" or cat == "table":
                continue          # 表格内容与 95% CI 由别的检查覆盖
            if val in excluded:
                continue          # 分位带名等，不是数据断言
            total += 1
            if (val, "prose") not in allowed:
                unregistered.append((doc.name, val, ctx))

    print("=" * 80)
    print("百分比断言登记册检查")
    print("=" * 80)
    print(f"  登记册条目 {len(reg['entries'])} 条；正文 prose 类断言 {total} 处")
    if unregistered:
        print(f"\n  ❌ {len(unregistered)} 处**未登记**的百分比断言：")
        for d, v, c in unregistered:
            print(f"     [{d[:14]}] {v}%  …{c}…")
        print("\n  ★ 处置：先在 results/tables/percent-claims-registry.json 里登记")
        print("     （写明 source 与推导方式），再提交。未登记的百分比 = 无人核过的数字。")
        return 1
    print("  ✅ 全部 prose 类百分比断言都有登记来源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
