"""spend 不变量硬断言（第 10 版 Major 6 要求）。

审稿人要求：不能只断言两个**理论**支出表达式相等，必须对**实际实现**设硬断言：
  1. Σmin(q,n,p) 与 Σmin(q,p) 逐行相等（主文的 spend 恒等式）
  2. 可花费条件下，实际发出框数 E(q)=Σmin(q,p) 与目标预算 S 的误差很小
  3. q>n 不得出现（cap 由构造施加）
  4. 饱和条件下 emitted 必须小于 S（名义预算没花完）

用法：python scripts/check_spend_invariants.py
退出码：0 = 全部成立；1 = 有违反
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "tables" / "spend-audit.json"

TOL_SPENDABLE_PCT = 0.05
SATURATED = ("B=100", "B=300")


def main() -> int:
    if not SRC.exists():
        print(f"  缺 {SRC.relative_to(ROOT)}")
        return 1
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = d["rows"]
    print("=" * 78)
    print("spend 不变量硬断言")
    print("=" * 78)

    fails: list[str] = []

    bad = [r for r in rows if abs(r["emitted"] - r["capped"]) > 0.5]
    print(f"  1. emitted == capped: {len(rows) - len(bad)}/{len(rows)} 行成立")
    for r in bad:
        fails.append(f"emitted!=capped seed{r['seed']}/{r['head']}/{r['condition']}/{r['allocation']}")

    spendable = [r for r in rows if r["condition"] not in SATURATED]
    over = [r for r in spendable if r["emitted_err_pct"] > TOL_SPENDABLE_PCT]
    mx = max((r["emitted_err_pct"] for r in spendable), default=0.0)
    print(f"  2. 可花费条件支出误差最大 {mx:.4f}% (阈值 {TOL_SPENDABLE_PCT}%)")
    for r in over:
        fails.append(f"spend error {r['emitted_err_pct']}% seed{r['seed']}/{r['head']}/{r['condition']}")

    qn = [r for r in rows if r["n_q_gt_n"] != 0]
    print(f"  3. q>n 的行数: {len(qn)} (必须为 0)")
    for r in qn:
        fails.append(f"q>n seed{r['seed']}/{r['head']}/{r['condition']}")

    sat = [r for r in rows if r["condition"] in SATURATED]
    bad_sat = [r for r in sat if r["emitted"] >= r["S_target"]]
    print(f"  4. 饱和条件 {len(sat)} 行, 其中 emitted>=S 的 {len(bad_sat)} 行 (必须为 0)")
    for r in bad_sat:
        fails.append(f"saturated emitted>=S {r['condition']}/{r['allocation']}")

    print()
    if fails:
        print(f"  [FAIL] {len(fails)} 项违反:")
        for f in fails[:10]:
            print(f"     {f}")
        return 1
    print("  [OK] 四项不变量全部成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())
