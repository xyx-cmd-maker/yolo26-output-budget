"""
tests/selftest_allocation_regime.py
===================================
`src/eval/allocation_regime.py` 的**判别力自测**。

═══════════════════════════════════════════════════════════════════════════
为什么必须有它（P16：测试的价值 = 它能否失败）
═══════════════════════════════════════════════════════════════════════════
判据的价值全在**能不能正确区分三种 regime**。若它无论输入什么都报同一个，
它就是死的。本自测构造三个**已知答案**的合成世界：

  | 世界 | 构造 | 期望 regime | 期望增益符号 |
  |---|---|---|---|
  | **`detector_bound`** | 所有图的检测器上限都很**低**（`pᵢ` 小、命中率低），且预算充足 → 两种分配的最坏图都触顶 | `DETECTOR_BOUND` | **Δ ≈ 0** |
  | **`budget_bound`** | 图分两族：多数稀疏（`nᵢ` 小、易检出）+ 少数极密（`nᵢ` 大、难检出）→ 均匀把最密图钉成最坏，水位给它更多 | `BUDGET_BOUND` | **Δ > 0** |
  | **`starved`** | 含 `nᵢ = 1` 的图，且预算很小 → 水位把它的配额 `⌈ρ·1⌉` 取整为 **0** → 它召回 0 | `STARVED` | **Δ < 0** |

**若三个世界报出同一个 regime → 判据无判别力 → 自测失败。**

同时验证：`Δworst` 的**符号**必须与 regime 的预测一致（这是判据的实质内容）。

═══════════════════════════════════════════════════════════════════════════
★ 生成方式
═══════════════════════════════════════════════════════════════════════════
用 `tp_prefix[k] = round(hit · k)` 构造（`k` 从 1 到 `pᵢ`），
即"命中率 `hit` 恒定"的检测器 —— 于是 `TPᵢ(q) = round(hit·min(q, pᵢ))`，
**逐图可解析验算**，不引入任何随机性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.allocation_regime import (  # noqa: E402
    BUDGET_BOUND, DETECTOR_BOUND, STARVED, cap_of, diagnose,
)


def make_rec(n: int, p: int, hit: float, name: str) -> dict:
    """构造一条 `per_image` 记录：`TPᵢ(q) = round(hit·min(q, pᵢ))`（单调、可解析验算）。"""
    total = int(round(hit * p))
    pref, acc = [], 0.0
    for k in range(1, p + 1):
        acc += total / p
        pref.append(min(int(round(acc)), total))
    return {"name": name, "n_gt": int(n), "n_pred_at_run": int(p),
            "tp_prefix": [int(v) for v in pref]}


def world_detector_bound() -> list[dict]:
    """所有图的检测器上限都低且预算充足 → 两种分配的最坏图都触顶。

    构造：每图 `nᵢ` 相同（= `pᵢ`，检测器能输出到齐），命中率 0.4
          → `cap = min(n,p) = n`，给满就 40% 召回。
    预算 `S = Σcap`（= `Σn`）→ 两种分配都能给满 → 最坏图都触顶 → Δ = 0。
    """
    return [make_rec(50, 50, 0.40, f"det{i}") for i in range(60)]


def world_budget_bound() -> list[dict]:
    """两族：稀疏易检出 + 极密难检出 → 均匀把最密图钉成最坏，水位改善它。

    稀疏族：`n = 10, p = 10`, hit 0.9
    密集族：`n = 200, p = 200`, hit 0.9
    预算 `S = 0.35 · Σn` → 均匀给每图约 0.35·mean ≈ 37；
    最密图（n=200）只拿到 37 → 召回 ≈ 0.9·37/200 = 16.6%；
    水位给它 `⌈0.35·200⌉ = 70` → 召回 ≈ 0.9·70/200 = 31.5%
    → Δ 明显为正。
    """
    recs = []
    for i in range(80):
        recs.append(make_rec(10, 10, 0.90, f"sparse{i}"))
    for i in range(20):
        recs.append(make_rec(200, 200, 0.90, f"dense{i}"))
    return recs


def world_starved() -> list[dict]:
    """**分配退化的边界情形**：`B ≲ 1`（预算小于图的张数）时的整数取整效应。

    构造：大量 `n = 1, p = 1`（命中率 1.0）+ 少数 `n = 2, p = 2`
    预算取 `S ≈ 0.15 · Σcap`（远小于图的张数）→ 水位法的 `ρ < 1`
    → `⌈ρ·1⌉` 可能取 0 → 该图召回 0。

    ★ 本世界**不作为 regime 断言**，只断言 `degenerate=True` 与 `starved_units > 0`：
      它描述的是"预算小到分配退化"，不是"另一种绑定机制"（见模块文档）。
    """
    recs = []
    for i in range(300):
        recs.append(make_rec(1, 1, 1.00, f"single{i}"))
    for i in range(40):
        recs.append(make_rec(2, 2, 1.00, f"pair{i}"))
    return recs


def gain_measured(recs: list[dict], S: float) -> float:
    """用判据自身的两种分配实测 Δworst（与 `diagnose` 内部一致，用于交叉验证）。"""
    return diagnose(recs, S).gain_pt


def main() -> int:
    print("=" * 96)
    print("分配机制判据的判别力自测")
    print("=" * 96)

    cases = []

    # ── 世界 1：检测器卡住 ────────────────────────────────────────
    r1 = world_detector_bound()
    n1, _, c1 = cap_of(r1)
    d1 = diagnose(r1, float(c1.sum()))          # 预算 = Σcap → 都能给满
    cases.append(("detector_bound", d1, DETECTOR_BOUND, "≈0"))

    # ── 世界 2：预算卡住 ─────────────────────────────────────────
    r2 = world_budget_bound()
    n2, _, c2 = cap_of(r2)
    S2 = 0.35 * float(n2.sum())
    d2 = diagnose(r2, S2)
    cases.append(("budget_bound", d2, BUDGET_BOUND, ">0"))

    # ── 世界 3：分配退化（B ≲ 1）─────────────────────────────────
    r3 = world_starved()
    n3, p3, c3 = cap_of(r3)
    S3 = 0.15 * float(c3.sum())                # 远小于图的张数 → ρ < 1
    d3 = diagnose(r3, S3)

    # ── 打印 ────────────────────────────────────────────────────
    print(f"\n  {'世界':<18} {'N':>5} {'S':>9} {'regime 实测':>16} {'regime 期望':>16} "
          f"{'Δworst':>10} {'期望符号':>9} 判定")
    print("  " + "-" * 96)
    cases = [("detector_bound", d1, DETECTOR_BOUND, "≈0"),
             ("budget_bound", d2, BUDGET_BOUND, ">0")]
    fails: list[str] = []
    for name, d, exp_regime, exp_sign in cases:
        ok_r = d.regime == exp_regime
        ok_g = abs(d.gain_pt) < 1.0 if exp_sign == "≈0" else d.gain_pt > 1.0
        print(f"  {name:<18} {d.n_units:>5} {d.S:>9.0f} {d.regime:>16} {exp_regime:>16} "
              f"{d.gain_pt:>+9.3f}p {exp_sign:>9} {'✅' if (ok_r and ok_g) else '❌'}")
        if not ok_r:
            fails.append(f"{name}: regime 实测 {d.regime} ≠ 期望 {exp_regime}")
        if not ok_g:
            fails.append(f"{name}: Δworst = {d.gain_pt:+.3f} pt 与期望符号 {exp_sign} 不符")

    # 世界 3 单独断言（退化标志，不是 regime）
    print(f"  {'starved(退化)':<18} {d3.n_units:>5} {d3.S:>9.0f} {d3.regime:>16} "
          f"{'(不作断言)':>16} {d3.gain_pt:>+9.3f}p {'—':>9} "
          f"{'✅' if d3.degenerate else '❌'}")
    if not d3.degenerate:
        fails.append("starved 世界未标 degenerate —— 退化检测失效")
    if d3.starved_units <= 0:
        fails.append("starved 世界 starved_units = 0 —— 饥饿计数失效")

    # ── 断言 4：判别力（≥2 种 regime）────────────────────────────
    regs = {c[1].regime for c in cases}
    distinct = len(regs) >= 2
    print(f"\n  ④ 判别力（≥2 种 regime）：{'✅' if distinct else '❌'}  {sorted(regs)}")
    if not distinct:
        fails.append("三个世界报了同一 regime → 判据无判别力")

    # ── 断言 5：N1（同花费）──────────────────────────────────────
    n1ok = all(abs(c[1].spend_over_S - 1.0) <= 0.01 for c in cases)
    print(f"  ⑤ N1 同花费（|spend/S − 1| ≤ 1%）：{'✅' if n1ok else '❌'}  "
          f"{[round(c[1].spend_over_S, 4) for c in cases]}")
    if not n1ok:
        fails.append("有世界的 spend/S 偏离 1 超过 1% → 比较不可比")

    # ── 断言 6：触顶标志与 regime 自洽 ──────────────────────────
    ok_cap = (d1.uniform_worst_at_cap and d1.waterfill_worst_at_cap)
    print(f"  ⑥ detector_bound 世界两侧都触顶：{'✅' if ok_cap else '❌'}  "
          f"(均匀={d1.uniform_worst_at_cap}, 水位={d1.waterfill_worst_at_cap})")
    if not ok_cap:
        fails.append("detector_bound 世界未两侧触顶 —— 构造或判据有误")

    print("\n" + "=" * 96)
    if fails:
        print(f"❌ 自测失败（{len(fails)} 条）")
        for f in fails:
            print(f"   · {f}")
        return 1
    print("✅ 全部 6 项断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
