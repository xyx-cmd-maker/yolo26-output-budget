"""
★★ 强化边界条件：扫 `B` 得到 `Δworst_abs(B)` 的**连续曲线**，精确给出它何时归零

═══════════════════════════════════════════════════════════════════════════
为什么这比"再加一个数据集"更强
═══════════════════════════════════════════════════════════════════════════
原计划是补第四个数据集点，但：
  · UAVDT / WiderPerson 的标签通道本机都不通（已探）
  · 更重要的：**四个点仍只是四个点**，而审稿人会问"拐点在哪"

**本方案的思路**：数据集不变，**扫预算 `B`**。
  · 每个 `B` 给出一个 (`util(B)`, `frac(nᵢ>B)`, `Δworst_abs(B)`) 三元组
  · 于是我们能画出 **`Δworst_abs` 作为 `util` 的函数**，并**读出它归零的临界 `util`**
  · 零下载、零训练：`tp_prefix` 已落盘，任意 `B` 都可离线算

**→ 这条曲线把"边界条件"从一个定性说法，变成一个有临界值的定量判据。**

═══════════════════════════════════════════════════════════════════════════
★ 预注册判据（写在这里，跑之前）
═══════════════════════════════════════════════════════════════════════════
  `Δworst_abs(B) = worst_waterfill(B) − worst_uniform(B)`（同一总花费 S = B·N 下）

  若存在 `B*` 使 `Δworst_abs` 在 `B < B*` 时显著为正、在 `B > B*` 时 ≈ 0
    → **边界存在**，报告 `B*` 与对应的 `util(B*)`
  若 `Δworst_abs` 处处为正或处处 ≈ 0
    → **无边界**，须如实报告（则边界条件那条主张要降级）
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rdr", ROOT / "scripts" / "run_decision_rule.py")
rdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rdr)

BUDGETS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200,
           250, 300, 400, 500]


def main() -> int:
    print("=" * 112)
    print("Δworst_abs 作为预算 B 的函数（真实检测器输出；B·N 为总花费）")
    print("=" * 112)

    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        if not recs:
            print(f"\n⚠️ {head} 无数据")
            continue
        n = np.array([r["n_gt"] for r in recs], dtype=float)
        p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
        N = len(recs)
        TOT = float(n.sum())

        print(f"\n{'#' * 112}\n### val / {head}   N={N}  Σnᵢ={TOT:.0f}  Σpᵢ={p.sum():.0f}")
        print(f"\n  {'B':>5} {'S=B·N':>8} {'ρ=S/Σnᵢ':>8} │ {'ceiling':>8} {'util':>7} "
              f"{'frac>B':>7} │ {'worst 均匀':>11} {'worst 水位':>11} {'Δ pt':>8} │ 判定")
        print("  " + "-" * 106)

        rows = []
        for B in BUDGETS:
            S = float(B * N)
            q1 = rdr.plan_uniform(recs, S)
            q2 = rdr.plan_waterfill_B(recs, B) if hasattr(rdr, "plan_waterfill_B") else \
                np.minimum(np.ceil(B * n / max(n.mean(), 1e-9)), p)
            r1 = rdr.evaluate(recs, q1)
            r2 = rdr.evaluate(recs, q2)
            d = (r2["worst"] - r1["worst"]) * 100

            spent = float(np.minimum(B, n).sum())
            ceiling = spent / TOT
            util = spent / (B * N)
            frac = float((n > B).mean())

            verdict = ("✅ 分配有意义" if d >= 5 else ("⚠️ 弱" if d >= 1 else "❌ 无"))
            print(f"  {B:>5} {S:>8.0f} {S / TOT:>8.3f} │ {ceiling * 100:>7.2f}% "
                  f"{util * 100:>6.1f}% {frac * 100:>6.1f}% │ {r1['worst'] * 100:>10.3f}% "
                  f"{r2['worst'] * 100:>10.3f}% {d:>+7.3f} │ {verdict}")
            rows.append({"B": B, "S": S, "rho": S / TOT, "ceiling": ceiling,
                         "util": util, "frac": frac, "worst_u": r1["worst"],
                         "worst_w": r2["worst"], "d": d,
                         "micro_u": r1["micro"], "micro_w": r2["micro"]})

        # ── 找临界 B ────────────────────────────────────────────
        print("\n  ★ 临界点分析")
        prev_pos = None
        cross = None
        for r in rows:
            pos = r["d"] >= 5.0
            if prev_pos is True and pos is False and cross is None:
                cross = r
            if pos:
                prev_pos = True
            elif prev_pos is True:
                pass
        if cross:
            print(f"     Δworst_abs 首次跌破 5 pt 于 **B = {cross['B']}**"
                  f"（util = {cross['util'] * 100:.1f}%，frac(nᵢ>B) = {cross['frac'] * 100:.1f}%）")
        else:
            print("     ⚠️ Δworst_abs 在全部 B 上均 ≥ 5 pt 或均 < 5 pt —— **未观察到临界点**")

        # ── Δworst vs util 的相关性 ──────────────────────────────
        us = np.array([r["util"] for r in rows])
        ds = np.array([r["d"] for r in rows])
        fs = np.array([r["frac"] for r in rows])
        print(f"\n  ★ 相关性（Δworst_abs 与各代理量）")
        for lab, x in (("util(B)", us), ("frac(nᵢ>B)", fs), ("ceiling(B)", np.array([r["ceiling"] for r in rows]))):
            if x.std() > 1e-12:
                rho = float(np.corrcoef(x, ds)[0, 1])
                print(f"     Pearson r(Δworst, {lab:<12}) = {rho:+.3f}")

        # ── micro 的代价是否随 B 变化 ────────────────────────────
        print(f"\n  ★ micro 代价：Δmicro 范围 = "
              f"[{min(r['micro_w'] - r['micro_u'] for r in rows) * 100:+.3f}, "
              f"{max(r['micro_w'] - r['micro_u'] for r in rows) * 100:+.3f}] pt"
              f"   （→ 总召回的守恒性在各 B 下是否稳定）")

    print("\n" + "=" * 112)
    print("★ 判读")
    print("=" * 112)
    print("""  · 若 Δworst_abs 随 util(B) 单调上升、并在某个 util 以下归零
     → **边界条件从定性说法升级为带临界值的定量判据**（这是论文最强的形态）
  · 若 Δworst_abs 处处为正、不随 util 衰减
     → 边界条件那条主张**站不住**，必须降级（诚实报告）
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
