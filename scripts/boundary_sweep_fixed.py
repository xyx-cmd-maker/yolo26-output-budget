"""
★★ 修正版边界扫描：**强制两种分配同总花费**（修 P39）

═══════════════════════════════════════════════════════════════════════════
上一版（`budget_sweep_boundary.py`）的两个缺陷
═══════════════════════════════════════════════════════════════════════════
**缺陷 1（致命）**：水填法用 `ceil(B·nᵢ/mean(n))` 近似 → 当 `B > mean(n)` 时
  每图配额可**超过 B** → `Σ spend` 远大于 `B·N` → **两种分配不是同总花费**。
  这违反了判据的 N1（花费相对误差 ≤ 1%），使整张表**不可用于比较**。

**缺陷 2**：小 `B`（如 5）时整数化使 `⌈ρnᵢ⌉` 对某些图取到 0 → 那些图召回 0，
  于是水位法的 `worst` 被"取整噪声"支配，而不是被分配策略支配。

═══════════════════════════════════════════════════════════════════════════
★ 本版的正确做法
═══════════════════════════════════════════════════════════════════════════
对每个目标总花费 `S`：
  1. **均匀**：二分最大 `c` 使 `Σ min(c, nᵢ, pᵢ) ≤ S` → `qᵢ = min(c, nᵢ, pᵢ)`
  2. **水位（整数）**：二分最大 `ρ` 使 `Σ min(⌈ρ·nᵢ⌉, pᵢ, nᵢ) ≤ S` → `qᵢ = min(⌈ρ·nᵢ⌉, pᵢ, nᵢ)`
  3. **补足差额**：两者都可能略欠 `S`。**把差额按最大边际收益贪心补到某个图上**，
     使两者的 `Σ spend` 相对误差 ≤ 1%（满足 N1）
  4. 只有通过 N1 的 (S, 均匀, 水位) 三元组才进入比较

**→ 这样比较的是"同一总花费下，均匀 vs 按 nᵢ 比例"的真实差异。**

★ 预注册判据（跑之前写下）
  报告 `Δworst_abs(S)` 与 `util(S)` 的散点；
  若存在一个 `util` 阈值使 `Δworst_abs` 在其上下**符号稳定不同** → 边界成立；
  若 `Δworst_abs` 恒为正（或恒为负）→ **边界条件主张必须降级或撤回**。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rdr", ROOT / "scripts" / "run_decision_rule.py")
rdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rdr)

OUT = ROOT / "results" / "tables" / "boundary-sweep.json"

# ★★ 第 49 轮（P42）：**删除本文件里重复的实现，改为 import 唯一来源。**
#   我曾在此处另写一份 `plan_uniform_eq` / `plan_waterfill_eq` / `topup`，
#   而 `run_decision_rule.py` 里又有一份旧版 → **同一算法两份实现静默漂移**，
#   导致手稿 Table 2 的 Δmicro/Δmacro 与判据实验对不上。
#   → 现在这些函数**只在 `run_decision_rule.py` 定义一次**。
cap_of = rdr._cap_of
plan_uniform_eq = rdr.plan_uniform_eq
plan_waterfill_eq = rdr.plan_waterfill_eq
topup = rdr.topup


def ev(recs, q):
    return rdr.evaluate(recs, q)


def main() -> int:
    targets = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500]
    all_out = {}

    print("=" * 118)
    print("修正版边界扫描：**同总花费**下 均匀 vs 整数水填（满足 N1 才采纳）")
    print("=" * 118)

    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        if not recs:
            continue
        n, p, cap = cap_of(recs)
        N = len(recs)
        TOT = float(n.sum())

        print(f"\n{'#' * 118}\n### val / {head}   N={N}  Σnᵢ={TOT:.0f}")
        print(f"\n  {'B':>5} {'S=B·N':>7} │ {'util':>7} {'frac>B':>7} │ {'spend/S 均':>11} "
              f"{'spend/S 水':>11} {'N1':>4} │ {'worst 均':>10} {'worst 水':>10} {'Δ pt':>8} │ 判定")
        print("  " + "-" * 112)

        rows = []
        for B in targets:
            S = float(B * N)
            q1 = topup(recs, plan_uniform_eq(recs, S), S)
            q2 = topup(recs, plan_waterfill_eq(recs, S), S)
            r1, r2 = ev(recs, q1), ev(recs, q2)

            s1 = r1["spend"] / S
            s2 = r2["spend"] / S
            n1 = abs(s1 - 1) <= 0.01 and abs(s2 - 1) <= 0.01
            d = (r2["worst"] - r1["worst"]) * 100

            spent = float(np.minimum(B, n).sum())
            util = spent / (B * N)
            frac = float((n > B).mean())

            verdict = "—" if not n1 else ("✅ 有意义" if d >= 5 else ("⚠️ 弱" if d >= 1 else "❌ 无"))
            print(f"  {B:>5} {S:>7.0f} │ {util * 100:>6.1f}% {frac * 100:>6.1f}% │ "
                  f"{s1:>11.4f} {s2:>11.4f} {'✅' if n1 else '❌':>4} │ "
                  f"{r1['worst'] * 100:>9.3f}% {r2['worst'] * 100:>9.3f}% {d:>+7.3f} │ {verdict}")
            rows.append({"B": B, "S": S, "util": util, "frac": frac,
                         "spend_u": s1, "spend_w": s2, "N1": bool(n1),
                         "worst_u": r1["worst"], "worst_w": r2["worst"], "d": d,
                         "micro_u": r1["micro"], "micro_w": r2["micro"]})

        ok = [r for r in rows if r["N1"]]
        print(f"\n  ★ 通过 N1 的档位：{len(ok)} / {len(rows)}")
        if ok:
            us = np.array([r["util"] for r in ok])
            ds = np.array([r["d"] for r in ok])
            fs = np.array([r["frac"] for r in ok])
            print(f"  ★ Δworst_abs 范围 = [{ds.min():+.3f}, {ds.max():+.3f}] pt")
            print(f"     负值档位数 = {int((ds < 0).sum())} / {len(ds)}"
                  f"   （>0 表示**与「分配有益」相矛盾**——必须如实报告）")
            for lab, x in (("util", us), ("frac>B", fs)):
                if x.std() > 1e-12:
                    print(f"     Pearson r(Δworst, {lab:<8}) = {np.corrcoef(x, ds)[0, 1]:+.3f}")
            print(f"  ★ Δmicro 范围 = [{(np.array([r['micro_w'] for r in ok]) - np.array([r['micro_u'] for r in ok])).min() * 100:+.3f}, "
                  f"{(np.array([r['micro_w'] for r in ok]) - np.array([r['micro_u'] for r in ok])).max() * 100:+.3f}] pt")

        all_out[head] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
