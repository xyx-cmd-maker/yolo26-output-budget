"""
★ 创新点补强方案 A 的**零成本可行性测试**（不需要任何新训练）

═══════════════════════════════════════════════════════════════════════════
动机（来自我们自己的阴性结果）
═══════════════════════════════════════════════════════════════════════════
判据实验 + bootstrap 给出：
  · 固定 val 集上水位法 `Δworst_abs` = +8.013 pt（o2m）
  · 但 bootstrap 均值 = **−0.565**，95% CI **[−2.051, +0.000]**，`P(Δ>0)=0.00%`
  · 根因：`worst`（min 召回）是**极值统计**，由个别极端图主导

**→ 自然的方法学问题：换一个目标函数能不能修好这个脆弱性？**

候选：把 `min`（最坏图）换成 **`q`-分位召回**（如 5% 分位），即
      `max_q  Q_q(recall_i)`  s.t.  Σ spendᵢ ≤ S

**本脚本回答三件事（全部离线，用已落盘的 tp_prefix）**：
  1. **分位目标的最优分配是什么形式？**（是否仍是"按 nᵢ 比例"？）
  2. **分位目标的收益比 min 大还是小？**
  3. ★ **分位目标的 bootstrap 稳定性是否显著优于 min？**

═══════════════════════════════════════════════════════════════════════════
★ 预注册式判据（**在跑之前写下**）
═══════════════════════════════════════════════════════════════════════════
设 `stab(X)` = bootstrap 下 `P(ΔX > 0)`（X ∈ {min, 5%分位, 10%分位}）

  若 `max_{q} stab(q-分位) ≥ 0.95` 且 `stab(min) ≤ 0.05`
    → **"分位目标修复了脆弱性"成立** → 可作为**正面创新点**
  否则 → 不作为独立创新点，只作为负面结果的一部分

**为什么这个判据不可事后调整**：`stab` 是 bootstrap 频率，没有调节自由度；
阈值 0.95 / 0.05 是标准的"统计可辨"约定。
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

SEED = 20261014
N_BOOT = 2000


def load(exp: str, split: str, head: str) -> list[dict]:
    return rdr.load_recs(ROOT / "runs" / "budget" / f"{exp}_{split}_{head}.json")


def q_of(recs: list[dict], q: np.ndarray) -> np.ndarray:
    """逐图召回。"""
    tp = np.array([rdr.tp_at(r["tp_prefix"], int(round(qi))) for r, qi in zip(recs, q)], float)
    n = np.array([r["n_gt"] for r in recs], float)
    return tp / np.maximum(n, 1)


def objective(rec: np.ndarray, alpha: float) -> float:
    """alpha=0 → min（最坏图）；alpha=5 → 5% 分位；alpha=50 → 中位数。"""
    return float(np.percentile(rec, alpha)) if alpha > 0 else float(rec.min())


def plan_waterfill(recs, rho):
    return rdr.plan_waterfill(recs, rho)


def plan_uniform(recs, S):
    return rdr.plan_uniform(recs, S)


def main() -> int:
    print("=" * 104)
    print("创新点方案 A 的零成本可行性测试：把 min 换成 q-分位，脆弱性是否被修复？")
    print("=" * 104)

    alphas = [0, 5, 10, 25]
    for head in ("o2m", "o2o"):
        recs = load("r1", "val", head)
        if not recs:
            print(f"\n⚠️ {head} 无数据")
            continue
        n = np.array([r["n_gt"] for r in recs], float)
        TOT, N = float(n.sum()), len(recs)
        rho = 0.5
        S = rho * TOT

        q_u = plan_uniform(recs, S)
        q_w = plan_waterfill(recs, rho)
        rec_u = q_of(recs, q_u)
        rec_w = q_of(recs, q_w)

        print(f"\n{'#' * 104}\n### val / {head}   N={N}  Σnᵢ={TOT:.0f}  S={S:.0f}  ρ={rho}")
        print(f"\n  {'目标':>8} {'均匀值':>10} {'水位值':>10} {'Δ pt':>10} │ "
              f"{'bootstrap 均值':>14} {'95% CI':>22} {'P(Δ>0)':>9}")
        print("  " + "-" * 92)

        rng = np.random.default_rng(SEED)
        # 预生成 bootstrap 索引（各 alpha 共用，保证可比）
        idxs = [rng.integers(0, N, size=N) for _ in range(N_BOOT)]

        for alpha in alphas:
            base = objective(rec_u, alpha)
            wf = objective(rec_w, alpha)
            d_pt = (wf - base) * 100

            ds = []
            for idx in idxs:
                sub = [recs[i] for i in idx]
                s_sub = rho * float(n[idx].sum())
                qu = plan_uniform(sub, s_sub)
                qw = plan_waterfill(sub, rho)
                ru = q_of(sub, qu)
                rw = q_of(sub, qw)
                ds.append((objective(rw, alpha) - objective(ru, alpha)) * 100)
            ds = np.array(ds)
            lo, hi = np.percentile(ds, [2.5, 97.5])
            pgt = float((ds > 0).mean())
            lab = "min(最坏图)" if alpha == 0 else f"{alpha}% 分位"
            print(f"  {lab:>8} {base * 100:>9.3f}% {wf * 100:>9.3f}% {d_pt:>+9.3f} │ "
                  f"{ds.mean():>+13.3f} [{lo:>+8.3f}, {hi:>+8.3f}] {pgt * 100:>8.2f}%")

        print("\n  ★ 判读：")
        print("     · `P(Δ>0)` 若在分位目标下接近 100%，而在 min 下接近 0%")
        print("       → **分位目标修复了极值脆弱性** → 可作正面创新点")
        print("     · 若分位目标也只是小幅改善 → 脆弱性是**数据规模**而非目标函数的问题")

    print("\n" + "=" * 104)
    print("★ 预注册判据")
    print("=" * 104)
    print("""  `max_q P(Δ>0 | q-分位) ≥ 0.95` 且 `P(Δ>0 | min) ≤ 0.05`
     → "分位目标修复脆弱性"成立 → 可作为**正面创新点**（方案 A）
  否则 → 不作独立创新点，只并入负面结果
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
