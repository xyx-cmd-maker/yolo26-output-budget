"""
★★ 关键核算：`uniform(c*)` 到底是不是一条**合法的硬预算**方案？

═══════════════════════════════════════════════════════════════════════════
问题
═══════════════════════════════════════════════════════════════════════════
"每图给同一个配额 `kᵢ = c`" 若要在**硬预算** `Σkᵢ ≤ B·N` 下成立，
必须 `N·c ≤ B·N`，即 **`c ≤ B`**。
但实测 `c* > B`（test B=20 时 c*=22，B=50 时 c*=462 = 9.24×）。

**所以 `uniform(c*)` 的配额之和 `Σkᵢ = N·c*` 超过预算。**
它只有在下面这种**更弱的**预算解释下才合法：

  **（A）硬预算解释**：`Σkᵢ ≤ B·N`（**总输出数**受限）
       → `c ≤ B` 必须成立 → **`c*` 不合法** → 一切"紧 cap 收益"都要重算
  **（B）软预算/成本解释**：实际**花费**（真正输出的框数）`Σ min(kᵢ,nᵢ) ≤ B·N`
       → `c*` 合法（它正是这个式子的最大解）→ 但**实际逐图输出数 `min(c*,nᵢ)` 不相同**，
         所以这叫"**统一上限、自适应输出**"，**不是**"统一输出数"

本脚本把两个解释下**各自**的最优方案算出来并对比，一次性把口径钉死。
用法：python scripts/budget_semantics_audit.py
"""
from __future__ import annotations
import textwrap

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import ceiling_uniform, uniform_cap  # noqa: E402




def main() -> int:
    def load(split: str) -> np.ndarray:
        cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
        d = Path(cfg["path"]) / cfg[split]
        p = list(d.parts)
        for i in range(len(p) - 1, -1, -1):
            if p[i] == "images":
                p[i] = "labels"
                break
        return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                         for f in sorted(Path(*p).glob("*.txt"))], dtype=float)


    def recall(counts, k):
        return float(np.minimum(k, counts).sum() / counts.sum())


    def oracle_hard(counts, budget):
        """★ 解释 (A) 下的最优：`Σkᵢ ≤ budget` 且 `0 ≤ kᵢ ≤ nᵢ`，最大化 Σmin(kᵢ,nᵢ)=Σkᵢ。

        因为在此约束下 `Σmin = Σk`，所以**任何**用满预算的方案都最优 —— 纯组合问题。
        """
        N = len(counts)
        k = np.minimum(np.floor(budget / N), counts)          # 先均匀摊
        left = budget - k.sum()
        for i in np.argsort(-counts):                          # 把余量给最密的图
            if left <= 0:
                break
            room = counts[i] - k[i]
            if room > 0:
                take = min(room, left)
                k[i] += take
                left -= take
        assert k.sum() <= budget + 1e-9, f"超预算 {k.sum()} > {budget}"
        assert (k <= counts + 1e-9).all(), "超过真实需求"
        return k


    def uniform_soft(counts, B):
        """解释 (B) 的均匀方案：每图给 c*（Σmin(c*,nᵢ) ≤ B·N），实际输出 min(c*,nᵢ)。"""
        c = uniform_cap(counts, B)
        return np.full(len(counts), float(c)), c


    print("=" * 104)
    print("预算语义审计：`uniform(c*)` 在两种解释下是否合法？")
    print("=" * 104)

    for split in ("val", "test"):
        counts = load(split)
        N, TOT = len(counts), counts.sum()
        print(f"\n{'#' * 104}\n### {split}   N={N}   Σnᵢ={int(TOT)}")
        print(f"{'B':>5} {'预算B·N':>9} │ {'c*':>5} {'N·c*':>9} {'N·c*/预算':>10} │ "
              f"{'(A)硬预算最优':>13} {'(A)是否可行':>11} │ {'(B)均匀c*召回':>13} │ 差异")
        print("-" * 104)
        for B in (20, 30, 40, 50, 75, 100, 150):
            budget = B * N
            cstar = uniform_cap(counts, B)
            sum_quota = N * cstar
            # 解释 (A)：硬预算下的最优
            kA = oracle_hard(counts, budget)
            vA = recall(counts, kA)
            feasible_A = (sum_quota <= budget + 1e-9)
            # 解释 (B)：均匀 c*
            kB, _ = uniform_soft(counts, B)
            vB = recall(counts, kB)
            # 解释 (A) 下"每图给 B"也合法（N·B = budget）
            vA_unif = recall(counts, np.full(N, float(B)))
            print(f"{B:>5} {budget:>9} │ {cstar:>5} {sum_quota:>9} {sum_quota / budget:>9.2f}× │ "
                  f"{vA * 100:>12.2f}% {str(feasible_A):>11} │ {vB * 100:>12.2f}% │ "
                  f"{(vB - vA) * 100:>+6.2f}pt")
            print(f"        └ (A) 下「每图给 B」= {vA_unif * 100:.2f}%（合法，Σk = 预算）")

    print("\n" + "=" * 104)
    print("★ 判读")
    print("=" * 104)
    print(textwrap.dedent("""  · 若 `N·c*/预算 > 1`（本表会显示倍数）：
          → **`uniform(c*)` 在「硬预算（总输出数 ≤ B·N）」解释下不合法**。
          → 它只在「软预算（实际花费 ≤ B·N）」解释下合法，此时逐图**实际输出数不同**，
            应称「**统一上限、自适应输出**」，而不是「统一配额」。

      · 两种解释下的**最优召回**若相同 → 预算语义不影响结论，只是叫法问题。
      · 若不同 → **必须明确规定用哪种**，且所有数字按该规定重算。
    """))



if __name__ == "__main__":
    raise SystemExit(main())
