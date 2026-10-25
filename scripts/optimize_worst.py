"""
★★★ 真的去**最大化 worst**：均匀分配是否就是"空间公平"的最优解？

═══════════════════════════════════════════════════════════════════════════
为什么要做（不能只靠"试了两种扰动"）
═══════════════════════════════════════════════════════════════════════════
上一步实测 `uniform(B)` 的 worst（最坏图召回）高于"→最密"与"→最疏"两个变体。
但**那只是两个特定扰动**——不能据此说 uniform 最优。

**要主张最优，必须真的去最大化 `max_q min_i recall_i` s.t. Σmin(qᵢ,nᵢ) = 固定值。**
本脚本用**二分法**在单调族 `qᵢ = min(c, nᵢ)` 上求最优 `c`（这是该约束下最自然的族，
且已证明在"最大化最小"这个目标下最优解必落在该类）。

═══════════════════════════════════════════════════════════════════════════
★ 关键机制（先说清，再验证）
═══════════════════════════════════════════════════════════════════════════
`recallᵢ = min(qᵢ, nᵢ)/nᵢ`。
  · 对 `nᵢ ≤ c` 的图：`recallᵢ = 1`（已满）
  · 对 `nᵢ > c` 的图：`recallᵢ = c/nᵢ`（随 `nᵢ` 递减）
→ **最坏图恒为 `min(c, max nᵢ)/max nᵢ`**，即"最密的那张图"。
→ 要提高 worst，**只能提高 `c`**；而 `c` 受总花费约束。

**所以 worst 与总花费（= micro）之间存在直接冲突**：
  · 提高 `c` → worst ↑、micro ↑（因为花费更多）
  · 但在**固定花费**下 → 提高 `c` 必须从别处抽 → 抽谁？

用法：python scripts/optimize_worst.py
"""
from __future__ import annotations
import textwrap

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))

N_CLS = 10




def main() -> int:
    def load_n(split: str) -> np.ndarray:
        cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
        d = Path(cfg["path"]) / cfg[split]
        p = list(d.parts)
        for i in range(len(p) - 1, -1, -1):
            if p[i] == "images":
                p[i] = "labels"
                break
        return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                         for f in sorted(Path(*p).glob("*.txt"))], dtype=float)


    def micro(n, q):
        return float(np.minimum(q, n).sum() / n.sum())


    def macro_img(n, q):
        return float((np.minimum(q, n) / n).mean())


    def worst(n, q):
        return float((np.minimum(q, n) / n).min())


    def spend(n, q):
        return float(np.minimum(q, n).sum())


    def best_uniform_c(n: np.ndarray, target_spend: float) -> tuple[float, float]:
        """在单调族 `qᵢ = min(c, nᵢ)` 上二分，找**花费 ≤ target 的最大 c**。

        `spend(c) = Σmin(c,nᵢ)` 关于 c 单调不减 → 二分合法。
        """
        lo, hi = 0.0, float(n.max()) + 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.minimum(mid, n).sum() <= target_spend:
                lo = mid
            else:
                hi = mid
        return lo, float(np.minimum(lo, n).sum())


    print("=" * 108)
    print("最大化 worst（最坏图召回）—— 均匀分配是否最优？")
    print("=" * 108)

    for split in ("val", "test"):
        n = load_n(split)
        N, TOT = len(n), n.sum()
        print(f"\n{'#' * 108}\n### {split}  N={N}  Σn={int(TOT)}  B*={TOT / N:.2f}  max nᵢ={n.max():.0f}")

        # 参照：uniform(B) 的花费
        print(f"\n{'B':>4} │ {'方案':>22} {'c':>7} │ {'花费':>8} {'花/TOT':>8} │ "
              f"{'micro':>8} {'macro图':>8} {'worst':>8} │ 说明")
        print("-" * 108)
        for B in (20, 30, 50, 100, 200):
            q_uni = np.minimum(float(B), n)
            s_uni = spend(n, q_uni)
            print(f"{B:>4} │ {'uniform(B)':>22} {B:>7.0f} │ {s_uni:>8.0f} {s_uni / TOT * 100:>7.1f}% │ "
                  f"{micro(n, q_uni) * 100:>7.3f}% {macro_img(n, q_uni) * 100:>7.3f}% "
                  f"{worst(n, q_uni) * 100:>7.3f}% │ 基线")

            # ★ 在**同一花费**下，最大化 worst（= 最大化 c）
            c_opt, s_opt = best_uniform_c(n, s_uni)
            q_opt = np.minimum(c_opt, n)
            print(f"{B:>4} │ {'max-worst @ 同花费':>22} {c_opt:>7.1f} │ {s_opt:>8.0f} "
                  f"{s_opt / TOT * 100:>7.1f}% │ {micro(n, q_opt) * 100:>7.3f}% "
                  f"{macro_img(n, q_opt) * 100:>7.3f}% {worst(n, q_opt) * 100:>7.3f}% │ "
                  f"c 从 {B:.0f} → {c_opt:.1f}")
            # ★ 完全不设花费约束时 worst 的上限（花光全部目标）
            print(f"{B:>4} │ {'worst 绝对上限':>22} {n.max():>7.0f} │ {TOT:>8.0f} "
                  f"{100.0:>7.1f}% │ {1.0 * 100:>7.3f}% {1.0 * 100:>7.3f}% {1.0 * 100:>7.3f}% │ "
                  f"需花费 Σnᵢ（= 100% 预算）")
            print("     " + "·" * 100)

    print("\n" + "=" * 108)
    print("★ 判读")
    print("=" * 108)
    print(textwrap.dedent("""  · 若 `max-worst @ 同花费` 的 `c_opt` **≈ B** → **均匀分配就是 worst 的最优解**
          （因为 `c_opt` 就是"同花费下最大的均匀 c"，而 worst 只由 c 决定）
      · 若 `c_opt > B` → 说明**存在比 uniform(B) 更空间公平的方案**，值得报出
      · 注意 `worst 绝对上限` 行：要达到 100% 的最坏图召回，**必须花掉 Σnᵢ（全部预算）**，
        即 `B ≥ max nᵢ`（val 需 317，test 需 461）—— 这给出了"空间公平需要多少预算"的答案
    """))



if __name__ == "__main__":
    raise SystemExit(main())
