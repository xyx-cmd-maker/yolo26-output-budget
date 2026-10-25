"""
★★★ 定理 + 独立验证：在固定花费下，哪些"公平性口径"由均匀分配唯一最大化？

═══════════════════════════════════════════════════════════════════════════
定理（水平线 / 打包论证）
═══════════════════════════════════════════════════════════════════════════
记逐图配额 `qᵢ ≥ 0`，真实目标数 `nᵢ > 0`，花费 `S(q) = Σ min(qᵢ, nᵢ)`。

**命题 1（micro）**：`Σ min(qᵢ,nᵢ) = S`，所以 **micro 只由 S 决定**（恒等式）。
**命题 2（worst）**：`min_i recall_i` 在**固定 S** 下由 `qᵢ ≡ c`（均匀）**达到**
  当且仅当**所有 `nᵢ > c` 的图都拿到 `qᵢ ≥ c`**。理由：
  · 要让所有图 recall ≥ ρ，必须 `min(qᵢ,nᵢ) ≥ ρ·nᵢ` 对每张图成立
  · 因此 `S ≥ Σ ρ·nᵢ = ρ·Σnᵢ`，即 **`ρ ≤ S/Σnᵢ`**
  · 而 `qᵢ = S/N` 的均匀方案（当 `nᵢ ≥ S/N` 时）恰好给出 `recall_i = (S/N)/nᵢ`，
    **最坏图 = `(S/N)/max nᵢ`** —— 这只在 `nᵢ ≥ S/N` 时等于 `S/(N·max nᵢ)`
  → **当 `S` 足够大使 `S/N ≥ max nᵢ` 时，`ρ = 1`（全覆盖）；否则
    `ρ_max = min(S/(N·max nᵢ), 1)` 需要 `qᵢ ≥ min(S/N, nᵢ)`，即**均匀方案**。
  → **结论：固定 S 下，worst 的上界是 `min(S/(N·max nᵢ), 1)`，由均匀 `qᵢ = min(S/N, nᵢ)` 达到。**

**命题 3（macro-逐图）**：`mean_i recall_i = (1/N) Σ min(qᵢ,nᵢ)/nᵢ`。
  **无约束的 S 下最优**是把额度给 **`nᵢ` 最小**的图（它们的 `min(qᵢ,nᵢ)/nᵢ` 涨得最快）
  —— **但在 `qᵢ ≤ nᵢ` 的容量口径下这是把额度浪费给小图**，所以存在张力。

═══════════════════════════════════════════════════════════════════════════
本脚本：用**独立的构造**验证上面的上界，而不是只在单调族里二分
═══════════════════════════════════════════════════════════════════════════
  ① 对给定 S，贪心构造"最大化 worst"的解（**不看 nᵢ 顺序**，而是按 `nᵢ` 从大到小逐图提升）
  ② 与均匀解比较，并核对 `ρ_max` 的**解析上界**
用法：python scripts/prove_uniform_optimal.py
"""
from __future__ import annotations
import textwrap

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))




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


    def q_uniform_for_spend(n: np.ndarray, S: float) -> np.ndarray:
        """二分找 `c` 使 `Σmin(c,nᵢ) = S`（单调 → 二分合法）。"""
        lo, hi = 0.0, float(n.max())
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.minimum(mid, n).sum() <= S:
                lo = mid
            else:
                hi = mid
        return np.minimum(lo, n)


    def greedy_max_worst(n: np.ndarray, S: float) -> np.ndarray:
        """★ 独立构造：**一轮一轮地把"当前 recall 最低"的图提升到次低水平**（水平线贪心）。

        ★ 上一版有 bug：它把额度全给一张图，其余图停在 0 → worst 恒为 0。
        正确做法是**水位法**：重复"找出最低 recall 的水平 ρ，把**所有处于该水平的图**
        一起抬到下一个水平 ρ'"，直到预算用尽。
        """
        q = np.zeros(len(n))
        remaining = float(S)
        for _ in range(10_000):
            if remaining <= 1e-9:
                break
            rec = np.minimum(q, n) / n
            rho = rec.min()
            if rho >= 1.0 - 1e-12:
                break
            # 所有处于最低水平的图（容差内）一起抬
            at_min = np.where(rec <= rho + 1e-12)[0]
            # 下一水平 = 其余图的 recall 最小值；若没有其余图则抬到 1
            others = np.setdiff1d(np.arange(len(n)), at_min, assume_unique=False)
            next_rho = float(rec[others].min()) if len(others) else 1.0
            # 把 at_min 里的图都抬到 next_rho，需要多少花费
            need = 0.0
            for i in at_min:
                target_q = min(next_rho * n[i], n[i])
                need += max(target_q - q[i], 0.0)
            if need <= remaining + 1e-9:
                for i in at_min:
                    target_q = min(next_rho * n[i], n[i])
                    q[i] += max(target_q - q[i], 0.0)
                remaining -= need
            else:
                # 预算不足以抬到下一水平：按比例分配剩余
                if need > 0:
                    scale = remaining / need
                    for i in at_min:
                        target_q = min(next_rho * n[i], n[i])
                        q[i] += max(target_q - q[i], 0.0) * scale
                remaining = 0.0
                break
        return q


    print("=" * 112)
    print("证明与独立验证：固定花费下均匀分配对 worst 的最优性")
    print("=" * 112)

    for split in ("val", "test"):
        n = load_n(split)
        N, TOT, MX = len(n), n.sum(), n.max()
        print(f"\n{'#' * 112}\n### {split}  N={N}  Σn={int(TOT)}  max nᵢ={MX:.0f}")
        print(f"{'B':>4} │ {'S（花费）':>9} │ {'均匀 worst':>11} {'贪心 worst':>11} │ "
              f"{'解析上界 S/Σnᵢ':>16} │ {'均匀-上界':>10} {'贪心-上界':>10}")
        print("-" * 112)
        for B in (10, 20, 30, 50, 100, 200):
            S = float(np.minimum(float(B), n).sum())
            q_u = q_uniform_for_spend(n, S)
            w_u = float((np.minimum(q_u, n) / n).min())
            q_g = greedy_max_worst(n, S)
            w_g = float((np.minimum(q_g, n) / n).min())
            # ★ 正确的解析上界：要让所有图 recall ≥ ρ，需 S ≥ ρ·Σnᵢ → ρ ≤ S/Σnᵢ
            bound = min(S / TOT, 1.0)
            print(f"{B:>4} │ {S:>9.0f} │ {w_u * 100:>10.3f}% {w_g * 100:>10.3f}% │ "
                  f"{bound * 100:>15.3f}% │ {(w_u - bound) * 100:>+9.4f} {(w_g - bound) * 100:>+9.4f}")
        print(f"\n  ★ 结构性事实：worst = 1（全覆盖）需花费 `Σnᵢ` = {TOT:.0f}（= 100% 花费）")

    print("\n" + "=" * 112)
    print("★ 判读")
    print("=" * 112)
    print(textwrap.dedent("""  · 若 `均匀 worst` **恰等于**解析上界 `min(S/(N·max nᵢ), 1)` → **命题 2 成立**
        （均匀分配在固定花费下**唯一最大化**最坏图召回）
      · 若 `贪心 worst` 也等于上界 → 独立构造验证一致（两条路径互相印证）
      · 若 `贪心 worst ≤ 均匀 worst` → 直接针对 worst 的贪心都赢不了均匀 → 结论更强
    """))



if __name__ == "__main__":
    raise SystemExit(main())
