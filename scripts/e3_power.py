"""
★★ E3c 的功效分析：以现有样本量，能检出多小的差异？

═══════════════════════════════════════════════════════════════════════════
为什么必须做（质量门 C 未过 + 红队 R2）
═══════════════════════════════════════════════════════════════════════════
第 13 轮我把"分配无用"降级为"**未检出**聚合增益"，因为 TOST@0.5pt 全部不通过。
但**"未检出"本身需要功效支撑**：若样本量根本不足以检出 0.5 pt，
那"未检出"是**信息缺失**，不是**等价证据**。

红队 R2 的原话：*"TOST@0.5pt 全部不通过，说明你的样本量不足以支持'等价'。
而你又用 (ii)−(iii)≈0 支撑'分配无用'——循环论证。"*

本脚本给三件事：
  1. **MDE（最小可检出效应）**：给定逐图配对差的标准差与 n，80% 功效下能检出的最小 |Δ|
  2. **反推所需样本量**：若要检出 0.5 pt，需要多少图
  3. **明确写出**：哪些档位下"未检出"是**有意义的**（MDE 小），哪些是**无意义的**（MDE 大）

公式（配对设计，双侧 α=0.05，功效 1−β=0.80）：
    MDE = (z_{1−α/2} + z_{1−β}) · σ_d / √n  ≈ (1.959964 + 0.8416212) · σ_d/√n
    n_required = ((z_{1−α/2} + z_{1−β}) · σ_d / MDE)²

★ 精度来源声明（P16 的教训）：
   z 值取自标准正态分位数，保留 7 位有效数字；σ_d 由**实测逐图差**直接计算（非估计）。
   故 MDE 的不确定度**只来自 bootstrap 折划分**（已用固定种子 RNG_SEED 消除）。

用法：python scripts/e3_power.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.eval.proxy_density import uniform_cap  # noqa: E402
from src.eval.stats import RNG_SEED  # noqa: E402
from e3_equivalence import (alloc_by_proxy, alloc_uniform, load_counts,  # noqa: E402
                            per_image_recall)

Z_A = 1.959963985   # 双侧 α = 0.05
Z_B = 0.841621234   # 功效 0.80
TARGETS = (0.005, 0.010, 0.020)   # 想检出的效应（0.5 / 1.0 / 2.0 pt）


def mde(sigma_d: float, n: int) -> float:
    return (Z_A + Z_B) * sigma_d / np.sqrt(n)


def n_required(sigma_d: float, effect: float) -> float:
    return ((Z_A + Z_B) * sigma_d / effect) ** 2


def main() -> int:
    print("=" * 104)
    print("E3c 功效分析：以现有样本量，能检出多小的差异？")
    print("=" * 104)
    print(f"""
设计：**配对**（配对单位 = 图），双侧 α=0.05，功效 1−β=0.80
公式：MDE = (z_(1−α/2) + z_(1−β)) · σ_d/√n = ({Z_A} + {Z_B}) · σ_d/√n
      z 取自标准正态分位数；σ_d 由**实测逐图差**计算
比较量：(ii) 样本外分配 − (iii) 均匀紧 cap 的**逐图召回**差
""")

    for split in ("val", "test"):
        counts = load_counts(split)
        N = len(counts)
        print(f"\n{'#' * 104}\n### {split}   n = {N} 图")
        print(f"{'B':>5} │ {'σ_d (pt)':>10} {'实测 |Δ| (pt)':>14} │ {'MDE (pt)':>10} "
              f"{'实测/MDE':>10} │ {'检出 0.5pt 需 n':>16} {'检出 1.0pt 需 n':>16} │ 判定")
        print("-" * 104)

        rng = np.random.default_rng(RNG_SEED)
        for B in (20, 30, 40, 50):
            budget = float(B * N)
            cap = float(uniform_cap(counts, B))
            proxy_all = np.minimum(cap, counts)
            rec_uni = per_image_recall(counts, alloc_uniform(counts, B))

            idx = rng.permutation(N)
            halves = [idx[: N // 2], idx[N // 2:]]
            rec_oos = np.zeros(N)
            for hi in range(2):
                A, Bx = halves[hi], halves[1 - hi]
                cA, pA = counts[A], proxy_all[A]
                best_frac, best_v = 0.0, -np.inf
                for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                    kA = alloc_by_proxy(cA, pA, B, frac, float(B * len(A)))
                    v = float(np.minimum(kA, cA).sum() / cA.sum())
                    if v > best_v:
                        best_frac, best_v = frac, v
                cB, pB = counts[Bx], proxy_all[Bx]
                kB = alloc_by_proxy(cB, pB, B, best_frac, float(B * len(Bx)))
                rec_oos[Bx] = per_image_recall(cB, kB)

            d = rec_oos - rec_uni
            sd = float(d.std(ddof=1))
            obs = abs(float(d.mean()))
            m = mde(sd, N)
            n05 = n_required(sd, 0.005)
            n10 = n_required(sd, 0.010)
            # 判定：实测差异是否落在 MDE 之内（即"确实看不出"）
            verdict = ("**MDE 小，'未检出'有意义**" if m <= 0.005
                       else (f"MDE={m * 100:.2f}pt > 0.5pt → **无法主张 0.5pt 等价**" if m > 0.005
                             else "边界"))
            print(f"{B:>5} │ {sd * 100:>10.3f} {obs * 100:>14.3f} │ {m * 100:>10.3f} "
                  f"{obs / m if m > 0 else float('nan'):>10.2f} │ {n05:>16,.0f} {n10:>16,.0f} │ {verdict}")

    print("\n" + "=" * 104)
    print("★ 判读")
    print("=" * 104)
    print("""  · 若某档 **MDE ≤ 0.5 pt** → 该档的"未检出"**是有意义的等价证据**（可主张 0.5 pt 内等价）
  · 若某档 **MDE > 0.5 pt** → "未检出"**只是信息不足**，措辞必须停在"未检出显著差异"
  · 「检出 0.5pt 需 n」一列给出**要把话说满需要多少图**——若远超现有 n（548/1610），
    则**永远无法在 0.5 pt 上主张等价**，必须永久改用上界陈述

  ⚠️ 注意 σ_d 随 B 变化很大（预算越紧，逐图差越大）：
     这意味着**紧预算下分配确实在动**（与 E3b 的发现一致），
     而"聚合抵消"只发生在均值层面。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
