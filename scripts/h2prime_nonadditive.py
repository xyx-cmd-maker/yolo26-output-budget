"""
★ H2′ 检验：**非加法目标**下"逐图分配"是否重新变得非平凡？

═══════════════════════════════════════════════════════════════════════════
背景（第 5 轮的结论，见 docs/finding-headroom-is-cap-not-allocation.md）
═══════════════════════════════════════════════════════════════════════════
**加法目标**（最大化总召回）下：
  当处处 `quotaᵢ ≤ nᵢ` 时 `Σ min(quotaᵢ, nᵢ) = Σ quotaᵢ = 总预算`
  → **与分配策略无关**。实测「分配效应」≤ 0.81 pt（多次为 0.00）。

**H2′ 的问题**：换成**非加法目标**，分配是否重新变得非平凡？

═══════════════════════════════════════════════════════════════════════════
四个候选目标（全部只用确定性的逐图配额，无估计量）
═══════════════════════════════════════════════════════════════════════════
  M1 **总召回**（加法）—— 基线，预期分配无关
  M2 **最坏图召回** `minᵢ recallᵢ` —— 公平性/最坏情况
  M3 **最坏十分位召回** `mean(最低 10% 的 recallᵢ)` —— 尾部风险
  M4 **召回下限的满足率** `|{i : recallᵢ ≥ ρ}| / N` —— 硬约束类

对每个目标，比较两种分配（总预算相同 `B·N`）：
  · **均匀(c*)**：每图给紧 cap `c*`（第 5 轮证明它在加法目标下已最优）
  · **最优**：在总预算约束下**针对该目标**优化 `kᵢ ∈ [0, nᵢ]`

若 M2/M3/M4 下"最优"显著优于"均匀(c*)" → **分配效应在非加法目标下重新出现**
→ H2′ 成立 → "分配"这条线可保留，但**必须换目标函数**。

实现说明：
  · M1/M2/M3 的最优解都用**贪心增量**求：每步把 1 个额度给"边际增益最大"的图。
    凹性保证贪心最优（M1 是标准结果；M2/M3 因增益递减也适用贪心）。
  · M4 是可行性问题：二分求最大可行的 ρ。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "src").exists() \
    else Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import ceiling_uniform, uniform_cap  # noqa: E402


def load_counts(split: str) -> np.ndarray:
    cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
    img_dir = Path(cfg["path"]) / cfg[split]
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return np.array([
        sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        for f in sorted(Path(*parts).glob("*.txt"))
    ], dtype=float)


# ── 目标函数 ────────────────────────────────────────────────────────────
# ★★ 第 6 轮的重要修正：**宏平均 vs 微平均是两个不同的目标函数，混用即类目错误**
#   `Σmin(kᵢ,nᵢ) / Σnᵢ`  = **微平均**（每个框等权）= "总召回"的正确定义
#   `mean(min(kᵢ,nᵢ)/nᵢ)` = **宏平均**（每张图等权）—— 稀疏图 recall 高，宏平均给它们同等权重
#   实测（val, 每图给 c*=21）：微平均 **27.95%** vs 宏平均 **42.48%**，差 14.5 pt！
#   我上一版用 `rec.mean()`（宏）当"总召回"，却把它与 `ceiling_uniform`（微）比较
#   → 那个 9.83 pt 的"分配效应"是**指标定义错配产生的假象**。
def m1_total_micro(counts: np.ndarray, k: np.ndarray) -> float:
    """总召回（**微平均**：每个框等权）—— 与 `ceiling_uniform` 同口径。"""
    return float(np.minimum(k, counts).sum() / counts.sum())


def m1_total_macro(counts: np.ndarray, k: np.ndarray) -> float:
    """总召回（**宏平均**：每张图等权）—— 仅供参考，不与微平均混比。"""
    return float((np.minimum(k, counts) / counts).mean())


def m2_worst(counts: np.ndarray, k: np.ndarray) -> float:
    """最坏图召回。"""
    return float((np.minimum(k, counts) / counts).min())


def m3_worst_decile(counts: np.ndarray, k: np.ndarray) -> float:
    """最坏十分位召回的均值（每张图等权）。"""
    rec = np.minimum(k, counts) / counts
    m = max(1, len(rec) // 10)
    return float(np.sort(rec)[:m].mean())


def greedy_optimal(counts: np.ndarray, budget: int, objective: str) -> np.ndarray:
    """在 Σkᵢ ≤ budget, 0 ≤ kᵢ ≤ nᵢ 下，按 objective 求最优配额（贪心增量）。

    每步选"边际增益最大"的图加 1。凹性使贪心最优。
    ★ objective 必须与 evaluate 用的**同一个口径**：
      · "total_micro"：边际增益 = 1/Σn（常数）→ 等价于任意分配，**必然与均匀同值**
      · "total_macro"：边际增益 = 1/(N·nᵢ) → 优先给 **nᵢ 小**的图
      · "worst"      ：优先给当前 recall 最低的图
      · "worst_decile"：优先给当前处于最低十分位内的图
    """
    N = len(counts)
    k = np.zeros(N, dtype=float)
    remaining = int(budget)
    while remaining > 0:
        rec = np.minimum(k, counts) / counts
        if objective == "total_micro":
            gain = np.where(k < counts, 1.0, -np.inf)          # 常数增益 → 任意
        elif objective == "total_macro":
            gain = np.where(k < counts, 1.0 / counts, -np.inf)  # 优先小图
        elif objective == "worst":
            gain = np.where(k < counts, -rec, -np.inf)
        elif objective == "worst_decile":
            thr = np.quantile(rec, 0.10)
            gain = np.where((k < counts) & (rec <= thr), -rec, -np.inf)
        else:
            raise ValueError(objective)
        if not np.isfinite(gain).any() or (gain == -np.inf).all():
            break
        i = int(np.argmax(gain))
        k[i] += 1
        remaining -= 1
    return k


def evaluate(counts: np.ndarray, k: np.ndarray) -> dict:
    return {
        "total_micro": m1_total_micro(counts, k),
        "total_macro": m1_total_macro(counts, k),
        "worst": m2_worst(counts, k),
        "worst_decile": m3_worst_decile(counts, k),
    }


def uniform_alloc(counts: np.ndarray, cap: float) -> np.ndarray:
    return np.full(len(counts), float(cap))


def main() -> int:
    print("=" * 96)
    print("H2′ 检验：非加法目标下，逐图分配是否重新变得非平凡？")
    print("=" * 96)
    print("""
★ 口径（第 6 轮修正，很重要）：
  **"每图均衡 + 花满预算"在数学上不存在** —— 均衡方案的最优就是每图给紧 cap c*，
  此时总花费 = Σmin(c*, nᵢ) ≤ B·N，**达不到预算上限**。
  我上一版误用了"每图给 c* 且花满预算"，那是**口径错误产生的假效应**。

**正确口径**（与第 5 轮一致）：
  · **均匀**：每图给紧 cap `c* = max{c : Σmin(c,nᵢ) ≤ B·N}`
  · **最优**：在**同一总预算** `B·N` 下针对目标逐图优化 `kᵢ ∈ [0, nᵢ]`
  · **分配效应 = 最优 − 均匀**（这是 H2′ 要的量）
""")

    for split in ("val", "test"):
        counts = load_counts(split)
        N = len(counts)
        print(f"\n{'#' * 96}\n### {split}：{N} 图 / {int(counts.sum())} 框 / "
              f"中位 {np.median(counts):.0f} / max {counts.max():.0f}")
        print(f"{'B':>5} {'紧cap c*':>8} │ {'M1 总召回':>22} │ {'M2 最坏图':>22} │ "
              f"{'M3 最坏十分位':>22}")
        print(f"{'':>5} {'':>8} │ {'均匀':>10} {'最优':>10} │ {'均匀':>10} {'最优':>10} │ "
              f"{'均匀':>10} {'最优':>10}")
        print("-" * 96)
        for B in (20, 30, 40, 50):
            budget = int(B * N)
            cstar = uniform_cap(counts, B)
            k_uni = uniform_alloc(counts, cstar)
            e_uni = evaluate(counts, k_uni)
            spend_uni = float(np.minimum(k_uni, counts).sum())
            print(f"    [核算] B={B:>3} 预算={budget:>7} c*={cstar:>4} "
                  f"均匀花费={spend_uni:>8.0f}（用掉 {spend_uni / budget * 100:5.1f}%）"
                  f"  微平均={e_uni['total_micro'] * 100:6.2f}%  宏平均={e_uni['total_macro'] * 100:6.2f}%")
            row = [f"{B:>5} {cstar:>8} │"]
            for obj, key in (("total_micro", "total_micro"), ("worst", "worst"),
                             ("worst_decile", "worst_decile")):
                e_opt = evaluate(counts, greedy_optimal(counts, budget, obj))
                row.append(f" {e_uni[key] * 100:>9.2f}% {e_opt[key] * 100:>9.2f}% │")
            print("".join(row))
        # 汇总"分配效应"（最优 − 均匀）在三个目标上的对比
        print(f"\n  ★ {split} 的「分配效应」= 最优 − 均匀（正 = 分配有用）")
        print(f"     {'B':>5} {'M1 总召回(微)':>14} {'M2 最坏图':>12} {'M3 最坏十分位':>14}")
        summary = {}
        for B in (20, 30, 40, 50):
            budget = int(B * N)
            cstar = uniform_cap(counts, B)
            e_uni = evaluate(counts, uniform_alloc(counts, cstar))
            line = [f"     {B:>5}"]
            for obj, key in (("total_micro", "total_micro"), ("worst", "worst"),
                             ("worst_decile", "worst_decile")):
                e_opt = evaluate(counts, greedy_optimal(counts, budget, obj))
                d = e_opt[key] - e_uni[key]
                line.append(f"{d * 100:>11.2f}pt")
                summary.setdefault(key, []).append(d)
            print("".join(line))
        print("     " + "-" * 52)
        print("     最大： " + "  ".join(
            f"{k}={max(v) * 100:>6.2f}pt" for k, v in summary.items()))

    print("\n" + "=" * 96)
    print("★ 判读")
    print("=" * 96)
    print("  · 若 M1（加法）的分配效应 ≈ 0，而 M2/M3（非加法）显著 > 0")
    print("    → **分配效应在非加法目标下重新出现** → H2′ 成立")
    print("    → 「分配」这条线可保留，但**必须把论文的目标函数从总召回换成尾部/最坏情况**。")
    print("  · 若三个目标的分配效应都 ≈ 0 → 分配在此问题下**整体无用**，应从框架中删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
