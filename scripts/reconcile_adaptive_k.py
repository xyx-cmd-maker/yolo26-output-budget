"""
★★ 第 24 轮的调和验证：**水位法（最坏情况最优）vs 逐单元自适应启发式**

═══════════════════════════════════════════════════════════════════════════
要调和的张力
═══════════════════════════════════════════════════════════════════════════
arXiv:2609.27953（CIKM'26）§7.2 实测：
> "Adaptive-K, per-user candidate sizing … **lifts Recall@K by 69–112% at 2–3× budget,
>  but leaves NDCG@10 indistinguishable from fixed K=100** (p ∈ {.17,.54,n/a})"

而我的水位法定理声称：在**固定总花费**下，`qᵢ = ρ*·nᵢ` 把 `min_i rᵢ` 最大化，
且相对均匀配额可提升 **40+ pt**。

**表面冲突**。三种可能的调和：
  (a) 他们优化 **NDCG（排序型）**，我优化 **worst recall（覆盖型）** → 目标不同
  (b) 他们的 Adaptive-K 是**启发式**（从 CF 分数分布定 K），**不是最坏情况最优**
  (c) 他们的 K 范围固定 `[50,500]`，我的 `qᵢ = ρ*·nᵢ` 可超出该范围

★★ 但还有一个我必须正视的可能：
  **(d) 我的水位法优势是"容量口径"的产物**——它假设"给多少配额就能检出多少"，
      即**排序完美**。真实系统里"给密集图更多配额"未必能换来更多真阳性。

═══════════════════════════════════════════════════════════════════════════
本脚本：把 (a)(b)(c) 用**受控算术**分开，并量化 (d) 的影响
═══════════════════════════════════════════════════════════════════════════
做法：构造一个"带排序缺陷"的模型——检测器在每图的命中率不是 100%，
而是随**配额/真实目标数之比**变化的函数：

    `hit_rate_i = f(qᵢ / nᵢ)`，`f` 单调不减、`f(1) = 1`，但有"注意力稀释"：
    `f(x) = x / (x + κ·(1 − x))`（κ > 1 时表示稀释；κ = 1 时退化为完美排序）

则 `TPᵢ = f(qᵢ/nᵢ) · min(qᵢ, nᵢ)`，花费仍是 `min(qᵢ, nᵢ)`。
→ 在**固定花费**下比较：均匀配额 vs 水位法（容量口径）vs 水位法（带稀释）

用法：python scripts/reconcile_adaptive_k.py
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


    def hit_rate(x: np.ndarray, kappa: float) -> np.ndarray:
        """`f(x) = x / (x + κ(1−x))`，x = qᵢ/nᵢ ∈ [0,1]。

        κ = 1 → f(x) = x（**完美排序**：配额全部转化为真阳性）
        κ > 1 → 稀释（配额越多，单位配额的真阳性率越低）—— 这是 CIKM 论文观测到的机制
        """
        x = np.clip(x, 0.0, 1.0)
        return x / (x + kappa * (1.0 - x) + 1e-12)


    def evaluate(n: np.ndarray, q: np.ndarray, kappa: float) -> dict:
        x = np.minimum(q, n) / n
        tp = hit_rate(x, kappa) * np.minimum(q, n)
        spend = float(np.minimum(q, n).sum())
        return {
            "spend": spend,
            "micro_recall": float(tp.sum() / n.sum()),          # TP 总量 / 目标总量
            "worst_recall": float((tp / n).min()),              # 最坏图的召回
            "macro_recall": float((tp / n).mean()),
        }


    def uniform_q(n: np.ndarray, S: float) -> np.ndarray:
        """均匀：每图给同一个 q（二分找 c 使 Σmin(c,nᵢ) = S）。"""
        lo, hi = 0.0, float(n.max())
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.minimum(mid, n).sum() <= S:
                lo = mid
            else:
                hi = mid
        return np.minimum(lo, n)


    def waterfill_q(n: np.ndarray, S: float) -> np.ndarray:
        """水位法（容量口径最优）：把每图 recall 抬到同一水平 ρ* = min(S/Σnᵢ, 1)。"""
        rho = min(S / n.sum(), 1.0)
        return np.minimum(rho * n, n)


    print("=" * 108)
    print("调和验证：水位法（最坏情况最优）在不同'稀释强度'下的表现")
    print("=" * 108)
    print(textwrap.dedent("""
    模型：`f(x) = x/(x + κ(1−x))`，`x = qᵢ/nᵢ`
      · κ = 1.0  → **完美排序**（配额全部变成真阳性）—— 我一直用的假设
      · κ > 1    → **注意力稀释**（CIKM 论文观测到的机制）
    """))

    for split in ("val", "test"):
        n = load_n(split)
        TOT = n.sum()
        print(f"\n{'#' * 108}\n### {split}  N={len(n)}  Σnᵢ={TOT:.0f}")
        print(f"{'κ':>6} {'S/TOT':>7} │ {'均匀 worst':>10} {'水位 worst':>10} {'Δ':>9} │ "
              f"{'均匀 micro':>11} {'水位 micro':>11} {'Δ':>9} │ 判定")
        print("-" * 108)
        for kappa in (1.0, 1.5, 3.0, 10.0, 100.0):
            for frac in (0.2, 0.5):
                S = frac * TOT
                qu = uniform_q(n, S)
                qw = waterfill_q(n, S)
                eu = evaluate(n, qu, kappa)
                ew = evaluate(n, qw, kappa)
                dw = (ew["worst_recall"] - eu["worst_recall"]) * 100
                dm = (ew["micro_recall"] - eu["micro_recall"]) * 100
                verdict = ("✅ 水位法仍显著更优" if dw > 2 else
                           ("⚠️ 优势消失" if abs(dw) <= 2 else "❌ 水位法更差"))
                print(f"{kappa:>6.1f} {frac:>7.2f} │ {eu['worst_recall'] * 100:>9.3f}% "
                      f"{ew['worst_recall'] * 100:>9.3f}% {dw:>+8.3f} │ "
                      f"{eu['micro_recall'] * 100:>10.3f}% {ew['micro_recall'] * 100:>10.3f}% "
                      f"{dm:>+8.3f} │ {verdict}")
            print("  " + "·" * 104)

    print("\n" + "=" * 108)
    print("★ 判读")
    print("=" * 108)
    print(textwrap.dedent("""  · 若 **κ = 1 时水位法大幅领先，κ 增大后优势快速消失** →
          说明我的"分配优势"**完全依赖完美排序假设**，与 CIKM 论文的实测（自适应预算无益）
          **不矛盾**，而是**我的假设过强**。→ **"分配"这条线必须降级或放弃。**
      · 若 **即使 κ 很大，水位法仍显著更优** →
          说明"最坏情况"目标与"平均指标"目标确实不同 → 我的定位（空间公平）成立。
    """))



if __name__ == "__main__":
    raise SystemExit(main())
