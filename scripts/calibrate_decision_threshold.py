"""
★ P30 纪律：**先算判据的可达范围**（判据 v2 的阈值校准）

═══════════════════════════════════════════════════════════════════════════
背景
═══════════════════════════════════════════════════════════════════════════
判据 v2（`docs/prereg-decision-rule-v2-absolute.md`）用 `Δworst_abs ≥ 10 pt` 判 A。
但手工推导给出上界：

    均匀下最坏图（最密，n = n_max）的配额份额 ≈ c/n_max，其中 c ≈ S/N = ρ·mean(n)
    水位下该图份额 → 1
    ⇒ **Δworst_abs ≤ hit_rate · (1 − ρ·mean(n)/n_max)**

**若该上界 < 10 pt，则判据 A 永远不可达** —— 这会是 P30 的第三次。
同理需要检查 B 的 5 pt 阈值。

═══════════════════════════════════════════════════════════════════════════
本脚本
═══════════════════════════════════════════════════════════════════════════
在**真实 VisDrone 标签分布** + **一族假设的检出率模型**下，
算出 `Δworst_abs` 的**可达上确界**，从而判断 10 / 5 pt 是否可达。

检出率模型（**不假设具体检测器**，只扫参数空间）：
    hitᵢ = clip(A - B·log10(nᵢ), 0, 1)
    —— 稀疏图（n 小）检出率高、密集图检出率低，符合检测器的一般规律
    A ∈ [0.3, 1.0]，B ∈ [0.0, 0.5] 网格扫描
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))


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


def hit_curve(n: np.ndarray, A: float, B: float) -> np.ndarray:
    return np.clip(A - B * np.log10(np.maximum(n, 1.0)), 0.0, 1.0)


def tp_of(n: np.ndarray, q: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """TPᵢ(q) = min(qᵢ, nᵢ) · hitᵢ（配额内可检出的比例恒定）。"""
    return np.minimum(q, n) * hit


def worst(n: np.ndarray, tp: np.ndarray) -> float:
    m = n > 0
    return float((tp[m] / n[m]).min())


def plan_uniform(n: np.ndarray, S: float) -> np.ndarray:
    lo, hi = 0.0, float(n.max())
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(mid, n).sum() <= S:
            lo = mid
        else:
            hi = mid
    return np.minimum(lo, n)


def plan_waterfill(n: np.ndarray, rho: float) -> np.ndarray:
    return np.minimum(rho * n, n)


def main() -> int:
    print("=" * 104)
    print("判据 v2 的阈值校准：Δworst_abs 的可达上确界（真实 VisDrone 标签分布）")
    print("=" * 104)

    for split in ("val", "test"):
        n = load_n(split)
        TOT, N = n.sum(), len(n)
        rho = 0.5
        S = rho * TOT
        c = None
        qu = plan_uniform(n, S)
        c = float(qu.max())  # 均匀配额（所有未饱和图的配额相同）

        print(f"\n{'#' * 104}\n### {split}   N={N}  Σnᵢ={TOT:.0f}  mean={n.mean():.2f}  "
              f"max={n.max():.0f}  median={np.median(n):.0f}")
        print(f"    ρ={rho}  S={S:.0f}  均匀配额 c={c:.2f}  "
              f"解析上界 (1 − ρ·mean/max) = {1 - rho * n.mean() / n.max():.4f}")

        best = (-1e9, None)
        rows = []
        for A in np.arange(0.30, 1.001, 0.05):
            for B in np.arange(0.0, 0.501, 0.05):
                hit = hit_curve(n, A, B)
                if hit.max() <= 0:
                    continue
                wu = worst(n, tp_of(n, qu, hit))
                ww = worst(n, tp_of(n, plan_waterfill(n, rho), hit))
                d = (ww - wu) * 100
                rows.append((d, A, B, wu, ww))
                if d > best[0]:
                    best = (d, (A, B, wu, ww))
        rows.sort(reverse=True)

        print(f"\n    {'Δworst_abs(pt)':>15} {'A':>6} {'B':>6} │ {'worst 均匀':>11} {'worst 水位':>11}")
        print("    " + "-" * 66)
        for d, A, B, wu, ww in rows[:8]:
            print(f"    {d:>15.3f} {A:>6.2f} {B:>6.2f} │ {wu * 100:>10.3f}% {ww * 100:>10.3f}%")
        print(f"    …（共 {len(rows)} 个参数组合）")
        # 最差的一档也要看（确认方向恒正）
        dmin = rows[-1]
        print(f"    {'最差':>15}: {dmin[0]:>7.3f} pt  (A={dmin[1]:.2f}, B={dmin[2]:.2f})")

        print(f"\n    ★ **Δworst_abs 的可达范围 = [{dmin[0]:.3f}, {best[0]:.3f}] pt**")
        print(f"    ★ 判据阈值：A ≥ 10 pt → {'✅ 可达' if best[0] >= 10 else '❌ **不可达**'}"
              f"；B ≥ 5 pt → {'✅ 可达' if best[0] >= 5 else '❌ **不可达**'}")

    print("\n" + "=" * 104)
    print("★ 结论")
    print("=" * 104)
    print("""  · 若 A 阈值（10 pt）不可达 → **必须下调**，并说明理由与新的可达依据（P30 纪律）。
  · 若 B 阈值（5 pt）也不可达 → 判据 v2 同样失去判别力，需要改用**相对量**：
        实现率 = Δworst_abs / 该参数下的解析上界  hit·(1 − ρ·mean/max)
    该相对量天然落在 [0,1]，且阈值可预注册为 0.5（收回一半可达空间）。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
