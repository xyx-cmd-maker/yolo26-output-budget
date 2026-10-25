"""
★★ 决定先定：**在判据出现之前**写下决策规则（防事后合理化）

═══════════════════════════════════════════════════════════════════════════
要判定的问题
═══════════════════════════════════════════════════════════════════════════
第 27 轮把"水位法定理"降级为 **经典 max-min 问题的检测实例化**。
红队 R1 质问：**"翻译 + 实例化 + 量化"够不够一篇期刊论文？**

**这个问题的答案取决于一个尚未实测的量 κ**：
  · κ 小（水位法的 worst 增益在真实检测器上仍 ≈ +20 pt）→ 有真实可操作的发现 → 可能够
  · κ 大（增益 ≈ 0）→ 只剩"已知问题的翻译" → **不够，须并入主线**

→ **必须在看到 κ 之前**写下判据，否则我会事后挑一个"好看的"阈值。

═══════════════════════════════════════════════════════════════════════════
★ 本轮同时修正一个我此前的推理错误
═══════════════════════════════════════════════════════════════════════════
第 24 轮的 κ 扫描用的是**松预算**（`S/TOT` = 0.2 / 0.5，配合 `hit_rate(x)=x/(x+κ(1−x))`）。
我当时断言"κ 增大后水位法优势消失"。

**但紧预算下稀释可能无法发生**：若处处 `qᵢ ≤ nᵢ`（紧预算的典型情形），
则 `x = qᵢ/nᵢ ≤ 1`，而**"稀释"的机制是"给了远超所需的配额"**——
在紧预算下**根本没有可稀释的余量**（每张图的配额都被真需求卡住）。

**→ 必须在**两种预算制度**下都扫 κ，才能给出可外推的结论。**
本脚本做这件事。

═══════════════════════════════════════════════════════════════════════════
★ 预先登记的判据（现在写下，不得事后修改）
═══════════════════════════════════════════════════════════════════════════
设 `Δworst(κ*)` 为**真实检测器**上、**紧预算**（S/TOT ≤ 0.5）下，
水位法相对均匀分配的**最坏图召回增益**（单位：pt）。

  判据 A：`Δworst ≥ 10 pt`  → **分配线成立**，作为独立贡献推进
  判据 B：`2 ≤ Δworst < 10 pt` → **降级为"稳健性观察"**，并入 DHU 主线的一节
  判据 C：`Δworst < 2 pt`  → **放弃分配线**，全部算力转 DHU / R-Scale

**注**：真实检测器无法直接给 `κ`（它是假设的函数形式），
故判据作用在**可直接测量**的 `Δworst` 上——这更稳健（不依赖我的 κ 参数化）。
"""
from __future__ import annotations

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


    def hit(x, kappa):
        x = np.clip(x, 0.0, 1.0)
        return x / (x + kappa * (1.0 - x) + 1e-12)


    def ev(n, q, kappa):
        x = np.minimum(q, n) / n
        tp = hit(x, kappa) * np.minimum(q, n)
        return float((tp / n).min()), float(tp.sum() / n.sum()), float(np.minimum(q, n).sum())


    def uni(n, S):
        lo, hi = 0.0, float(n.max())
        for _ in range(100):
            m = (lo + hi) / 2
            if np.minimum(m, n).sum() <= S:
                lo = m
            else:
                hi = m
        return np.minimum(lo, n)


    def wf(n, S):
        r = min(S / n.sum(), 1.0)
        return np.minimum(r * n, n)


    print("=" * 104)
    print("★ 两种预算制度 × 稀释强度：水位法的增益在紧预算下是否仍存在？")
    print("=" * 104)
    print("""
    `hit_rate(x) = x/(x+κ(1−x))`，x = qᵢ/nᵢ
      · κ=1 完美排序；κ 越大越"稀释"
      · **紧预算**（S/TOT=0.05/0.1/0.2）：处处 qᵢ ≤ nᵢ 的图占多数 → 稀释**难以发生**
      · **松预算**（S/TOT=0.5/0.8）：有大量"超需求"配额 → 稀释**容易发生**
    """)

    for split in ("val", "test"):
        n = load_n(split)
        TOT = n.sum()
        print(f"\n{'#' * 104}\n### {split}  N={len(n)}  Σnᵢ={TOT:.0f}")
        print(f"{'制度':>8} {'S/TOT':>7} │ {'κ':>6} │ {'均匀worst':>10} {'水位worst':>10} {'Δworst':>9} │ "
              f"{'均匀micro':>10} {'水位micro':>10} {'Δmicro':>9}")
        print("-" * 104)
        for regime, fracs in (("紧", (0.05, 0.1, 0.2)), ("松", (0.5, 0.8))):
            for frac in fracs:
                S = frac * TOT
                qu, qw = uni(n, S), wf(n, S)
                # 紧预算的关键统计：多少比例的图 qᵢ ≥ nᵢ（即"无余量可稀释"）
                sat = float((qw >= n - 1e-9).mean())
                for kappa in (1.0, 3.0, 10.0, 100.0):
                    wu, mu, _ = ev(n, qu, kappa)
                    ww, mw, _ = ev(n, qw, kappa)
                    print(f"{regime:>8} {frac:>7.2f} │ {kappa:>6.1f} │ {wu * 100:>9.3f}% {ww * 100:>9.3f}% "
                          f"{(ww - wu) * 100:>+8.3f} │ {mu * 100:>9.3f}% {mw * 100:>9.3f}% "
                          f"{(mw - mu) * 100:>+8.3f}")
                print(f"         └ 该档水位法下 **已饱和（qᵢ≥nᵢ）的图占比 {sat * 100:.1f}%**"
                      f"  ← 无余量可稀释的比例")
                print("  " + "·" * 100)

    print("\n" + "=" * 104)
    print("★ 判读要点")
    print("=" * 104)
    print("""  · 若**紧预算**下 Δworst 不随 κ 衰减（而松预算下衰减）→
          说明"稀释"只在有超需求配额时才发生 → **判据应落在紧预算制度**（这是我的实际场景）
      · 若两种制度下都衰减 → 分配线的收益普遍脆弱
    """)



if __name__ == "__main__":
    raise SystemExit(main())
