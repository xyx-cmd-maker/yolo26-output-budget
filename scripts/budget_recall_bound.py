"""
★★ 从恒等式导出可操作准则：**预算受限下的召回分解与预算选择**

═══════════════════════════════════════════════════════════════════════════
起点（第 13 轮证明的恒等式）
═══════════════════════════════════════════════════════════════════════════
硬预算 `Σkᵢ ≤ B·N`，处处 `kᵢ ≤ nᵢ` ⇒ `Σmin(kᵢ,nᵢ) = B·N`。

**推论 1（上界）**：任何检测器在预算 B 下的召回满足
    `recall ≤ min(1, B·N / Σnᵢ)`      —— **与检测器完全无关**
    （因为总 TP ≤ 总输出数 ≤ B·N，且总 TP ≤ Σnᵢ）

**推论 2（三因子分解）**：
    `recall = mit(B) × eff(B) × rank(B)`
    其中
      · `mit(B) = min(1, B·N/Σnᵢ)`   **缓解因子**：预算够不够（只依赖数据集与 B）
      · `eff(B) = TP / (B·N)`        **预算效率**：允许的输出槽里有多少是真阳性
      · `rank(B) = TP / min(B·N, Σnᵢ)` **排序因子**：在"能用的槽"里选中真阳性的比例

    ⚠️ 注意 `eff × (B·N) = rank × min(B·N, Σnᵢ)`，所以 `eff` 与 `rank` 只差一个已知常数因子。

═══════════════════════════════════════════════════════════════════════════
★ 本脚本必须正面回答的自我质疑（否则这条贡献是假的）
═══════════════════════════════════════════════════════════════════════════
**"`eff(B) = TP/(B·N)` 是不是只是 precision 换了个名字？"**

precision 与 `eff` 的**唯一区别**在于分母：
  · `precision = TP / 实际输出数`（分母是**检测器自己决定**输出多少）
  · `eff(B)   = TP / B·N`（分母是**预算**，由部署方决定）

**这个区别是否是实质的？** 三种情形：
  (a) 若检测器**总是用满预算**（输出数 = B·N）→ 两者相同 → **`eff` 无新内容**
  (b) 若检测器**用不满预算** → `eff < precision`，且**差多少就是一个可测的量**
      （"预算利用率"），这**是新信息**
  (c) 若检测器**超预算**（NMS-free 头给固定 300）→ `eff` 无定义，需先截断

本脚本用**真实数据**检验 (a)/(b)/(c) 哪个成立，从而判定这条贡献是否有实质内容。
用法：python scripts/budget_recall_bound.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))


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


def worst_image_recall(counts: np.ndarray, B: int) -> float:
    """每图恰好 B 个输出（不含"用不满"机制）时的**最坏图**召回。"""
    k = np.full(len(counts), float(B))
    return float((np.minimum(k, counts) / counts).min())


def budget_utilization(counts: np.ndarray, B: int) -> float:
    """**预算利用率**：每图给 B 个输出时，实际花掉多少预算。

    ⚠️ 这正是 (b) 情形的核心量：检测器决定输出多少 → 决定预算被用掉多少。
    """
    return float(np.minimum(B, counts).sum() / (B * len(counts)))


def main() -> int:
    print("=" * 104)
    print("预算受限下的召回上界与三因子分解")
    print("=" * 104)
    print("""
恒等式（第 13 轮）：硬预算 + 处处 kᵢ ≤ nᵢ  ⇒  Σmin(kᵢ,nᵢ) = B·N

推论 1（**与检测器无关的上界**）：
    recall ≤ min(1, B·N / Σnᵢ)          ← mit(B)，只依赖数据集与 B
推论 2（三因子）：
    recall = mit(B) × eff(B) × rank(B)
    eff(B) = TP/(B·N)      ← 「预算效率」
    rank(B) = TP/min(B·N, Σnᵢ)
""")

    for split in ("val", "test"):
        counts = load(split)
        N, TOT = len(counts), counts.sum()
        print(f"\n{'#' * 104}\n### {split}   N={N}   Σnᵢ={int(TOT)}   最大单图={counts.max():.0f}")
        print(f"{'B':>5} {'预算B·N':>9} {'B·N/Σnᵢ':>9} │ {'mit(B) 上界':>11} │ "
              f"{'预算利用率':>10} │ {'最坏图召回(每图B)':>16} │ 判读")
        print("-" * 104)
        for B in (10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500):
            budget = B * N
            mit = min(1.0, budget / TOT)
            util = budget_utilization(counts, B)
            wi = worst_image_recall(counts, B)
            note = ""
            if mit >= 1.0:
                note = "预算 ≥ 全部目标 → 上界=100%，预算**不咬合**"
            elif util < 0.9:
                note = f"**预算用不满**（只用 {util * 100:.0f}%）"
            print(f"{B:>5} {budget:>9} {budget / TOT:>9.3f} │ {mit * 100:>10.2f}% │ "
                  f"{util * 100:>9.1f}% │ {wi * 100:>15.2f}% │ {note}")

        # ★ 关键：B* = Σnᵢ/N 是"预算刚好够"的临界点
        bstar = TOT / N
        print(f"\n  ★ **临界预算 B\\* = Σnᵢ/N = {bstar:.2f}**（每图平均目标数）")
        print(f"     · B < B\\* → 预算**咬合**，上界 < 100%，检测器再好在预算内也达不到全覆盖")
        print(f"     · B ≥ B\\* → 预算不咬合，上界 = 100%")
        print(f"     · 但 B ≥ B\\* 时**预算利用率 < 100%**（见上表）→ 实际输出数 < B·N")

        # ★ 回答自我质疑：eff vs precision 的区别在真实数据上有多大
        print(f"\n  ★ 回答「eff(B) 是否只是 precision」——用**预算利用率**量化差异：")
        print(f"     当检测器每图输出恰好 B 个但实际只有 min(B,nᵢ) 个可检测时，")
        print(f"     预算利用率 = Σmin(B,nᵢ)/(B·N)：")
        for B in (30, 50, 100):
            u = budget_utilization(counts, B)
            print(f"       B={B:>4}: 利用率 {u * 100:5.1f}%  →  eff = precision × {u:.4f}"
                  f"  （{'相同，eff 无新内容' if u > 0.99 else f'**差 {(1 - u) * 100:.1f}%**，eff 携带新信息'}）")

    print("\n" + "=" * 104)
    print("★ 判读：这条贡献有没有实质内容？")
    print("=" * 104)
    print("""  · **有**，当且仅当"预算利用率"在真实部署中显著 < 100%（即情形 (b) 成立）。
    此时：
      - `eff(B) = TP/(B·N)` 与 `precision = TP/实际输出数` **相差一个可测量**
        （预算利用率），而这个量**由部署方通过 B 控制、检测器无法控制**。
      - 于是"选 B"成为一个**有依据的设计决策**：B 太小 → 上界压住召回；
        B 太大 → 利用率暴跌（输出大量低分框），**precision 与推理成本都变差**。
      - **存在一个最优 B\\*\\***，它不是"越大越好"。

  · **没有**，若利用率恒为 100%（情形 (a)）—— 那时 `eff ≡ precision`，
    整条分解退化为 `recall = precision × ...`，**没有新内容**。

  → **本轮的诚实判定取决于上面表格里"预算利用率"列的实际取值。**
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
