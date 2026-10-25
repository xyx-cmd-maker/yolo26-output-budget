"""
纠正一个我差点犯的严重错误：cap 与总预算的关系

═══════════════════════════════════════════════════════════════════════════
我上一版 `k3_rule_check.py` 的致命缺陷
═══════════════════════════════════════════════════════════════════════════
它拿 `ceiling_uniform(counts, c)` 当"用 cap=c 时的召回"，得出
`cap = 300` 相对 `cap = B` 有 **5911%** 的增益。

**这些数字无意义** —— 因为 `ceiling_uniform(counts, c) = Σmin(c,nᵢ)/Σnᵢ`
在 `c > B` 时**已经超出预算 `B·N`**：它描述的是"**每张图都允许输出 c 个**"，
即**总预算被放大成 c·N**，而不是 `B·N`。

**所以"把逐图上限设大"不是免费的** —— 它是超预算的多输出。
我差点把它当成"无需标定、c 无关的部署修正"，那是错的。

═══════════════════════════════════════════════════════════════════════════
正确的图景
═══════════════════════════════════════════════════════════════════════════
  · **均匀方案**：每图给 `c` 个 → 总花费 `Σmin(c,nᵢ)`，**必须 ≤ B·N**。
    → 可行的最大均匀 `c` **就是** `c* = max{c : Σmin(c,nᵢ) ≤ B·N}`。
    → **所以 `c*` 是均匀方案能达到的最好结果，不存在"更大的 cap 更好"**。

  · **按图分配方案**：总预算固定 `B·N`，逐图 `kᵢ` 可不同（**可超过 c***，只要总量不超）。
    这才是"分配"起作用的地方。

→ 因此本脚本只做一件正确的事：
     **画出 `Σmin(c,nᵢ)/Σnᵢ` 与 `Σmin(c,nᵢ)/(B·N)`（预算利用率）随 c 的曲线**，
     标出 `c*` 的位置，并明确"超过 c* 之后的任何数字都对应超预算"。
"""
from __future__ import annotations
import textwrap

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import uniform_cap  # noqa: E402




def main() -> int:
    def load(split):
        cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
        d = Path(cfg["path"]) / cfg[split]
        p = list(d.parts)
        for i in range(len(p) - 1, -1, -1):
            if p[i] == "images":
                p[i] = "labels"
                break
        return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                         for f in sorted(Path(*p).glob("*.txt"))], dtype=float)


    print("=" * 96)
    print("纠正后的图景：均匀方案的可达召回 vs 逐图配额上限 c（并标明预算边界）")
    print("=" * 96)

    for split in ("val", "test"):
        counts = load(split)
        N, TOT, MX = len(counts), counts.sum(), counts.max()
        print(f"\n{'#' * 96}\n### {split}：N={N}  Σnᵢ={int(TOT)}  max nᵢ={MX:.0f}")
        print(f"{'B':>5} {'预算B·N':>9} {'c*':>5} {'c*/B':>6} │ "
              f"{'c*可达召回':>11} {'均匀(B)':>9} {'真实增益':>9} │ {'c 继续增大时':>26}")
        print("-" * 96)
        for B in (20, 30, 40, 50, 75, 100, 150, 200, 300):
            budget = B * N
            cstar = uniform_cap(counts, B)
            v_cap = float(np.minimum(cstar, counts).sum() / TOT)
            v_base = float(np.minimum(B, counts).sum() / TOT)
            # 超过 c* 之后：Σmin 会超预算 → 标明
            over = int(np.minimum(min(cstar + 1, int(MX)), counts).sum())
            tag = (f"c={cstar+1} 时花费 {over} > 预算 {budget}"
                   if over > budget else "已达上限（c* = max nᵢ）")
            print(f"{B:>5} {budget:>9} {cstar:>5} {cstar / B:>5.2f}× │ "
                  f"{v_cap * 100:>10.2f}% {v_base * 100:>8.2f}% {(v_cap - v_base) * 100:>8.2f}pt │ {tag:>26}")

    print("\n" + "=" * 96)
    print("★ 结论（纠正后）")
    print("=" * 96)
    print(textwrap.dedent("""  1. **`c*` 已经是"均匀方案"的最优**：更大的 cap 会让 `Σmin(c,nᵢ) > B·N`，
         即**超预算**。所以不存在"把 cap 设大更好"这条规则 —— 我上一版差点这么写。

      2. **K3 的正确表述**：
         「把逐图配额上限从 `B` 换成紧 cap `c*`（= 在预算约束下可行的最大均匀上限）
           可多拿 10.87–27.53 pt。`c*` 由数据集密度分布决定，**不等于 `B`**，
           在 test 上 `c*/B` 可达 **1.10–9.24×**。」

      3. **`c*` 的标定性结论（第 12 轮 W1–W3 实测）**：
         · 从 **val** 标定后施加到 **test**，在关键档位（B=40/50）只拿到上界增益的 **24–26%**
           → **跨切分不可直接迁移**（两切分均值比 0.660、中位比 0.554）。
         · 用**目标域自身**少量图（m=50）估分布、再反解 `c*`，在 B=30 拿到 **90%**
           → **需要在目标域上标定**，不能借用源域。
         · 松预算档（B≥50）的 `c*` 由**尾部**（max nᵢ / 高分位）决定，
           均值/中位这类摘要**携带不了**该信息 → 简单凭均值估 `c*` 会失败。

      4. **因此 K3 不是"一个 c 无关的免费修正"，而是"一个需要在目标域上标定的标量"。**
         这比第 5 轮的表述弱，但比"oracle"强 —— 它是**可实现的**，只是有标定成本。
    """))



if __name__ == "__main__":
    raise SystemExit(main())
