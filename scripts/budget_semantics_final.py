"""
★★ 预算语义的最终裁定 + macro/micro 双口径对照

═══════════════════════════════════════════════════════════════════════════
要解决的两个悬而未决的问题
═══════════════════════════════════════════════════════════════════════════
**Q1（P24 的收尾）**：`uniform(c*)`（每图给 c*）与 `uniform(B)`（每图给 B）到底
  哪一个在**硬预算** `Σkᵢ ≤ B·N` 下合法？

  · 硬预算要求 `Σkᵢ ≤ B·N`。`uniform(c*)` 的 `Σkᵢ = N·c* > B·N`（实测 1.05–9.24×）→ **非法**。
  · `uniform(B)` 的 `Σkᵢ = N·B` = 预算 → **合法**（但可能花不完）。
  · **所以硬预算下的合法均匀方案只有 `uniform(B)`**，而"紧 cap c*"是**软预算**下的概念。

**Q2（红队 R2）**：我的"两区间"定位（`B≫B*` 跨类排序 / `B≲B*` 容量硬约束）
  是否只是**宏平均 vs 微平均**的差异？

  → 本脚本**同时**输出：
      · **micro**（`Σmin(kᵢ,nᵢ)/Σnᵢ`，每框等权）—— "总召回"的定义
      · **macro**（`mean(min(kᵢ,nᵢ)/nᵢ)`，每图等权）—— AP 的同类口径
    并在**同一口径内**比较不同分配策略。

═══════════════════════════════════════════════════════════════════════════
三个参照方案
═══════════════════════════════════════════════════════════════════════════
  A `uniform(B)`        ：每图给 B  → Σk = B·N（硬预算合法）
  B `uniform_soft(c*)`  ：每图给 c* → Σk = N·c*（**仅软预算合法**）
  C `sat_hard(B)`       ：硬预算下**用满预算**的分配（`Σkᵢ = B·N` 且处处 `kᵢ ≤ nᵢ`）——
                          这才是硬预算下的**真最优**（恒等式保证用满即最优）

用法：python scripts/budget_semantics_final.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import uniform_cap  # noqa: E402


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


def micro(counts, k):
    return float(np.minimum(k, counts).sum() / counts.sum())


def macro(counts, k):
    return float((np.minimum(k, counts) / counts).mean())


def worst(counts, k):
    return float((np.minimum(k, counts) / counts).min())


def alloc_uniform(counts, c):
    return np.full(len(counts), float(c))


def alloc_sat_hard(counts, B):
    """硬预算 `Σkᵢ ≤ B·N` 下**用满预算**的分配（处处 kᵢ ≤ nᵢ）。

    恒等式保证：只要用满预算且处处 kᵢ ≤ nᵢ，`Σmin(kᵢ,nᵢ) = B·N` —— **与分配细节无关**。
    这里用最朴素的做法：先均匀摊 B，再把余量给最密的图。
    """
    N = len(counts)
    budget = int(round(B * N))
    k = np.minimum(np.floor(budget / N), counts).astype(float)
    left = budget - k.sum()
    for i in np.argsort(-counts):
        if left <= 0:
            break
        room = counts[i] - k[i]
        if room > 0:
            take = min(room, left)
            k[i] += take
            left -= take
    assert k.sum() <= budget + 1e-6, f"超预算 {k.sum()} > {budget}"
    assert (k <= counts + 1e-9).all()
    return k, k.sum() / budget


def main() -> int:
    print("=" * 108)
    print("预算语义最终裁定 + macro/micro 双口径对照")
    print("=" * 108)

    for split in ("val", "test"):
        counts = load(split)
        N, TOT = len(counts), counts.sum()
        Bstar = TOT / N
        print(f"\n{'#' * 108}\n### {split}   N={N}  Σnᵢ={int(TOT)}  "
              f"**B\\* = Σnᵢ/N = {Bstar:.2f}**  max nᵢ={counts.max():.0f}")
        print(f"{'B':>4} │ {'方案':>16} {'Σk':>9} {'Σk/预算':>9} {'花费':>9} {'花/预算':>8} │ "
              f"{'micro':>8} {'macro':>8} {'worst':>8} │ 合法")
        print("-" * 108)
        for B in (10, 20, 30, 40, 50, 75, 100, 150, 300):
            budget = B * N
            cstar = uniform_cap(counts, B)
            plans = [
                ("uniform(B)", alloc_uniform(counts, B)),
                (f"uniform(c*={cstar})", alloc_uniform(counts, cstar)),
                ("sat_hard(B)", alloc_sat_hard(counts, B)[0]),
            ]
            for name, k in plans:
                sk = float(k.sum())
                used = float(np.minimum(k, counts).sum())
                ok_hard = sk <= budget + 1e-6
                tag = ("✅ 硬+软" if ok_hard else "❌ 硬 / ✅ 软")
                print(f"{B:>4} │ {name:>16} {sk:>9.0f} {sk / budget:>8.2f}× {used:>9.0f} "
                      f"{used / budget * 100:>7.1f}% │ {micro(counts, k) * 100:>7.2f}% "
                      f"{macro(counts, k) * 100:>7.2f}% {worst(counts, k) * 100:>7.2f}% │ {tag}")
            print("     " + "·" * 100)

    print("\n" + "=" * 108)
    print("★ 裁定")
    print("=" * 108)
    print("""  **Q1 的答案**：
    硬预算 `Σkᵢ ≤ B·N` 下**唯一合法的均匀方案是 `uniform(B)`**；
    `uniform(c*)` 的 `Σkᵢ = N·c* > B·N`（1.05–9.24×）→ **只在软预算下合法**。
    → 所以此前的"cap 效应 10.87–27.53 pt"必须重新标注为
      「**软预算（允许逐图配额超过平均值）下的收益**」。

  **Q2 的答案**：看同一 B 下不同方案的 **micro vs macro** 差异模式：
    · 若**两个口径都变化** → "两区间"不是 macro/micro 的假象 → 定位成立
    · 若**只有 macro 变化、micro 不变** → 定位塌缩为"口径差异"
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
