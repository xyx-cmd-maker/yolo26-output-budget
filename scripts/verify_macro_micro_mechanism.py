"""
★★ 验证机制：**"余量给谁"只影响 macro，不影响 micro**

═══════════════════════════════════════════════════════════════════════════
要检验的机制论断（第 17 轮 docs/finding-budget-semantics-and-macro-micro.md §2.3）
═══════════════════════════════════════════════════════════════════════════
在硬预算 `Σkᵢ ≤ B·N` 且处处 `kᵢ ≤ nᵢ` 下，恒等式给出 `Σmin(kᵢ,nᵢ) = Σkᵢ`。
所以**只要用满预算，micro（总召回）就相同** —— 与"余量给谁"无关。

而 macro（逐图平均召回 `mean(min(kᵢ,nᵢ)/nᵢ)`）**取决于余量给谁**：
  · 给**最密**的图 → 它们的 `kᵢ/nᵢ` 仍小 → macro 升得慢
  · 给**最疏**的图 → 它们接近 100% → macro 升得快

**若这条成立**，则必须能构造出 **micro 完全相同、macro 显著不同**的一对方案。
构造不出来 → 机制论断错，必须撤回。

═══════════════════════════════════════════════════════════════════════════
对照的四种"余量分配"（预算与处处 kᵢ ≤ nᵢ 全部相同）
═══════════════════════════════════════════════════════════════════════════
  D1 给最密的图（按 nᵢ 降序）
  D2 给最疏的图（按 nᵢ 升序）
  D3 随机
  D4 按 nᵢ 比例（nᵢ 大的多给）—— 另一种自然的启发式

用法：python scripts/verify_macro_micro_mechanism.py
"""
from __future__ import annotations
import textwrap

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))

RNG = np.random.default_rng(20261014)




def main() -> int:
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


    def macro_image(counts, k):
        """逐图召回的平均（per-image macro）"""
        return float((np.minimum(k, counts) / counts).mean())


    def worst(counts, k):
        return float((np.minimum(k, counts) / counts).min())


    def alloc_with_surplus(counts: np.ndarray, B: int, order: np.ndarray) -> np.ndarray:
        """先给每图 `min(B, nᵢ)`，再把**用不完的预算**按 `order` 顺序补出去。

        ⚠️ 关键：处处 `kᵢ ≤ nᵢ`，且总额**正好**是 `Σmin(B,nᵢ)`（= 预算内能用的最大额）。
        这就是"用满可用预算"的所有方案族 —— 它们的 micro 必然相同。
        """
        k = np.minimum(float(B), counts).copy()
        # 上一步已把每个能用的槽都用满了 → Σk = Σmin(B,nᵢ)，micro 已达该 B 的上限。
        # "余量给谁"只可能发生在**还不能用满**的时候；
        # 这里改为：从每图抽掉一部分（保持总额不变），再按 order 加回去 —— 从而在同总额下
        # 改变**分布**，检验 macro 是否随之变化。
        return k


    def alloc_permuted(counts: np.ndarray, B: int, order: np.ndarray, shift_frac: float) -> np.ndarray:
        """在同总额 `Σmin(B,nᵢ)` 下**重分布**配额。

        做法：从"配额充足"的图（`nᵢ ≥ B`，即 `min(B,nᵢ)=B`）里抽出一部分额度，
        按 `order` 重新分配（受 `kᵢ ≤ nᵢ` 约束，并把抽出的量精确补回）。
        → 总额不变 → micro 不变（恒等式）→ 只检验 macro。
        """
        N = len(counts)
        k = np.minimum(float(B), counts).copy()
        total = k.sum()
        rich = np.where(counts >= B)[0]          # 这些图的 min(B,nᵢ)=B，抽走即减少其 min
        if len(rich) == 0:
            return k
        pool = int(shift_frac * B * len(rich))
        if pool <= 0:
            return k
        # 按 order 给 rich 图排序，从**靠后**的抽，给**靠前**的
        rank = {int(i): r for r, i in enumerate(order) if i in set(rich.tolist())}
        rich_sorted = sorted(rich.tolist(), key=lambda i: rank.get(i, 10**9))
        taken = 0
        for i in rich_sorted[::-1]:               # 从不喜欢的一端抽
            if taken >= pool:
                break
            take = min(k[i], pool - taken)
            k[i] -= take
            taken += take
        # 补回：按 order 从喜欢的一端加（受 nᵢ 约束）
        given = 0
        for i in order:
            if given >= taken:
                break
            room = counts[i] - k[i]
            if room > 0:
                add = min(room, taken - given)
                k[i] += add
                given += add
        assert abs(k.sum() - total) < 1e-6, f"总额未守恒：{k.sum()} vs {total}"
        assert (k <= counts + 1e-9).all(), "超过真实需求"
        return k


    print("=" * 104)
    print("机制验证：在**同总额**下重分配配额，micro 是否不变、macro 是否变化？")
    print("=" * 104)

    for split in ("val", "test"):
        counts = load(split)
        N, TOT = len(counts), counts.sum()
        print(f"\n{'#' * 104}\n### {split}  N={N}  Σnᵢ={int(TOT)}  B*={TOT / N:.2f}")
        print(f"{'B':>4} │ {'余量给谁':>14} {'Σk':>8} {'micro':>9} {'Δmicro':>9} │ "
              f"{'macro(逐图)':>11} {'Δmacro':>9} │ {'worst':>8} {'Δworst':>9}")
        print("-" * 104)

        for B in (30, 50, 100):
            base = alloc_with_surplus(counts, B, np.arange(N))
            m0, a0, w0 = micro(counts, base), macro_image(counts, base), worst(counts, base)
            variants = [
                ("给最密（nᵢ↓）", np.argsort(-counts, kind="mergesort")),
                ("给最疏（nᵢ↑）", np.argsort(counts, kind="mergesort")),
                ("随机", RNG.permutation(N)),
            ]
            for name, order in variants:
                k = alloc_permuted(counts, B, order, shift_frac=0.5)
                m, a, w = micro(counts, k), macro_image(counts, k), worst(counts, k)
                print(f"{B:>4} │ {name:>14} {k.sum():>8.0f} {m * 100:>8.3f}% {(m - m0) * 100:>+8.4f} │ "
                      f"{a * 100:>10.3f}% {(a - a0) * 100:>+8.3f} │ {w * 100:>7.3f}% {(w - w0) * 100:>+8.3f}")
            print(f"     └ 基线（不重分布）：micro={m0 * 100:.3f}%  macro={a0 * 100:.3f}%  worst={w0 * 100:.3f}%")
            print("     " + "·" * 100)

    print("\n" + "=" * 104)
    print("★ 判读")
    print("=" * 104)
    print(textwrap.dedent("""  · **Δmicro 全部 ≈ 0** → 恒等式成立（同总额 ⇒ 同 micro），机制论断第 1 半成立
      · **Δmacro 或 Δworst 显著 ≠ 0** → "余量给谁"确实改变 macro，机制论断第 2 半成立
      · 若 **Δmicro 与 Δmacro 都 ≈ 0** → 机制论断错，必须撤回
    """))



if __name__ == "__main__":
    raise SystemExit(main())
