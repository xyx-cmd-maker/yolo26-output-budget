"""
P0-2 最低成本路径 Step 1：预算「头部空间」的精确解剖（纯读标签，零 GPU）

背景（来自 literature/refs/p0-2-zerogain-null-protocol.md）：
  arXiv:2608.13087 的"零增益"结论**不能直接套用到我们**，因为它的 in-distribution
  天花板本身就是零；而我们在 B=50 上有约 12.7 个召回点的头部空间。
  但**头部空间里有多少是非平凡的**，必须先拆开——否则就会重蹈 v1 被推翻的同一个陷阱
  （"加法目标下分配有平凡最优解"）。

本脚本计算两个量（全部是**确定性算术**，无估计、无随机）：

  ceiling_uniform(B) = Σᵢ min(B, nᵢ) / Σᵢ nᵢ
      均匀预算：每图都给 B，完美排序下的召回上界。

  ceiling_alloc(B)   = Σᵢ min(B*, nᵢ) / Σᵢ nᵢ  其中 B* 是满足 Σᵢ min(B*, nᵢ) ≤ B·N 的最大整数
      均匀**总量**约束下的 oracle 逐图分配：总输出预算 = B·N（与均匀方案**总预算相同**），
      但可以按图分配。因为 min(k, nᵢ) 凹，贪心边际分配最优，其解恰为"每图 min(B*, nᵢ)"。

  ★ 关键分解（P0-2 的 C-4 要求）：
      Δ_total(B) = ceiling_alloc(B) − ceiling_uniform(B)
                 = Δ_saturation(B) + Δ_ranking(B)
      其中
        Δ_saturation = "稀疏图本来就只需 nᵢ < B 个输出"的**算术**部分 —— **平凡**
        Δ_ranking    = 剩下真正需要"把预算挪给密集图"的部分 —— **非平凡**

  实际上 Δ_saturation 就是"如果不设逐图上限、直接给每图 min(B*, nᵢ)"的吸收量；
  更干净的做法是同时报三个数：
    A) ceiling_uniform(B)                      —— 均匀逐图上限
    B) ceiling_nolimit  = 1.0                  —— 无上限（平凡上界）
    C) ceiling_alloc(B)                        —— 同总预算的 oracle 分配
  非平凡增量 = C − A；而 C 相对 1.0 的差距是"预算总量本身不够"，也不是分配能解决的。

用法：
    python scripts/budget_headroom.py --data data/processed/visdrone/visdrone.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _labels_dir_for(img_dir: Path) -> Path:
    """从右往左替换最后一个 `images` 组件（全局 replace 会改到祖先目录，见 P5）。"""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def load_counts(data_yaml: str, split: str) -> list[int]:
    import yaml

    p = Path(data_yaml)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    img_dir = Path(cfg[split])
    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()
    lbl = _labels_dir_for(img_dir)
    counts = []
    for f in sorted(lbl.glob("*.txt")):
        n = sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        counts.append(n)
    return counts


def ceiling_uniform(counts: list[int], B: int) -> float:
    tot = sum(counts)
    return sum(min(B, n) for n in counts) / tot if tot else 0.0


def ceiling_alloc(counts: list[int], B: int) -> tuple[float, int]:
    """**最优**逐图分配（总预算 = B·N）：贪心，凹目标下可证最优。

    最大化 Σ min(kᵢ, nᵢ) s.t. Σkᵢ ≤ B·N, kᵢ ≥ 0 整数。

    ★ 修正（2026-10-14，由 tests/selftest_proxy_density.py 发现）：
      原实现"二分求均匀 cap B* 再令 kᵢ = min(B*, nᵢ)"**不是最优的**——
      它给出 70.61%，而贪心给出 **70.69%**。数学上不可能"常数代理打败 oracle"，
      所以是**这个"oracle"算错了**。
    为什么贪心最优：目标关于每个 kᵢ 是**凹**的，最优时所有未饱和者的边际收益相等；
    把预算优先给 **nᵢ 最小**的图（边际损失最小）正是贪心。
    """
    c = sorted(counts)
    N = len(c)
    budget = B * N
    prev = 0.0
    i = 0
    while i < N and budget >= (c[i] - prev) * (N - i):
        budget -= (c[i] - prev) * (N - i)
        prev = c[i]
        i += 1
    level = prev + (budget / (N - i) if i < N else 0.0)
    total = sum(c[:i]) + (level * (N - i) if i < N else 0.0)
    tot_sum = sum(c)
    return (min(total, tot_sum) / tot_sum if tot_sum else 0.0), int(level)


def main() -> int:
    ap = argparse.ArgumentParser(description="预算头部空间解剖（纯算术）")
    ap.add_argument("--data", default="data/processed/visdrone/visdrone.yaml")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 500])
    ap.add_argument("--json", default="results/tables/budget-headroom.json")
    a = ap.parse_args()

    out: dict = {"data": a.data, "note": "全部为确定性算术，无估计量、无随机性", "splits": {}}

    for split in a.splits:
        counts = load_counts(a.data, split)
        if not counts:
            print(f"[跳过] {split} 无标签")
            continue
        N, tot = len(counts), sum(counts)
        srt = sorted(counts)
        dens = {
            "n_images": N, "n_boxes": tot,
            "median": srt[N // 2], "p90": srt[int(N * 0.9)], "p99": srt[int(N * 0.99)],
            "max": srt[-1], "mean": round(tot / N, 2),
        }
        print(f"\n{'=' * 78}\n=== {split} === {N} 图 / {tot} 框 | 中位 {dens['median']} "
              f"P90 {dens['p90']} P99 {dens['p99']} max {dens['max']}")
        print(f"{'B':>5} {'uniform':>9} {'oracle分配':>10} {'非平凡增量':>10} {'B*':>5} "
              f"{'受约束图占比':>12} {'饱和图占比':>11}")
        rows = []
        for B in a.budgets:
            cu = ceiling_uniform(counts, B)
            ca, bstar = ceiling_alloc(counts, B)
            frac_capped = sum(1 for n in counts if n > B) / N
            frac_sat = sum(1 for n in counts if n <= B) / N
            rows.append({
                "B": B, "ceiling_uniform": round(cu, 5), "ceiling_alloc": round(ca, 5),
                "delta_nontrivial": round(ca - cu, 5), "B_star": bstar,
                "frac_images_capped": round(frac_capped, 4),
                "frac_images_saturated": round(frac_sat, 4),
            })
            print(f"{B:>5} {cu * 100:>8.2f}% {ca * 100:>9.2f}% {(ca - cu) * 100:>9.2f}pt "
                  f"{bstar:>5} {frac_capped * 100:>11.1f}% {frac_sat * 100:>10.1f}%")
        out["splits"][split] = {"density": dens, "rows": rows}

        # 判读
        print(f"\n  ★ 判读（{split}）：")
        best = max(rows, key=lambda r: r["delta_nontrivial"])
        print(f"    非平凡增量最大出现在 B={best['B']}：{best['delta_nontrivial'] * 100:.2f} pt"
              f"（oracle 分配 {best['ceiling_alloc'] * 100:.2f}% vs 均匀 {best['ceiling_uniform'] * 100:.2f}%）")
        at300 = next((r for r in rows if r["B"] == 300), None)
        if at300:
            print(f"    B=300：非平凡增量仅 {at300['delta_nontrivial'] * 100:.2f} pt "
                  f"→ {'⚠️ 太小，不能做主档' if at300['delta_nontrivial'] < 0.02 else '可用'}")
        at50 = next((r for r in rows if r["B"] == 50), None)
        if at50:
            print(f"    B=50 ：非平凡增量 {at50['delta_nontrivial'] * 100:.2f} pt，"
                  f"受约束图 {at50['frac_images_capped'] * 100:.1f}%")
        print("    ⚠️ 非平凡增量是**上界**：它假设逐图密度已知（oracle）。")
        print("       可实现多少取决于「代理密度 vs GT 数」的 Spearman ρ —— 那是 Step 2。")

    jp = PROJECT_ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果 -> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
