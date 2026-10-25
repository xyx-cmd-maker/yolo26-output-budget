"""
★★ E3 前置：**样本外可实现增益** —— 能否用"在一半图上学到的规则"在另一半上拿到尾部增益？

═══════════════════════════════════════════════════════════════════════════
为什么这是当前最高优先级（选刊报告的红队质疑 #2）
═══════════════════════════════════════════════════════════════════════════
我在第 6 轮报出"非加法目标下分配效应 30–67 pt"，那是 **oracle** 上界
（假设知道每张图的真实 nᵢ）。但同一份文献里，Bae 的
`Sampling Luck Masquerades as Allocation Gain`（arXiv:2608.13087）**恰恰证明**：
**在同一批样本上计算并评估的 oracle 分配增益是伪影，样本外不可区分于零。**

→ **我引它当盟友，却又用 oracle 报数 = 自相矛盾。** 必须测出三件事并**分别报告**：

  (i)  **oracle 上界**：完美知道 nᵢ 时的增益（我已有：30–67 pt）
  (ii) **样本外可实现增益**：规则在 A 半学、在 B 半用（★ 本脚本要测的）
  (iii) **均匀紧 cap 增益**：不分配、只把逐图上限从 B 换成 c*（≈10.87–27.53 pt）

**若 (ii) ≈ 0** → 那 30–67 pt 是伪影，**必须改写叙事**（仍可发表，但更干净）。
**若 (ii) 显著 > 0** → C3 站得住。

═══════════════════════════════════════════════════════════════════════════
本脚本的做法（**零 GPU，纯算术，可在训练期间安全运行**）
═══════════════════════════════════════════════════════════════════════════
真实场景里部署时**不知道** nᵢ。可得的只有**检测器自己的输出**。本脚本用
**"代理 = 该图在均匀 cap 下的实测 TP 数"** 来近似（这是真实的、无需标注的信号）。

两折交叉拟合（配对单位 = 图）：
  1. 图对半分为 A / B 折
  2. 在 A 折上学一个**规则**：把 A 折的图按代理排序，决定"前 x% 的图多给配额"
  3. 把该规则**原样**应用到 B 折，算 B 折的目标值
  4. 换向（B 学 → A 用），两折结果取平均
  5. 与两个基线比：均匀紧 cap（不分配）、oracle（在同一折上直接优化 —— 这是**样本内**，必然最高）

**关键**：oracle 与样本外规则**用同一总预算**，所以差值就是"样本运气"的贡献。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import uniform_cap  # noqa: E402

RNG = np.random.default_rng(20261014)


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


# ── 目标函数（全部显式标注口径）────────────────────────────────────────
def obj_total_micro(counts, k):
    """总召回（微平均：每框等权）"""
    return float(np.minimum(k, counts).sum() / counts.sum())


def obj_worst(counts, k):
    """最坏图召回（非加法）"""
    return float((np.minimum(k, counts) / counts).min())


def obj_worst_decile(counts, k):
    """最坏十分位召回均值（非加法）"""
    rec = np.minimum(k, counts) / counts
    m = max(1, len(rec) // 10)
    return float(np.sort(rec)[:m].mean())


# ── 分配规则 ────────────────────────────────────────────────────────────
def alloc_uniform_tight(counts: np.ndarray, B: int) -> np.ndarray:
    """均匀紧 cap：每图给 c*，不逐图差异化。"""
    return np.full(len(counts), float(uniform_cap(counts, B)))


def alloc_by_proxy(counts: np.ndarray, proxy: np.ndarray, B: int,
                   top_frac: float) -> np.ndarray:
    """按代理分配：**代理最高的 `top_frac` 比例的图**获得加倍的配额。

    规则只有**一个参数**（`top_frac`），所以它可以从 A 折学到、在 B 折用 —— 这才叫"样本外"。
    """
    N = len(counts)
    total = float(B * N)
    cap = float(uniform_cap(counts, B))
    base = np.full(N, cap * (1 - top_frac))          # 基础配额
    order = np.argsort(-np.asarray(proxy, float), kind="mergesort")
    n_top = max(1, int(round(top_frac * N)))
    bonus = np.zeros(N)
    # 把"省下来"的额度给代理最高的一批
    saved = (cap * top_frac) * (N - n_top)           # 非 top 图省下的量
    per_top = saved / n_top if n_top else 0.0
    bonus[order[:n_top]] = per_top
    k = base + bonus
    k = np.minimum(k, counts)                        # 不能超过真实需求
    # 花不完的再按代理序补齐（保证预算用尽）
    left = total - k.sum()
    if left > 1e-9:
        for i in order:
            if left <= 0:
                break
            room = float(counts[i]) - k[i]
            if room > 0:
                take = min(room, left)
                k[i] += take
                left -= take
    return k


def alloc_oracle(counts: np.ndarray, B: int, objective: str) -> np.ndarray:
    """oracle：在该折上**直接**针对目标贪心优化（样本内，必然最高）。"""
    N = len(counts)
    budget = int(B * N)
    k = np.zeros(N, dtype=float)
    remaining = budget
    while remaining > 0:
        rec = np.minimum(k, counts) / counts
        if objective == "total":
            gain = np.where(k < counts, 1.0, -np.inf)
        elif objective == "worst":
            gain = np.where(k < counts, -rec, -np.inf)
        elif objective == "worst_decile":
            thr = np.quantile(rec, 0.10)
            gain = np.where((k < counts) & (rec <= thr), -rec, -np.inf)
        else:
            raise ValueError(objective)
        if (gain == -np.inf).all():
            break
        k[int(np.argmax(gain))] += 1
        remaining -= 1
    return k


def main() -> int:
    print("=" * 100)
    print("E3 前置：样本外可实现增益 vs oracle 上界 vs 均匀紧 cap")
    print("=" * 100)
    print("""
口径声明（红队质疑 #2 要求分别报告）：
  (i)   oracle 上界        —— 完美知道 nᵢ，在**同一折上**优化（样本内，必然最高）
  (ii)  样本外可实现增益   —— 规则在 A 折学、在 B 折用（**这是真问题的答案**）
  (iii) 均匀紧 cap 增益    —— 不分配，只把逐图上限从 B 换成 c*
""")

    for split in ("val", "test"):
        counts = load_counts(split)
        N = len(counts)
        print(f"\n{'#' * 100}\n### {split}：{N} 图 / {int(counts.sum())} 框")
        print(f"{'B':>4} {'目标':>14} │ {'(iii) 均匀紧cap':>16} {'(ii) 样本外':>14} "
              f"{'(i) oracle':>14} │ {'(ii)−(iii)':>12} {'(i)−(ii) 样本运气':>18}")
        print("-" * 100)

        for B in (20, 30, 40, 50):
            # 代理：均匀紧 cap 下的**实测 TP 数上限**（= min(c*, nᵢ)）。
            # 这是真实可得的信号（无需 GT 标注），因为它只依赖 c* 与该图被截断后的输出数。
            cap = uniform_cap(counts, B)
            proxy = np.minimum(cap, counts)     # ★ 这里仍用了 nᵢ —— 见下方 [未完成] 说明
            for obj, fn in (("total(微)", obj_total_micro),
                            ("worst(非加法)", obj_worst),
                            ("worst_decile", obj_worst_decile)):
                # (iii) 均匀紧 cap，两个折分别算（其实两折同分布，但保持口径一致）
                k_uni_all = alloc_uniform_tight(counts, B)
                v_uni = fn(counts, k_uni_all)

                # (ii) 样本外：两折交叉拟合
                idx = RNG.permutation(N)
                halves = [idx[: N // 2], idx[N // 2:]]
                outs = []
                for hi in range(2):
                    A, Bx = halves[hi], halves[1 - hi]
                    cA = counts[A]
                    # 在 A 折上学 top_frac（扫一个粗网格，选 A 折上目标最好的）
                    best_frac, best_v = 0.0, -np.inf
                    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                        kA = alloc_by_proxy(cA, np.minimum(uniform_cap(cA, B), cA), B, frac)
                        v = fn(cA, kA)
                        if v > best_v:
                            best_frac, best_v = frac, v
                    # 在 B 折上**用同一个 frac**（这才是样本外）
                    cB = counts[Bx]
                    kB = alloc_by_proxy(cB, np.minimum(uniform_cap(cB, B), cB), B, best_frac)
                    outs.append(fn(cB, kB))
                v_oos = float(np.mean(outs))

                # (i) oracle：在同一折上直接优化（两折各算再平均，与 (ii) 的折划分一致）
                orc = []
                for hi in range(2):
                    A = halves[hi]
                    cA = counts[A]
                    kA = alloc_oracle(cA, B, {"total(微)": "total",
                                              "worst(非加法)": "worst",
                                              "worst_decile": "worst_decile"}[obj])
                    orc.append(fn(cA, kA))
                v_or = float(np.mean(orc))

                print(f"{B:>4} {obj:>14} │ {v_uni * 100:>15.2f}% {v_oos * 100:>13.2f}% "
                      f"{v_or * 100:>13.2f}% │ {(v_oos - v_uni) * 100:>11.2f}pt "
                      f"{(v_or - v_oos) * 100:>17.2f}pt")

    print("\n" + "=" * 100)
    print("★ 判读（红队质疑 #2 的答案）")
    print("=" * 100)
    print("  · 看 `(ii)−(iii)` 列：**这是「分配是否真有用」的答案**（样本外、可比预算）")
    print("      ≥10 pt → C3 强主张成立")
    print("      2–5 pt → 改写为「真实可用增益远小于 oracle 上界」")
    print("      ≈0     → **那 30–67 pt 是样本运气**，须与 Bae 的 NCO 结论合并为跨域普适伪影")
    print("  · 看 `(i)−(ii)` 列：**这是「样本运气」的大小**（oracle 相对样本外多出来的部分）")
    print()
    print("  ⚠️ [未完成] 本脚本的代理 `min(c*, nᵢ)` **仍用了 nᵢ**（真实场景不可得）。")
    print("     要得到严格样本外的结论，必须用**检测器自己的输出**当代理")
    print("     （如均匀 cap 下的实测预测数 / 置信度和 / 双头分歧）—— 那需要 R1 的推理输出。")
    print("     → 本脚本给出的是**上界性质的**样本外估计；真值只会**更低**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
