"""
★★ E3 的统计补强：TOST 等价检验 + 逐图配对自助置信区间

═══════════════════════════════════════════════════════════════════════════
为什么必须做（质量门 C 未过 + Kill List #3）
═══════════════════════════════════════════════════════════════════════════
第 9 轮我报出「样本外可实现增益 = −3.66 ~ +1.17 pt」，并据此说
**"分配在加法目标下无用"**。但那只报告了**点估计的取值范围**，没有：
  · **置信区间** —— 无法区分"确实 ≈0"与"噪声太大看不出"
  · **等价检验** —— **统计上，不显著 ≠ 等价**。Kill List #3 正是此条：
    "声称无差异/无效应却未做等价或非劣检验"

本脚本补两件事：
  1. **逐图配对自助**（配对单位 = 图，与项目预注册一致）给 (ii)−(iii) 的 CI
  2. **TOST 等价检验**（two one-sided tests），在**预先声明**的 margin 下判定是否等价

⚠️ margin 的来源必须声明（P16 的教训：容差必须注明精度来源）：
  · 本项目已预注册 `EQUIVALENCE_MARGIN_RECALL = 0.010`（1.0 pt）
  · 本脚本额外报 **0.5 pt（严格档）**，用来说明"能在多窄的范围内宣称等价"

用法（纯算术，零 GPU，训练期间可安全运行）：
    python scripts/e3_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.stats import RNG_SEED, tost_equivalence  # noqa: E402
from src.eval.proxy_density import uniform_cap  # noqa: E402

N_BOOT = 10_000
MARGINS = (0.005, 0.010)   # 严格档 / 项目预注册档（单位：召回比例）


def load_counts(split: str) -> np.ndarray:
    cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
    d = Path(cfg["path"]) / cfg[split]
    p = list(d.parts)
    for i in range(len(p) - 1, -1, -1):
        if p[i] == "images":
            p[i] = "labels"
            break
    return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                     for f in sorted(Path(*p).glob("*.txt"))], dtype=float)


def alloc_uniform(counts: np.ndarray, B: int) -> np.ndarray:
    """均匀紧 cap（无逐图差异化）—— 基线 (iii)。"""
    return np.full(len(counts), float(uniform_cap(counts, B)))


def alloc_by_proxy(counts: np.ndarray, proxy: np.ndarray, B: int, top_frac: float,
                   total_budget: float) -> np.ndarray:
    """按代理分配（与 e3_outsample_allocation.py 同一实现，保证可比）。"""
    N = len(counts)
    cap = float(uniform_cap(counts, B))
    base = np.full(N, cap * (1 - top_frac))
    order = np.argsort(-np.asarray(proxy, float), kind="mergesort")
    n_top = max(1, int(round(top_frac * N)))
    saved = (cap * top_frac) * (N - n_top)
    k = base.copy()
    k[order[:n_top]] += (saved / n_top if n_top else 0.0)
    k = np.minimum(k, counts)
    left = total_budget - k.sum()
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


def per_image_recall(counts: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.minimum(k, counts) / np.maximum(counts, 1e-12)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOT,
                        seed: int = RNG_SEED) -> tuple[float, float, float]:
    """逐图配对自助：对**差值向量**重采样。返回 (mean, lo, hi) 的 95% CI。"""
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    print("=" * 100)
    print("E3 统计补强：TOST 等价检验 + 逐图配对自助 CI")
    print("=" * 100)
    print(f"""
口径（与 docs/finding-e3-allocation-gain-is-sample-luck.md 严格一致）：
  (iii) 均匀紧 cap   —— 每图给 c*，**不**逐图差异化
  (ii)  样本外分配   —— 规则在 A 折学 top_frac、在 B 折用（两折交叉，换向取平均）
  检验的量 = 逐图 recall 的差 (ii) − (iii)，**配对单位 = 图**

★ 关键统计点：**不显著 ≠ 等价**。必须做 TOST 才能说"无差异"（Kill List #3）。
   margin = 0.5 pt（严格档）/ 1.0 pt（本项目预注册的 EQUIVALENCE_MARGIN_RECALL）
   自助次数 = {N_BOOT}，种子 = {RNG_SEED}
""")

    for split in ("val", "test"):
        counts = load_counts(split)
        N = len(counts)
        print(f"\n{'#' * 100}\n### {split}：N={N} 图")
        print(f"{'B':>5} │ {'(ii)−(iii) 均值':>15} {'95% CI':>24} │ "
              f"{'TOST@0.5pt':>12} {'TOST@1.0pt':>12} │ 判定")
        print("-" * 100)

        rng = np.random.default_rng(RNG_SEED)
        for B in (20, 30, 40, 50):
            budget = float(B * N)
            cap = float(uniform_cap(counts, B))
            proxy_all = np.minimum(cap, counts)
            k_uni = alloc_uniform(counts, B)
            rec_uni = per_image_recall(counts, k_uni)

            # 两折交叉：规则在 A 折学、在 B 折用；对**每张图**都得到一个预测
            idx = rng.permutation(N)
            halves = [idx[: N // 2], idx[N // 2:]]
            rec_oos = np.zeros(N)
            for hi in range(2):
                A, Bx = halves[hi], halves[1 - hi]
                cA, pA = counts[A], proxy_all[A]
                best_frac, best_v = 0.0, -np.inf
                for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                    kA = alloc_by_proxy(cA, pA, B, frac, float(B * len(A)))
                    v = float(np.minimum(kA, cA).sum() / cA.sum())
                    if v > best_v:
                        best_frac, best_v = frac, v
                cB, pB = counts[Bx], proxy_all[Bx]
                kB = alloc_by_proxy(cB, pB, B, best_frac, float(B * len(Bx)))
                rec_oos[Bx] = per_image_recall(cB, kB)

            diff = rec_oos - rec_uni
            mean, lo, hi = paired_bootstrap_ci(diff)
            row = f"{B:>5} │ {mean * 100:>14.3f}pt [{lo * 100:>7.3f}, {hi * 100:>7.3f}]pt │"

            results = {}
            for m in MARGINS:
                # ★ 真实签名（已 read 确认）：tost_equivalence(a, b, *, margin, alpha, margin_source)
                #   它内部自己做配对差 d = a - b，所以传**两组逐图召回**，不传差值。
                r = tost_equivalence(
                    rec_oos, rec_uni,
                    margin=m,
                    margin_source=("0.5 pt：本脚本声明的严格档（用于说明能在多窄的范围内宣称等价）"
                                   if m == 0.005 else
                                   "1.0 pt：本项目预注册的 EQUIVALENCE_MARGIN_RECALL (src/eval/stats.py)"),
                )
                results[m] = bool(r["equivalent"])
                if B == 20:
                    print(f"      [TOST margin={m * 100:.1f}pt] n_pairs={r['n_pairs']} "
                          f"Δ={r['delta_mean'] * 100:+.4f}pt p_tost={r['p_tost']:.4g} "
                          f"equivalent={r['equivalent']}")
                row += f" {'✅等价' if results[m] else '❌不等价':>12}"

            verdict = ("**CI 含 0 且等价**" if (lo < 0 < hi and results.get(0.010))
                       else ("CI 含 0" if lo < 0 < hi else "CI 不含 0"))
            print(row + f" │ {verdict}")
            print(f"      实测逐图差：均值 {diff.mean() * 100:+.3f}pt  标准差 {diff.std() * 100:.3f}pt  "
                  f"中位 {np.median(diff) * 100:+.3f}pt  非零图占比 {np.mean(np.abs(diff) > 1e-9) * 100:.1f}%")

    print("\n" + "=" * 100)
    print("★ 判读")
    print("=" * 100)
    print("""  · **TOST@0.5pt 通过** → 可在 **0.5 pt 以内**宣称"样本外分配与均匀紧 cap 等价"（强主张）
  · **TOST@1.0pt 通过但 @0.5pt 不过** → 只能在 1.0 pt 以内宣称等价（弱主张）
  · **两者都不过** → **不能宣称等价**；只能说"未检出显著差异"（措辞必须降级）

  ⚠️ 若两者都不过，第 9 轮的"分配在加法目标下无用"必须改写为
     「**未检出**样本外分配相对均匀紧 cap 的显著增益」，并**补齐功效分析**
     （样本量能否检出 1 pt 的差异）。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
