"""
scripts/bootstrap_quintile_ci.py —— 分位代价的 bootstrap 置信区间（**支持多种子**）

═══════════════════════════════════════════════════════════════════════════
为什么需要（质量门 C 的唯一未过项）
═══════════════════════════════════════════════════════════════════════════
手稿里最有冲击力的代价数字是"最稀疏 5% 的图掉 24.2 pt"，
但它**只是 27 张图的均值、没有任何区间**。审稿人必然盯这一点。

═══════════════════════════════════════════════════════════════════════════
★ 方法学选择（必须写进论文，否则区间不可解释）
═══════════════════════════════════════════════════════════════════════════
分位带的**边界**由标签 `n_i` 的分位点决定（`n_i` 与模型无关，是数据集属性）。
因此 bootstrap 的正确做法是：**在每个带内部对图片重采样**，边界固定。
  · 若同时重采样并重算边界，区间会混入"哪张图落在哪个带"的变动，
    而那不是我们要推断的量（我们要推断的是"该带的平均代价"）。
  · 这与"条件于标签分布"的口径一致：论文的 Table 1 也是条件于标签的。

配对性：同一批重采样索引同时用于均匀与水位两种分配，
故 `Δ = mean(rec_w) − mean(rec_u)` 的区间是**配对**区间。

═══════════════════════════════════════════════════════════════════════════
★ 第 54 轮新增：种子维度（P56）
═══════════════════════════════════════════════════════════════════════════
初版只对 `r1`（seed 42）算 CI。而手稿 Table 3 在三种子替换后需要**每个种子各自的 CI**。
**若每个种子用同一个 RNG 起点，`rng` 会在种子之间继续推进 → 结果依赖循环顺序。**
故现在**每个种子用由 `rng_seed` 与种子名派生的独立 RNG**，做到"单独跑某个种子 = 一起跑时该种子的结果"。

用法：
    python scripts/bootstrap_quintile_ci.py
    python scripts/bootstrap_quintile_ci.py --seeds r1 r1_s43 r1_s44
    python scripts/bootstrap_quintile_ci.py --n-boot 10000 --rng-seed 20261014
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "tables" / "quintile-ci.json"

BANDS = (("sparsest 5%", 0.00, 0.05), ("5-25%", 0.05, 0.25), ("25-50%", 0.25, 0.50),
         ("50-75%", 0.50, 0.75), ("densest 25%", 0.75, 1.00))
HEADS = ("o2m", "o2o")


def _rdr():
    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def seed_rng(base: int, exp: str) -> np.random.Generator:
    """由 (基础种子, 实验名) 派生**独立** RNG。

    ★ 为什么：若所有种子共用同一个 `rng` 对象，它会跨种子推进 →
      结果依赖循环顺序，单独跑某种子与一起跑结果不同（不可复现）。
      用名字的哈希派生，保证"单独跑 = 一起跑"。
    """
    h = hashlib.sha256(f"{base}:{exp}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def main() -> int:
    ap = argparse.ArgumentParser(description="分位代价的 bootstrap CI（多种子）")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20261014)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seeds", nargs="+", default=["r1", "r1_s43", "r1_s44"])
    ap.add_argument("--json", type=Path, default=OUT)
    a = ap.parse_args()

    rdr = _rdr()
    out: dict = {
        "rho": a.rho, "n_boot": a.n_boot, "rng_seed": a.rng_seed,
        "method": ("bands fixed by n_i quantiles; images resampled within each band; "
                   "paired indices across the two allocations; percentile interval; "
                   "per-seed RNG derived from (rng_seed, exp) so a single seed reproduces"),
        "seeds": {},
    }

    for exp in a.seeds:
        missing = [h for h in HEADS
                   if not (ROOT / "runs" / "budget" / f"{exp}_val_{h}.json").exists()]
        if missing:
            print(f"  ⚠️ {exp}: 缺 {missing} 的曲线文件，跳过")
            continue
        rng = seed_rng(a.rng_seed, exp)
        out["seeds"][exp] = {"heads": {}}

        for head in HEADS:
            recs = rdr.load_recs(ROOT / "runs" / "budget" / f"{exp}_val_{head}.json")
            n = np.array([r["n_gt"] for r in recs], float)
            N = len(n)
            S = a.rho * float(n.sum())
            q_u = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
            q_w = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
            r_u = rdr._per_image_recall(recs, q_u)
            r_w = rdr._per_image_recall(recs, q_w)
            # ★ 稳定排序（P49）：n_i 有大量并列值，非稳定排序会让并列元素跨带
            order = np.argsort(n, kind="stable")

            out["seeds"][exp]["heads"][head] = {"N": N, "bands": []}
            print("=" * 108)
            print(f"### {exp} / {head}   (N={N},  ρ={a.rho},  bootstrap={a.n_boot})")
            print("=" * 108)
            print(f"  {'band':<13} {'n':>4} {'n_i 范围':>10} │ {'均匀':>8} {'水位':>8} {'Δ':>9} │ "
                  f"{'Δ 的 95% CI':>20} {'P(Δ<0)':>8}")

            for lab, lo_f, hi_f in BANDS:
                idx = order[int(lo_f * N):int(hi_f * N)]
                nb = len(idx)
                mu_u = float(np.nanmean(r_u[idx]))
                mu_w = float(np.nanmean(r_w[idx]))
                d_obs = (mu_w - mu_u) * 100
                boot = np.empty(a.n_boot, dtype=float)
                for b in range(a.n_boot):
                    pick = idx[rng.integers(0, nb, nb)]
                    boot[b] = (np.nanmean(r_w[pick]) - np.nanmean(r_u[pick])) * 100
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p_neg = float((boot < 0).mean())

                out["seeds"][exp]["heads"][head]["bands"].append({
                    "band": lab, "n_images": nb,
                    "n_min": int(n[idx].min()), "n_max": int(n[idx].max()),
                    "uniform": round(mu_u * 100, 3), "proportional": round(mu_w * 100, 3),
                    "delta": round(d_obs, 3),
                    "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                    "boot_mean": round(float(boot.mean()), 3),
                    "p_delta_lt0": round(p_neg, 5),
                })
                print(f"  {lab:<13} {nb:>4} {int(n[idx].min()):>4}-{int(n[idx].max()):<5} │ "
                      f"{mu_u*100:>7.2f}% {mu_w*100:>7.2f}% {d_obs:>+8.2f} │ "
                      f"{f'[{lo:+.2f}, {hi:+.2f}]':>20} {p_neg:>8.5f}")
        print()

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 已写出 {a.json.relative_to(ROOT)}（{len(out['seeds'])} 个种子）")
    print("\n★ 区间**条件于标签分位带边界**、且为配对百分位区间；")
    print("  它回答的是「该带的平均代价是多少」，不是「哪些图落在该带」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
