"""
scripts/run_fwer_and_equivalence.py —— 把 Methods 里那句**断言**变成**计算**

═══════════════════════════════════════════════════════════════════════════
⛔ 为什么必须做（独立审查 R2 的发现）
═══════════════════════════════════════════════════════════════════════════
手稿 `Statistical analysis` 原话：
  "a family-wise correction [holm1979] would not change any decision we report"

**这是一个从未被计算的断言。** 而 Table S1 里 ρ=0.40 的 one-to-many 条件**达到了预注册的
A 档阈值（+10.207 pt）** —— 所以"校正不会改变任何决定"**并不显然成立**。
若审稿人要求证据而我们给不出，这就是"过度声称"。

═══════════════════════════════════════════════════════════════════════════
另外关闭一个缺口：等价检验
═══════════════════════════════════════════════════════════════════════════
手稿说"total recall changes by at most 1.4 points"。这是**观测上界**，
不是"等价"的统计主张。而 `src/eval/stats.py` 里**预注册**了
`EQUIVALENCE_MARGIN_RECALL = 0.010`（**1.0 pt**），TOST 函数也已存在但从未被调用。
**用完好的预注册 margin 做一次 TOST，比"观测到 1.4 所以差不多等价"强得多。**

★★ margin 来源必须写明：**1.0 pt 是预注册的**（`src/eval/stats.py:61`），
   **不是**看结果定的。这是 TOST 合法的前提。

用法：
    python scripts/run_fwer_and_equivalence.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "tables" / "fwer-and-equivalence.json"

RHO = (0.30, 0.40, 0.50, 0.60, 0.70)
HEADS = ("o2m", "o2o")
N_BOOT = 10000
RNG_SEED = 20261014


def _rdr():
    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="FWER 校正与等价检验")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    rdr = _rdr()
    from src.eval.stats import (EQUIVALENCE_MARGIN_RECALL,
                                EQUIVALENCE_MARGIN_RECALL_SOURCE,
                                holm_bonferroni, tost_equivalence)

    out: dict = {
        "n_boot": a.n_boot, "rng_seed": RNG_SEED,
        "equivalence_margin_pt": EQUIVALENCE_MARGIN_RECALL * 100,
        "equivalence_margin_source": EQUIVALENCE_MARGIN_RECALL_SOURCE,
        "family": "the ten rho-sweep conditions (5 rho x 2 heads), one-to-many and one-to-one",
        "tests": {}, "holm": {}, "tost": {},
    }

    # ══════════════════════════════════════════════════════════════════
    # ① FWER：ρ 扫描的 10 个条件，H0: Δworst ≤ 0
    # ══════════════════════════════════════════════════════════════════
    print("=" * 104)
    print("① Holm–Bonferroni：ρ 扫描 10 个条件（H0: Δworst ≤ 0，单侧）")
    print("=" * 104)
    p_raw: dict[str, float] = {}
    obs: dict[str, float] = {}

    for head in HEADS:
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        n = np.array([r["n_gt"] for r in recs], float)
        N = len(n)
        rng = np.random.default_rng(RNG_SEED + (0 if head == "o2m" else 1))
        for rho in RHO:
            S = rho * float(n.sum())
            q_u = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
            q_w = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
            q_a, _ = rdr.plan_adaptive(recs, S)
            e_u = rdr.evaluate(recs, q_u)
            e_w = rdr.evaluate(recs, q_w)
            e_a = rdr.evaluate(recs, q_a)
            # ★★ P121：**TOST 必须测论文报的那个量**。
            #   此前写 `q_best = q_a if e_a["worst"] >= e_w["worst"] else q_w`，
            #   于是 TOST 测的是 **best(adaptive, proportional) − uniform**，
            #   而论文报的 Δmicro 是 **proportional − uniform**。
            #   实测两者有 6 处不一致（最大 0.060）。
            #   这是 P57（macro/micro 混用）同类错误的残留：
            #   **统计检验的对象必须与报告的对象逐字对应**。
            q_best = q_w
            d_obs = (max(e_a["worst"], e_w["worst"]) - e_u["worst"]) * 100

            r_u = rdr._per_image_recall(recs, q_u)
            r_b = rdr._per_image_recall(recs, q_best)
            boot = np.empty(a.n_boot)
            for b in range(a.n_boot):
                pick = rng.integers(0, N, N)
                boot[b] = (np.nanmin(r_b[pick]) - np.nanmin(r_u[pick])) * 100
            # 单侧 p：H0 为 Δ ≤ 0 → p = P(boot ≤ 0)
            p_one = float((boot <= 0).mean())
            p_one = max(p_one, 1.0 / a.n_boot)          # 避免恰好 0（不可检验）
            key = f"{head}/rho={rho:.2f}"
            p_raw[key] = p_one
            obs[key] = round(d_obs, 3)
            print(f"  {key:<18} Δ={d_obs:>+8.3f} pt   p(单侧)={p_one:.5f}")

    holm = holm_bonferroni(p_raw, alpha=0.05)
    # ★ holm_bonferroni 返回 {"test","alpha","n_comparisons","results","note"}，
    #   逐条结果在 ["results"] 下 —— 初版按平铺解析，报 "string indices must be integers"。
    holm_res = holm["results"]
    print("\n  Holm 校正（step-down）：")
    print(f"  {'condition':<18} {'Δ pt':>9} {'p 原始':>10} {'Holm 阈值':>11} {'拒绝 H0':>8}")
    n_rej_raw = sum(1 for v in p_raw.values() if v <= 0.05)
    n_rej_holm = 0
    for k, v in sorted(holm_res.items(), key=lambda kv: kv[1]["rank"]):
        rej = bool(v["reject"])
        n_rej_holm += rej
        print(f"  {k:<18} {obs[k]:>+9.3f} {v['p_raw']:>10.5f} "
              f"{v['holm_threshold']:>11.5f} {'✅' if rej else '—':>8}")
    print(f"\n  ★ 原始 α=0.05 下拒绝 {n_rej_raw}/10；Holm 校正后拒绝 {n_rej_holm}/10")
    changed = (n_rej_raw != n_rej_holm)
    print(f"  ★ 校正是否改变了任何决定：{'**是**' if changed else '否'}")
    out["tests"] = {k: {"delta_pt": obs[k], "p_one_sided": round(p_raw[k], 6)} for k in p_raw}
    out["holm"] = {"n_reject_raw": n_rej_raw, "n_reject_holm": n_rej_holm,
                   "changes_any_decision": bool(changed),
                   "detail": holm_res}

    # ══════════════════════════════════════════════════════════════════
    # ② 等价检验：Δmicro 是否落在预注册的 1.0 pt 内
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 104)
    print(f"② TOST 等价检验：Δmicro（预注册 margin = "
          f"{EQUIVALENCE_MARGIN_RECALL*100:.1f} pt，来源：{EQUIVALENCE_MARGIN_RECALL_SOURCE[:60]}）")
    print("=" * 104)
    for head in HEADS:
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        n = np.array([r["n_gt"] for r in recs], float)
        N = len(n)
        rng = np.random.default_rng(RNG_SEED + (10 if head == "o2m" else 11))
        for rho in RHO:
            S = rho * float(n.sum())
            q_u = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
            q_w = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
            q_a, _ = rdr.plan_adaptive(recs, S)
            e_u, e_w, e_a = (rdr.evaluate(recs, q) for q in (q_u, q_w, q_a))
            # P121：TOST 必须测论文报的量（proportional − uniform），不是 best-of-two
            q_best = q_w
            # ★★ 修（P57）：必须测**与论文同一个量**。
            #   论文的 Δmicro 是**聚合真阳性率之差**：`TP.sum()/n.sum()` 之差。
            #   初版我算的是 `(tp_b - tp_u).mean()`，那是**逐图平均**（= macro 口径），
            #   于是得到 −6.37 pt（恰好是论文的 Δmacro），却拿去和论文的 Δmicro(+1.38) 比。
            #   两者的差异是**加权方式**：micro 按 nᵢ 加权，macro 等权。
            _, _, cap = rdr._cap_of(recs)
            tp_u = np.array([rdr.tp_at(r["tp_prefix"], int(q))
                             for r, q in zip(recs, np.minimum(q_u, cap))], float)
            tp_b = np.array([rdr.tp_at(r["tp_prefix"], int(q))
                             for r, q in zip(recs, np.minimum(q_best, cap))], float)
            dmicro_obs = (tp_b.sum() - tp_u.sum()) / n.sum() * 100
            # TOST 需要**配对样本**。micro 是聚合比值，故把每张图的贡献
            #   `(tp_bᵢ − tp_uᵢ)/Σn` 作为配对样本 a，b 全为 0；
            #   这样 `mean(a − b)` 恰好等于聚合的 Δmicro/100。
            # ★★ P122：**单位错误修正**。
            #   原写 `contrib = (tp_b - tp_u) / n.sum()`，注释声称
            #   "mean(a-b) 恰好等于聚合的 Δmicro/100" —— 这是**错的**：
            #   mean((tp_b-tp_u)/Σn) = Δmicro_fraction / N，比目标量小 **N 倍**。
            #   而 margin 是 Δmicro 的分数单位，于是检验统计量永远远小于 margin，
            #   TOST 永远判"等价" —— 实测 mean=0.000025 而 Δmicro 应为 0.0133。
            #   修法：把每张图的贡献放大 N 倍，使 mean 恰好等于 Δmicro_fraction。
            contrib = (tp_b - tp_u) * N / n.sum()
            r = tost_equivalence(contrib, np.zeros_like(contrib),
                                 margin=EQUIVALENCE_MARGIN_RECALL,
                                 margin_source=EQUIVALENCE_MARGIN_RECALL_SOURCE)
            key = f"{head}/rho={rho:.2f}"
            out["tost"][key] = {
                "quantity": "micro (aggregate TP rate)",
                "dmicro_pt": round(float(dmicro_obs), 3),
                "equivalent_at_pre_registered_margin": bool(r.get("equivalent", False)),
                "p_tost": r.get("p_tost"),
                # P122：函数返回的键是 delta_ci_low / delta_ci_high，
                #   原写 ci_lo / ci_hi 不存在→永远取到 None
                "ci_pt": [None if r.get("delta_ci_low") is None else round(r["delta_ci_low"] * 100, 3),
                          None if r.get("delta_ci_high") is None else round(r["delta_ci_high"] * 100, 3)],
                "margin_pt": EQUIVALENCE_MARGIN_RECALL * 100,
            }
            pt = r.get("p_tost")
            print(f"  {key:<18} Δmicro={dmicro_obs:>+7.3f} pt   "
                  f"p_TOST={pt if pt is None else f'{pt:.4f}'}   "
                  f"等价(±{EQUIVALENCE_MARGIN_RECALL*100:.1f}pt)={r.get('equivalent')}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 已写出 {OUT.relative_to(ROOT)}")
    print("\n★ 这两项把 Methods 里的**断言**变成了**计算**。若结果与断言不符，")
    print("  必须改断言 —— 这正是独立审查要求我们做的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
