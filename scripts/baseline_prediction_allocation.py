#!/usr/bin/env python
"""
scripts/baseline_prediction_allocation.py —— **可部署配额基线**（回应外部评估 4.1）

═══ 问题 ═══════════════════════════════════════════════════════════════════
比例配额 `q_i = min(⌈ρ·n_i⌉, cap_i)` 用的是**真实标注数 `n_i`**，
而 `n_i` 在部署时不可得 ⇒ 它是 **oracle**，只给出"若提前知道目标数能买到什么"的上界。

外部评估（`SCI投稿前研究评估与修改路线图.md` §4.1）要求补一条**不依赖标注**的基线：
**用检测器自己输出的预测数 `p_i = n_pred_at_run` 作为分配信号**（跑一次高 max_det 即可得到，
推理时可得，因此是可部署的）。

═══ 本脚本做什么 ═══════════════════════════════════════════════════════════
在同一总预算 `S = ρ·Σ n_i`、同一取整规则、同一 topup 收尾下，比较三种配额：

  ① uniform              —— 每图同配额（基线）
  ② oracle-proportional  —— 按真实标注 `n_i`（上界，不可部署）
  ③ prediction-proportional —— 按检测器预测数 `p_i`（**可部署**）

输出 `results/tables/prediction-baseline.json` 与一张 markdown 表。

用法：python scripts/baseline_prediction_allocation.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "runs" / "budget"


def load_rdr():
    spec = importlib.util.spec_from_file_location("rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rdr = load_rdr()


def plan_proportional_to(key: np.ndarray, cap: np.ndarray, S: float) -> np.ndarray:
    """按给定的信号向量成比例分配：`min(⌈λ·key⌉, cap)`，λ 二分使总花费 ≤ S。"""
    lo, hi = 0.0, 64.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.minimum(np.ceil(mid * key), cap).sum() <= S:
            lo = mid
        else:
            hi = mid
    return np.minimum(np.ceil(lo * key), cap)


def summarise(recs, q, S):
    ev = rdr.evaluate(recs, q)
    n, pred, cap = rdr._cap_of(recs)
    # 实花 = Σ min(q, n, p)（与 run_decision_rule 的约定一致；该处只有标量接口）
    spend = float(np.minimum(np.minimum(q, n), pred).sum())
    # evaluate 返回的是百分数还是 0–1？统一按"若 <=1 则视为分数"处理
    def _pct(v):
        return float(v) * 100.0 if abs(float(v)) <= 1.0 else float(v)

    return {
        "worst": _pct(ev["worst"]),
        "micro": _pct(ev["micro"]),
        "macro": _pct(ev["macro"]),
        "spend": spend,
        "spend_rel_err": (spend - S) / S * 100 if S else 0.0,
    }


def main() -> int:
    files = [("s42", "r1_val_o2m.json"), ("s42-o2o", "r1_val_o2o.json"),
             ("s43", "r1_s43_val_o2m.json"), ("s44", "r1_s44_val_o2m.json")]
    results: dict = {"rho": 0.5, "cases": {}}

    for tag, fn in files:
        p = BUDGET / fn
        if not p.exists():
            print(f"  ⊘ 跳过 {tag}（缺 {fn}）")
            continue
        recs = rdr.load_recs(p)
        n, pred, cap = rdr._cap_of(recs)
        S = 0.5 * float(n.sum())

        q_u = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
        q_o = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
        q_p = rdr.topup(recs, plan_proportional_to(pred, cap, S), S)

        row = {
            "N": len(recs),
            "S": S,
            "sum_n": float(n.sum()),
            "p_median": float(np.median(pred)),
            "n_median": float(np.median(n)),
            "p_ge_n_frac": float((pred >= n).mean()),
            "uniform": summarise(recs, q_u, S),
            "oracle_proportional": summarise(recs, q_o, S),
            "prediction_proportional": summarise(recs, q_p, S),
        }
        # 两个关键差值（pt）
        row["d_worst_oracle"] = row["oracle_proportional"]["worst"] - row["uniform"]["worst"]
        row["d_worst_pred"] = row["prediction_proportional"]["worst"] - row["uniform"]["worst"]
        row["d_macro_oracle"] = row["oracle_proportional"]["macro"] - row["uniform"]["macro"]
        row["d_macro_pred"] = row["prediction_proportional"]["macro"] - row["uniform"]["macro"]
        results["cases"][tag] = row

        print(f"  已算 {tag}:")
        print(f"    uniform  worst {row['uniform']['worst']:.2f}  micro {row['uniform']['micro']:.2f}"
              f"  macro {row['uniform']['macro']:.2f}")
        print(f"    oracle   worst {row['oracle_proportional']['worst']:.2f}"
              f"  (Δworst {row['d_worst_oracle']:+.2f}, Δmacro {row['d_macro_oracle']:+.2f})")
        print(f"    predict  worst {row['prediction_proportional']['worst']:.2f}"
              f"  (Δworst {row['d_worst_pred']:+.2f}, Δmacro {row['d_macro_pred']:+.2f})")

    out = ROOT / "results" / "tables" / "prediction-baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✅ 已写出 {out.relative_to(ROOT)}")

    # markdown 表
    md = ["| Case | N | p median | n median | Δworst oracle | **Δworst prediction** | "
          "Δmacro oracle | **Δmacro prediction** |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, r in results["cases"].items():
        md.append(f"| {tag} | {r['N']} | {r['p_median']:.0f} | {r['n_median']:.0f} | "
                  f"{r['d_worst_oracle']:+.2f} | **{r['d_worst_pred']:+.2f}** | "
                  f"{r['d_macro_oracle']:+.2f} | **{r['d_macro_pred']:+.2f}** |")
    (ROOT / "results" / "tables" / "prediction-baseline.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
