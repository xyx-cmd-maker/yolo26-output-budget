"""
scripts/verify_seed_replication.py —— **路线 A 的跨种子复现检验**（取代 verify_criterion_across_seeds.py）

═══════════════════════════════════════════════════════════════════════════
⛔ 为什么重写（第 54 轮）
═══════════════════════════════════════════════════════════════════════════
旧脚本 `verify_criterion_across_seeds.py` 的立论是"判据的价值全在**预测力**"。
但该判据已被**证伪**（置换检验：打乱检测器后标签 198/200 不变而 P(gain>0)=0；
另有 10 个实测反例；且 `pᵢ < nᵢ` 在全部记录上从未发生 → `capᵢ ≡ nᵢ`）：
见 `docs/finding-CORE-CLAIM-FALSIFIED.md`。

**所以旧脚本在验证一个已撤回的主张。** 若继续跑，它会生成一份"判据跨种子复现"的报告，
而那个主张已经不存在 —— 这正是 P47 那类"验证错的东西"的复发。

═══════════════════════════════════════════════════════════════════════════
本脚本现在验证什么（路线 A 的**幸存主张**）
═══════════════════════════════════════════════════════════════════════════
  A1 **最坏图召回的增益**在哪些档位为正、量级跨种子差多少（手稿 C1）
  A2 **位置性**：收益是否来自"最坏图换人"而非改善固定图（手稿 C2）
  A3 **总分位代价**是否同向：最稀疏带为负、最密带为正（手稿 C5）
  A4 **零增益是否结构性**：两配额向量是否逐项相同（手稿 C6）
  A5 **Δmicro 上界**：全域最大 |Δmicro|（手稿 C4）

用法：
    python scripts/verify_seed_replication.py
    python scripts/verify_seed_replication.py --seeds r1 r1_s43 r1_s44
    python scripts/verify_seed_replication.py --json out.json
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
OUT = ROOT / "results" / "tables" / "seed-replication.json"

# ★ 预算表必须**覆盖手稿 Table 4/5 用到的全部档位**。
#   旧脚本硬编码 (2,6,10,15,25,50,70,100,300)，**漏了 20/30/40**，
#   于是手稿"ten conditions"这个计数只是那张表的产物（真值是 12 或 22）。见 P47。
BUDGETS = (2, 6, 10, 15, 20, 25, 30, 40, 50, 70, 100, 300)
BANDS = (("sparsest 5%", 0.00, 0.05), ("5-25%", 0.05, 0.25), ("25-50%", 0.25, 0.50),
         ("50-75%", 0.50, 0.75), ("densest 25%", 0.75, 1.00))


def _rdr():
    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def analyse_seed(rdr, exp: str) -> dict | None:
    """对一个种子，算 A1–A5 所需的全部量。缺数据返回 None。"""
    out = {"exp": exp, "heads": {}}
    for head in ("o2m", "o2o"):
        p = ROOT / "runs" / "budget" / f"{exp}_val_{head}.json"
        if not p.exists():
            return None
        recs = rdr.load_recs(p)
        n = np.array([r["n_gt"] for r in recs], float)
        N = len(n)
        order = np.argsort(n, kind="stable")      # ★ 必须稳定，见 P49
        rows, all_dmicro = [], []
        band_rows = None
        for B in BUDGETS:
            S = float(B * N)
            q1 = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
            q2 = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
            q3, _ = rdr.plan_adaptive(recs, S)
            e1, e2, e3 = rdr.evaluate(recs, q1), rdr.evaluate(recs, q2), rdr.evaluate(recs, q3)
            # 报哪个臂：与 analyse() 一致，取最坏图更优的那一个
            q_best = q3 if e3["worst"] >= e2["worst"] else q2
            eb = e3 if e3["worst"] >= e2["worst"] else e2
            # A2：最坏图是否换人（★ 传**配额向量**，不是 evaluate 的返回字典）
            r1_ = rdr._per_image_recall(recs, q1)
            rb_ = rdr._per_image_recall(recs, q_best)
            i_u, i_b = int(np.nanargmin(r1_)), int(np.nanargmin(rb_))
            rows.append({
                "B": B,
                "dworst": round((eb["worst"] - e1["worst"]) * 100, 3),
                "dmicro": round((eb["micro"] - e1["micro"]) * 100, 3),
                "dmacro": round((eb["macro"] - e1["macro"]) * 100, 3),
                "identical_vectors": bool(np.array_equal(q1, q2)),
                "worst_same_image": bool(i_u == i_b),
            })
            all_dmicro.append(abs(rows[-1]["dmicro"]))
        # A3：分位代价，**固定 ρ = 0.5**（与手稿 Table 3 / Figure 4 同一条件）。
        # ★ 第 54 轮 P54：初版按"第一个 ≥ 0.5·mean(n) 的档位"选，会落到 B=40，
        #   于是这里的分位代价（−13.04）与手稿 Table 3（−24.20，在 B≈25）**对不上**。
        #   跨种子比较必须在**同一个条件**上做，否则看起来像不一致。
        S = 0.5 * float(n.sum())
        q1 = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
        q2 = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
        r_u = rdr._per_image_recall(recs, q1)
        r_w = rdr._per_image_recall(recs, q2)
        band_rows = {"rho": 0.5, "bands": []}
        for lab, lo, hi in BANDS:
            idx = order[int(lo * N):int(hi * N)]
            band_rows["bands"].append({
                "band": lab,
                "n_images": len(idx),
                "delta": round(float(np.nanmean(r_w[idx]) - np.nanmean(r_u[idx])) * 100, 3),
            })
        out["heads"][head] = {
            "rows": rows,
            "bands": band_rows,
            "max_abs_dmicro": round(max(all_dmicro), 3),
            "n_positive_dworst": int(sum(r["dworst"] > 0 for r in rows)),
            "n_negative_dworst": int(sum(r["dworst"] < 0 for r in rows)),
            "n_structural_zero": int(sum(r["identical_vectors"] for r in rows)),
            "n_worst_image_changed": int(sum(not r["worst_same_image"] for r in rows)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="路线 A 的跨种子复现检验")
    ap.add_argument("--seeds", nargs="+", default=["r1", "r1_s43"])
    ap.add_argument("--json", type=Path, default=OUT)
    a = ap.parse_args()

    rdr = _rdr()
    res = {"budgets": list(BUDGETS), "seeds": {}}
    for exp in a.seeds:
        r = analyse_seed(rdr, exp)
        if r is None:
            print(f"  ⚠️ {exp}: 缺 runs/budget/{exp}_val_*.json，跳过")
            continue
        res["seeds"][exp] = r

    if len(res["seeds"]) < 2:
        print("  ❌ 少于 2 个种子，无法做跨种子检验")
        return 1

    print("=" * 108)
    print("A1  最坏图增益：跨种子逐档对照（手稿 C1）")
    print("=" * 108)
    names = list(res["seeds"])
    for head in ("o2m", "o2o"):
        print(f"\n  ### {head}")
        print(f"  {'B':>4} │ " + " │ ".join(f"{n:>10}" for n in names) + " │  符号一致 │  同向量")
        for k, B in enumerate(BUDGETS):
            vals = [res["seeds"][n]["heads"][head]["rows"][k]["dworst"] for n in names]
            idv = res["seeds"][names[0]]["heads"][head]["rows"][k]["identical_vectors"]
            same = all((v > 0) == (vals[0] > 0) for v in vals)
            print(f"  {B:>4} │ " + " │ ".join(f"{v:>+10.3f}" for v in vals)
                  + f" │ {'✅' if same else '❌':>8} │ {'是' if idv else '否':>6}")

    print("\n" + "=" * 108)
    print("A2/A4/A5  汇总（手稿 C2 / C6 / C4）")
    print("=" * 108)
    print(f"  {'seed':<8} {'head':<5} {'正档':>5} {'负档':>5} {'结构性零':>9} "
          f"{'最坏图换人':>11} {'|Δmicro|max':>12}")
    for n, r in res["seeds"].items():
        for head, h in r["heads"].items():
            print(f"  {n:<8} {head:<5} {h['n_positive_dworst']:>5} {h['n_negative_dworst']:>5} "
                  f"{h['n_structural_zero']:>9} {h['n_worst_image_changed']:>11} "
                  f"{h['max_abs_dmicro']:>12.3f}")

    print("\n" + "=" * 108)
    print("A3  分位代价：跨种子同向性（手稿 C5）")
    print("=" * 108)
    for head in ("o2m", "o2o"):
        print(f"\n  ### {head}（固定 ρ = 0.5，与手稿 Table 3 / Figure 4 同一条件）")
        print(f"  {'band':<14} │ " + " │ ".join(f"{n:>10}" for n in names) + " │  同向")
        for k, (lab, _, _) in enumerate(BANDS):
            vals = []
            for n in names:
                b = res["seeds"][n]["heads"][head]["bands"]
                vals.append(b["bands"][k]["delta"] if b else float("nan"))
            same = all((v > 0) == (vals[0] > 0) for v in vals if not np.isnan(v))
            print(f"  {lab:<14} │ " + " │ ".join(f"{v:>+10.2f}" for v in vals)
                  + f" │ {'✅' if same else '❌':>5}")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 已写出 {a.json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
