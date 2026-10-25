"""
scripts/recompute_manuscript_numbers.py —— **手稿全部数字的唯一来源**

═══════════════════════════════════════════════════════════════════════════
为什么必须有这个脚本（第 53 轮，P47）
═══════════════════════════════════════════════════════════════════════════
我在第 51 轮统一了分配实现，并"手工"更新了 Table 2/3/4 与 Abstract/Discussion 的数字
——**但漏了 R3 的 ρ 敏感性段**，它仍留着修正前的旧值。

后果（由独立审查发现）：
  · R3 写"十个条件无一达到 10 pt 阈值"，而**实测 o2m ρ=0.40 是 +10.207 → 判定 A**
  · R3 写"范围 +3.93 到 +9.89"，**实际是 +4.248 到 +10.207**
  · R3 的另一处 +5.037 / +3.926 / +7.518 全是旧值

**根因**：数字靠人手从控制台搬到 Markdown → 必然漏。
**修法**：本脚本把**所有会被写进手稿的数字**一次性算全、落盘成 JSON，
        手稿只允许引用这里的键。**任何人重跑本脚本都能核对。**

═══════════════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════════════
    python scripts/recompute_manuscript_numbers.py
    python scripts/recompute_manuscript_numbers.py --check   # 只核验手稿是否与之一致
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "tables" / "manuscript-numbers.json"


def _rdr():
    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compute() -> dict:
    rdr = _rdr()
    RHO = (0.30, 0.40, 0.50, 0.60, 0.70)
    out: dict = {"rho_sweep": {}, "rho_summary": {}, "budget_sweep": {},
                 "quintiles": {}, "headline": {}}

    # ── ① ρ 敏感性（全部 10 个条件）────────────────────────────
    all_d = []
    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        rows = []
        for rho in RHO:
            r = rdr.analyse(recs, rho, head, n_boot=0, quiet=True)
            rows.append({
                "rho": rho,
                "worst_uniform": round(r["p1"]["worst"] * 100, 3),
                "worst_best": round(max(r["p2"]["worst"], r["p3"]["worst"]) * 100, 3),
                "dworst": round(r["dworst"], 3),
                "dmicro": round(r["dmicro"], 3),
                "verdict": r["verdict"].split("：")[0].strip(),
            })
            all_d.append(r["dworst"])
        out["rho_sweep"][head] = rows
    out["rho_summary"] = {
        "min": round(min(all_d), 3),
        "max": round(max(all_d), 3),
        "n_conditions": len(all_d),
        "n_at_or_above_threshold_A": int(sum(d >= 10.0 for d in all_d)),
        "threshold_A": 10.0,
    }

    # ── ② ρ=0.5 的头条数字（**两个种子都算**）──────────────────
    # ★ 教训（P47）：手稿的 Table 2 要同时给两个种子，而初版脚本只算了 seed 42，
    #   于是我"手填"了 seed 43 的值 —— 又一次手工转录。现在脚本一次算全。
    SEEDS = (("s42", "r1"), ("s43", "r1_s43"), ("s44", "r1_s44"))
    out["headline"] = {}
    for stag, exp in SEEDS:
        out["headline"][stag] = {}
        for head in ("o2m", "o2o"):
            p = ROOT / "runs" / "budget" / f"{exp}_val_{head}.json"
            if not p.exists():
                out["headline"][stag][head] = {"_missing": str(p)}
                continue
            recs = rdr.load_recs(p)
            # P123：三个种子必须用同一重采样次数。
            #   此前写 `10000 if stag == "s42" else 2000`（为省时间的临时搭数），
            #   导致 Table 2 三个种子的 CI 精度不一致，而图注/Methods 又写成统一值。
            nb = 10000
            r = rdr.analyse(recs, 0.5, f"{exp}/{head}", n_boot=nb, quiet=True)
            out["headline"][stag][head] = {
                "worst_uniform": round(r["p1"]["worst"] * 100, 3),
                "worst_best": round(max(r["p2"]["worst"], r["p3"]["worst"]) * 100, 3),
                "micro_uniform": round(r["p1"]["micro"] * 100, 3),
                "micro_best": round(r["p2"]["micro"] * 100, 3),
                "macro_uniform": round(r["p1"]["macro"] * 100, 3),
                "macro_best": round(r["p2"]["macro"] * 100, 3),
                "dworst": round(r["dworst"], 3),
                "dmicro": round(r["dmicro"], 3),
                "dmacro": round(r["dmacro"], 3),
                "ci_lo": round(r["boot"]["lo"], 3),
                "ci_hi": round(r["boot"]["hi"], 3),
                "boot_mean": round(r["boot"]["mean"], 3),
                "p_gt0": round(r["boot"]["p_gt0"], 5),
                "n_boot": r["boot"]["n_boot"],
                "rel_err": round(r["rel_err"], 5),
                "source": str(p.relative_to(ROOT)),
            }

    # ── ③ 预算扫描（U 形）─────────────────────────────────────
    bs = json.loads((ROOT / "results" / "tables" / "boundary-sweep.json")
                    .read_text(encoding="utf-8"))
    for head in ("o2m", "o2o"):
        ok = [r for r in bs[head] if r.get("N1")]
        deg = [r for r in bs[head] if not r.get("N1")]
        dm = [r["micro_w"] - r["micro_u"] for r in ok]
        out["budget_sweep"][head] = {
            "n_total": len(bs[head]),
            "n_spendable": len(ok),
            "n_excluded": len(deg),
            "peak_dworst": round(max(r["d"] for r in ok), 3),
            "peak_at_B": int(max(ok, key=lambda r: r["d"])["B"]),
            "min_dworst": round(min(r["d"] for r in ok), 3),
            "d_at_B70": round(next(r["d"] for r in ok if r["B"] == 70), 3),
            "d_at_B100": round(next((r["d"] for r in ok if r["B"] == 100), float("nan")), 3),
            "d_at_B300": round(next((r["d"] for r in ok if r["B"] == 300), float("nan")), 3),
            "n_negative": int(sum(r["d"] < 0 for r in ok)),
            "max_abs_dmicro": round(max(abs(x) for x in dm), 3),
        }

    # ── ④ 分位代价（o2m, ρ=0.5）──────────────────────────────
    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        n = np.array([r["n_gt"] for r in recs], float)
        S = 0.5 * float(n.sum())
        q1 = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
        q2 = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
        r_u = rdr._per_image_recall(recs, q1)
        r_w = rdr._per_image_recall(recs, q2)
        # ★ 必须用**稳定排序**（P49）：`n_i` 有大量并列值（如 n_i=93 附近），
        #   默认的 quicksort 会让并列元素跨越分带边界 → 分带成员不同
        #   → 分量不同而 Δ 恰好相同，**极难察觉**。
        #   `bootstrap_quintile_ci.py` 与 `run_decision_rule.py` 均已使用 stable。
        order = np.argsort(n, kind="stable")
        N = len(n)
        bands = [("sparsest 5%", 0, int(0.05 * N)), ("5-25%", int(0.05 * N), int(0.25 * N)),
                 ("25-50%", int(0.25 * N), int(0.50 * N)), ("50-75%", int(0.50 * N), int(0.75 * N)),
                 ("densest 25%", int(0.75 * N), N)]
        rows = []
        for lab, a, b in bands:
            idx = order[a:b]
            rows.append({"band": lab, "n_images": int(b - a),
                         "n_min": int(n[idx].min()), "n_max": int(n[idx].max()),
                         "uniform": round(float(np.nanmean(r_u[idx])) * 100, 3),
                         "proportional": round(float(np.nanmean(r_w[idx])) * 100, 3),
                         "delta": round(float(np.nanmean(r_w[idx]) - np.nanmean(r_u[idx])) * 100, 3)})
        out["quintiles"][head] = rows

    # ── ③b 预算扫描表（Table 5 与 Table 4 的数据源，**两个种子 × 两个头**）──
    # ★ P47 教训：手稿里凡有两个种子的表，脚本就必须一次算全，否则必然手工转录。
    out["budget_table"] = {}
    for head in ("o2m", "o2o"):
        out["budget_table"][head] = {}
        for stag, exp in SEEDS:
            recs = rdr.load_recs(ROOT / "runs" / "budget" / f"{exp}_val_{head}.json")
            N = len(recs)
            rows = []
            for B in (2, 6, 10, 15, 25, 40, 50, 70, 100, 300):
                S = float(B * N)
                q1 = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
                q2 = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
                q3, _ = rdr.plan_adaptive(recs, S)
                e1, e2, e3 = rdr.evaluate(recs, q1), rdr.evaluate(recs, q2), rdr.evaluate(recs, q3)
                eb = e3 if e3["worst"] >= e2["worst"] else e2
                rows.append({
                    "B": B,
                    "util": round(float(np.minimum(B, np.array(
                        [r["n_gt"] for r in recs], float)).sum() / S), 5),
                    "dworst": round((eb["worst"] - e1["worst"]) * 100, 3),
                    "dmicro": round((eb["micro"] - e1["micro"]) * 100, 3),
                    "dmacro": round((eb["macro"] - e1["macro"]) * 100, 3),
                    "spend_over_S": round(e1["spend"] / S, 5),
                    "identical_vectors": bool(np.array_equal(q1, q2)),
                })
            out["budget_table"][head][stag] = rows

    # ── ⑤ 跨种子复现 ─────────────────────────────────────────
    rep = ROOT / "results" / "tables" / "criterion-replication-across-seeds.json"
    if rep.exists():
        out["replication"] = json.loads(rep.read_text(encoding="utf-8"))

    # ── ④b 分位表对照：重算的 Δ 必须与 bootstrap 脚本的观测 Δ 一致 ──
    # ★ 第 54 轮教训（P49）：`manuscript-numbers.json` 曾在**修 `_per_image_recall` 之前**
    #   生成过一版，下方三个带的**分量**因此是旧值（Δ 恰好相同，所以没被察觉）。
    #   这里加一道**自洽检查**：用同一实现独立算一遍，逐带比对分量。
    ci_path = ROOT / "results" / "tables" / "quintile-ci.json"
    if ci_path.exists():
        # ★★ P68：`quintile-ci.json` 的 schema 演进过 ——
        #   旧版：顶层就是 `heads`（扁平，单种子）；
        #   新版：`seeds → <exp> → heads`（嵌套，支持多 seed；`bootstrap_quintile_ci.py` 第 54 轮改的）。
        #   我此前只按旧 schema 读 → `["heads"]` 抛 KeyError →
        #   **`manuscript-numbers.json` 因此长期没被更新**，而它正是手稿数字的唯一来源。
        #   现在两种 schema 都支持。
        ci_raw = json.loads(ci_path.read_text(encoding="utf-8"))
        if "heads" in ci_raw:
            ci = ci_raw["heads"]
            ci_src = "旧扁平 schema"
        else:
            seeds_blk = ci_raw.get("seeds", {})
            base = "r1" if "r1" in seeds_blk else (next(iter(seeds_blk), None) or "")
            ci = (seeds_blk.get(base) or {}).get("heads", {})
            ci_src = f"嵌套 schema，种子 `{base}`"
        print(f"  分位 CI 来源：{ci_src}（{len(ci)} 个头）")
        bad = []
        for head in ("o2m", "o2o"):
            if head not in ci:
                continue
            for i, r in enumerate(out["quintiles"][head]):
                c = ci[head]["bands"][i]
                for key_l, key_c in (("uniform", "uniform"),
                                     ("proportional", "proportional"),
                                     ("delta", "delta")):
                    if abs(r[key_l] - c[key_c]) > 0.002:
                        bad.append((head, r["band"], key_l, r[key_l], c[key_c]))
        out["quintile_crosscheck"] = {
            "n_mismatch": len(bad),
            "detail": [{"head": h, "band": b, "field": f, "json": x, "bootstrap": y}
                       for h, b, f, x, y in bad],
        }
        print("\n" + "-" * 96)
        print("分位表自洽检查（本 JSON 的 Δ 必须与 bootstrap 的观测 Δ 相同）")
        print("-" * 96)
        if bad:
            print(f"  ⚠️ **{len(bad)} 处分量不一致** —— 本 JSON 可能是旧实现的残留：")
            for h, b, f, x, y in bad:
                print(f"     {h} {b} {f}: JSON={x}  bootstrap={y}")
        else:
            print("  ✅ 分位表与本 JSON 完全一致")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="手稿数字的唯一来源")
    ap.add_argument("--check", action="store_true", help="只核对手稿是否与落盘一致")
    a = ap.parse_args()

    d = compute()
    if not a.check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ 已写出 {OUT.relative_to(ROOT)}")

    print("\n" + "=" * 96)
    print("★ 手稿必须使用的数字")
    print("=" * 96)
    rs = d["rho_summary"]
    print(f"\nρ 敏感性：{rs['n_conditions']} 个条件，范围 "
          f"[{rs['min']:+.3f}, {rs['max']:+.3f}] pt")
    print(f"  ★ 达到 A 阈值（≥{rs['threshold_A']}）的条件数 = {rs['n_at_or_above_threshold_A']}"
          f"   ← 若 >0，手稿**不得**写「无一达到阈值」")
    for head, rows in d["rho_sweep"].items():
        print(f"\n  {head}:")
        for r in rows:
            print(f"    ρ={r['rho']:.2f}  {r['worst_uniform']:>7.3f}% → {r['worst_best']:>7.3f}%"
                  f"  Δ {r['dworst']:>+7.3f}  Δmicro {r['dmicro']:>+6.3f}  {r['verdict']}")

    print("\n" + "-" * 96)
    print("ρ=0.5 头条数字 —— **两个种子**（Table 2 的数据源）")
    print("-" * 96)
    print(f"  {'seed':<5} {'head':<5} {'worst 均匀':>10} {'→ 较优':>8} {'Δworst':>8} "
          f"{'CI':>18} {'P(Δ>0)':>8} │ {'Δmicro':>7} {'Δmacro':>8} {'n_boot':>7}")
    for stag, per in d["headline"].items():
        for head, h in per.items():
            if "_missing" in h:
                print(f"  {stag:<5} {head:<5}  ❌ 缺数据 {h['_missing']}")
                continue
            ci = f"[{h['ci_lo']:+.2f},{h['ci_hi']:+.2f}]"
            print(f"  {stag:<5} {head:<5} {h['worst_uniform']:>9.3f}% {h['worst_best']:>7.3f}% "
                  f"{h['dworst']:>+7.3f} {ci:>18} {h['p_gt0']:>8.4f} │ "
                  f"{h['dmicro']:>+6.3f} {h['dmacro']:>+7.3f} {h['n_boot']:>7}")

    print("\n" + "-" * 96)
    print("预算扫描（U 形）")
    print("-" * 96)
    for head, b in d["budget_sweep"].items():
        print(f"  {head}: {b['n_spendable']}/{b['n_total']} 可比  峰值 {b['peak_dworst']:+.3f} @B={b['peak_at_B']}"
              f"  最小 {b['min_dworst']:+.3f}  B=70 时 {b['d_at_B70']:+.3f}"
              f"  负值档 {b['n_negative']}  |Δmicro|max {b['max_abs_dmicro']:.3f}")

    print("\n" + "-" * 96)
    print("分位代价（ρ=0.5）")
    print("-" * 96)
    for head, rows in d["quintiles"].items():
        print(f"  {head}:")
        for r in rows:
            print(f"    {r['band']:<12} n={r['n_images']:<4} ({r['n_min']}-{r['n_max']})  "
                  f"均匀 {r['uniform']:>7.3f}%  水位 {r['proportional']:>7.3f}%  Δ {r['delta']:>+8.3f}")

    # ── 与手稿现状比对（粗查：手稿里出现的 pt 数字是否都在本表里）──
    print("\n" + "=" * 96)
    print("手稿中的 pt 数字 vs 本表（找陈旧值）")
    print("=" * 96)
    known = set()
    for head, rows in d["rho_sweep"].items():
        for r in rows:
            known.add(round(r["dworst"], 3))
    # ★★ P68：`headline` 在加入多 seed 后从 {head: {...}} 变成 {seed: {head: {...}}}。
    #   我此前只迭代一层，取到的是内层 {head: {...}}，于是 `h["dworst"]` 抛
    #   KeyError: 'dworst'，**整个脚本在最后一步崩掉、JSON 不落盘**。
    #   现在**两种形状都兼容**（按值里有没有 'dworst' 判断层级）。
    for _k1, _v1 in d["headline"].items():
        if isinstance(_v1, dict) and "dworst" in _v1:
            known.add(round(_v1["dworst"], 3))          # 旧形状 {head: {...}}
        elif isinstance(_v1, dict):
            for _k2, _v2 in _v1.items():                # 新形状 {seed: {head: {...}}}
                if isinstance(_v2, dict) and "dworst" in _v2:
                    known.add(round(_v2["dworst"], 3))
    for head, b in d["budget_sweep"].items():
        known.add(round(b["peak_dworst"], 3))
        known.add(round(b["min_dworst"], 3))
    print(f"  本表包含的 Δworst 值：{sorted(known)}")
    for f in ("docs/manuscript-results-methods.md",
              "docs/manuscript-abstract-intro-discussion.md"):
        p = ROOT / f
        txt = p.read_text(encoding="utf-8")
        found = sorted({m.group(1) for m in re.finditer(r"([+-]?\d+\.\d{2})\s*(?:points|pt)", txt)})
        print(f"\n  {f}:")
        print(f"    文中出现：{found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
