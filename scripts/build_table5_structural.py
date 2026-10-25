#!/usr/bin/env python
"""scripts/build_table5_structural.py —— 生成 Table 5 的全部四列（消除手工列）

═══ 为什么需要 ═══════════════════════════════════════════════════════════
Table 5 的 `degenerate` 与 `worst image at cap` 两列此前**不在任何生成脚本里**（手工填写），
因此与实际数据脱节（B=20/25/30 实测有"配额=1 且 n_i>1"的图而表里填 no）。
手填的布尔列是最隐蔽的陈旧数据 —— 14 项自动检查全部放行，因为它们只查数字可溯源。

本脚本从原始预测记录算出四列，落盘 results/tables/table5-structural.json，
供 check_table5_consistency.py 与手稿逐格比对。

用法：python scripts/build_table5_structural.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [2, 6, 10, 15, 20, 25, 30, 40, 50, 70, 100, 300]


def load_rdr():
    spec = importlib.util.spec_from_file_location("rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    rdr = load_rdr()
    out: dict = {"budgets": BUDGETS, "heads": {}}

    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        n, p, cap = rdr._cap_of(recs)
        N = len(recs)
        rows = []
        for B in BUDGETS:
            S = B * N
            q_u = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
            q_w = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
            ru = rdr._per_image_recall(recs, q_u)
            rw = rdr._per_image_recall(recs, q_w)
            iu, iw = int(np.argmin(ru)), int(np.argmin(rw))
            rows.append({
                "B": B,
                "identical_vectors": bool(np.array_equal(q_u, q_w)),
                # P126：表注与正文说的是
                #   「the same image is the worst under both allocations」（identity），
                #   而不是「各自最差图是否到 cap」。
                #   两者在当前数据下恰好同真同假，但定义不同——
                #   必须按表注写的那个算，否则数据一变就脱节。
                "worst_same_image": bool(iu == iw),
                "worst_at_cap_uniform": bool(q_u[iu] >= cap[iu] - 1e-9),
                "worst_at_cap_prop": bool(q_w[iw] >= cap[iw] - 1e-9),
                # 当前数据下两种定义是否一致（供日后变动时告警）
                "definitions_agree": bool((iu == iw) ==
                                          ((q_u[iu] >= cap[iu] - 1e-9) and
                                           (q_w[iw] >= cap[iw] - 1e-9))),
                # degenerate：某图配额被压到 1 **且** 该图目标数 > 1
                #   （n_i == 1 的图配额天生是 1，不算 degenerate）
                "degenerate": bool((((q_u == 1) & (n > 1)) | ((q_w == 1) & (n > 1))).any()),
                "n_degenerate_images": int((((q_u == 1) & (n > 1)) | ((q_w == 1) & (n > 1))).sum()),
            })
        out["heads"][head] = {"N": N, "rows": rows}

    dst = ROOT / "results" / "tables" / "table5-structural.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print("Table 5 结构性审计（从原始预测记录生成）")
    print("=" * 74)
    print(f"  {'B':>4}{'identical':>11}{'cap(u/p)':>14}{'degenerate':>12}{'#deg':>6}")
    print("  " + "-" * 50)
    for r in out["heads"]["o2m"]["rows"]:
        cu = "yes" if r["worst_at_cap_uniform"] else "no"
        cp = "yes" if r["worst_at_cap_prop"] else "no"
        print(f"  {r['B']:>4}{'yes' if r['identical_vectors'] else 'no':>11}"
              f"{cu + ' / ' + cp:>14}{'yes' if r['degenerate'] else 'no':>12}"
              f"{r['n_degenerate_images']:>6}")
    print(f"\n  ✅ 已写出 {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
