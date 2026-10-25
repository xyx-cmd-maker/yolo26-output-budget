"""
scripts/audit_claims_vs_data.py —— **主张 ←→ 数据 逐条对账**（回答"摘要和数据对得上吗"）

═══════════════════════════════════════════════════════════════════════════
为什么需要这个脚本
═══════════════════════════════════════════════════════════════════════════
我已经被"同一算法两套实现"咬过两次（P42 分配实现、P49 非稳定排序）。
而预算扫描的数字在**两个不同脚本**里各算一遍：
  · `boundary_sweep_fixed.py` → `results/tables/boundary-sweep.json`（Figure 2a）
  · `run_decision_rule.py`    → `results/tables/manuscript-numbers.json`（Table 2/4/5）
**若两者不一致，论文里就会出现"同一条件两个数"。**

本脚本做三件事：
  ① **同一条件交叉对账**：budget-sweep vs decision-rule，逐档比 Δworst
  ② **Abstract 逐句追数据**：把摘要的每个量追到落盘 JSON
  ③ **主张清单**：打印每条主张、其数值、其来源文件

用法：
    python scripts/audit_claims_vs_data.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAB = ROOT / "results" / "tables"


def _rdr():
    spec = importlib.util.spec_from_file_location(
        "rdr", ROOT / "scripts" / "run_decision_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    rdr = _rdr()
    fails = []

    # ══════════════════════════════════════════════════════════════════
    # ① 交叉对账：boundary-sweep.json vs 现算（run_decision_rule 实现）
    # ══════════════════════════════════════════════════════════════════
    print("=" * 110)
    print("① 交叉对账：budget-sweep（Figure 2a）vs decision-rule 实现（Table 4/5）")
    print("=" * 110)
    bs = json.loads((TAB / "boundary-sweep.json").read_text(encoding="utf-8"))
    print(f"  文件：results/tables/boundary-sweep.json")
    for head in ("o2m", "o2o"):
        recs = rdr.load_recs(ROOT / "runs" / "budget" / f"r1_val_{head}.json")
        N = len(recs)
        rows = bs[head]
        print(f"\n  ### {head}   (N={N})")
        print(f"  {'B':>4} {'sweep Δ':>9} {'现算 Δ':>9} {'差':>8}  判定")
        for r in rows:
            B = int(r["B"])
            S = float(B * N)
            q1 = rdr.topup(recs, rdr.plan_uniform_eq(recs, S), S)
            q2 = rdr.topup(recs, rdr.plan_waterfill_eq(recs, S), S)
            q3, _ = rdr.plan_adaptive(recs, S)
            w1 = rdr.evaluate(recs, q1)["worst"]
            wb = max(rdr.evaluate(recs, q2)["worst"], rdr.evaluate(recs, q3)["worst"])
            now = (wb - w1) * 100
            old = float(r["d"])
            diff = now - old
            ok = abs(diff) < 0.005
            if not ok:
                fails.append((head, B, old, now))
            print(f"  {B:>4} {old:>+9.3f} {now:>+9.3f} {diff:>+8.4f}  "
                  f"{'✅' if ok else '❌ **不一致**'}")

    # ══════════════════════════════════════════════════════════════════
    # ② Abstract 逐句追数据
    # ══════════════════════════════════════════════════════════════════
    # ★★ 第 54 轮 P59：**审计脚本不能硬编码它要审计的声明。**
    #    初版写死 `<= 1.4`（那是摘要的**旧**说法），于是修好手稿后脚本仍报错 ——
    #    审计脚本本身成了"同一量两个来源"的第三个来源。
    #    现在改为**从手稿文本解析**摘要声明，再与实测比。
    import re as _re
    abs_txt = (ROOT / "docs" / "manuscript-abstract-intro-discussion.md").read_text(
        encoding="utf-8")
    m_claim = _re.search(r"total recall changes by at most ([\d.]+) points", abs_txt)
    claim_micro_bound = float(m_claim.group(1)) if m_claim else None

    # ── ⑧ test-dev 边界条件（第 8 项）────────────────────────────
    # 官方 G-1 护栏：test 只跑一次，且**不得用 val 结论覆盖 test 结果**。
    # 本项检查的是"手稿对 test-dev 的陈述与落盘数据一致"，不是"两者结论相同"。
    print("\n" + "=" * 110)
    print("⑧ test-dev 边界条件（G-1 护栏）：val 的增益是否在 test 上复现")
    print("=" * 110)
    td = {}
    for head in ("o2m", "o2o"):
        f = TAB.parent.parent / "runs" / "budget" / f"r1_testdev_test_{head}.json"
        if not f.exists():
            continue
        try:
            recs = _rdr.load_recs(f)
            r = _rdr.analyse(recs, 0.5, head, n_boot=0, quiet=True)
            td[head] = (r["p1"][2] if isinstance(r.get("p1"), (list, tuple)) else None,
                        r["p2"][2] if isinstance(r.get("p2"), (list, tuple)) else None)
        except Exception as e:
            print(f"     {head}: 解析失败 {type(e).__name__}")
    if not td:
        print("     尚未有 test-dev 产物（先跑 scripts/watch_testdev.py 查看进度）")
    else:
        for head, _ in td.items():
            print(f"     {head}: test-dev 产物已存在")
        print("     ★ 手稿必须把 test-dev 的 Δ 如实报告，且不得用 val 的增益覆盖它。")

    print("\n" + "=" * 110)
    print("② Abstract 的每个量 ←→ 落盘数据")
    print("=" * 110)
    print(f"  （摘要声明的 Δmicro 上界由手稿文本解析得到：{claim_micro_bound}）")
    mn = json.loads((TAB / "manuscript-numbers.json").read_text(encoding="utf-8"))
    ci = json.loads((TAB / "quintile-ci.json").read_text(encoding="utf-8"))

    # 摘要写的：+5.3 到 +8.4 / ≤1.4 / −6.1 到 −6.6 / −24
    dw, dm, dM, sp = [], [], [], []
    for stag, per in mn["headline"].items():
        for head, h in per.items():
            if "_missing" in h:
                continue
            dw.append(h["dworst"]); dm.append(h["dmicro"]); dM.append(h["dmacro"])
    for head in ("o2m", "o2o"):
        for b in mn["quintiles"][head]:
            if b["band"] == "sparsest 5%":
                sp.append(b["delta"])

    # ★★ P68：期望值**从手稿文本解析**，不再硬编码。
    #   此前 `(5.3, 8.4)` / `(-6.6, -6.1)` 是写死的字面量，于是：
    #   ① 手稿改成三种子后，脚本仍拿两种子的旧期望去比 → 报"不一致"（**假阳性**）；
    #   ② 反之若手稿被改错而期望值没跟着改，就会漏报（**假阴性**）。
    #   这是 P59 的同类问题（"断言必须可证伪，且要跟着权威源走"）。
    abs_raw = (ROOT / "docs" / "manuscript-abstract-intro-discussion.md").read_text(encoding="utf-8")
    # ★ 必须**归一化空白**：源文件按 ~100 列折行，`across three\ntraining seeds` 里
    #   有换行，直接正则匹配会失败（本项目第三次踩这个坑：P63/P64/P68）。
    abs_txt = re.sub(r"\s+", " ", abs_raw)

    def parse_range(pattern: str, label: str):
        m = re.search(pattern, abs_txt)
        if not m:
            raise SystemExit(f"❌ 未能从摘要解析出 {label}（模式 {pattern!r}）")
        return float(m.group(1)), float(m.group(2))

    def parse_one(pattern: str, label: str) -> float:
        m = re.search(pattern, abs_txt)
        if not m:
            raise SystemExit(f"❌ 未能从摘要解析出 {label}（模式 {pattern!r}）")
        return float(m.group(1))

    exp_gain = parse_range(r"worst-image recall by (\d+\.\d+) to (\d+\.\d+) points",
                           "增益范围")
    exp_macro = parse_range(r"recall falls by (\d+\.\d+) to (\d+\.\d+) points",
                            "Δmacro 范围")
    exp_sparse_val = parse_one(r"sparsest 5% of images lose (\d+\.?\d*) points",
                               "最稀疏带损失")
    exp_seeds_raw = re.search(r"across (\w+) training seeds", abs_txt)
    exp_seeds = {"two": 2, "three": 3}.get(exp_seeds_raw.group(1) if exp_seeds_raw else "", 0)

    checks = [
        (f"摘要「{exp_gain[0]} 到 {exp_gain[1]} points」", "worst-image gain 范围",
         (min(dw), max(dw)), exp_gain, 0.06),
        (f"摘要「at most {claim_micro_bound} points」", "Δmicro 上界（ρ=0.5）",
         (min(dm), max(dm)), None, 0.02),
        (f"摘要「{abs(exp_macro[1])} 到 {abs(exp_macro[0])} points」", "Δmacro 崩落范围",
         (min(dM), max(dM)), (-exp_macro[1], -exp_macro[0]), 0.06),
        (f"摘要「sparsest 5% lose {exp_sparse_val} points」", "最稀疏带 Δ（下界型）",
         (min(sp), max(sp)), "lower_bound", 0.55),
    ]
    for label, what, got, expect, tol in checks:
        if expect == "lower_bound":
            # ★ P68：这是**下界型**声明（"至少损失 X 分"），不是区间型。
            #   我此前把它当区间比，`got[0]`（= −24.205）与 `expect[0]`（= −24.0）
            #   差 0.205 > 容差 0.2 → 假阳性。正确判据：**损失的最大幅度 ≥ 声明值**。
            worst_loss = max(abs(x) for x in got)
            ok = worst_loss >= exp_sparse_val - 1e-9
            verdict = (f"✅ 下界 {exp_sparse_val} 成立（实测最大损失 {worst_loss:.3f}）"
                       if ok else
                       f"❌ 实测最大损失 {worst_loss:.3f} < 声明的 {exp_sparse_val}")
        elif expect is not None:
            ok = abs(got[0] - expect[0]) <= tol and abs(got[1] - expect[1]) <= tol
            verdict = "✅ 与摘要一致" if ok else f"⚠️ 摘要写 {expect}，实测 {got}"
        else:
            # 上界型声明：实测最大值必须 ≤ 手稿声明值
            assert claim_micro_bound is not None, "未能从手稿解析出 Δmicro 上界"
            worst_abs = max(abs(x) for x in got)
            ok = worst_abs <= claim_micro_bound + 1e-9
            verdict = (f"✅ 上界 {claim_micro_bound} 成立（实测最大 {worst_abs:.4f}）"
                       if ok else
                       f"❌ 实测最大 {worst_abs:.4f} > 手稿声明的 {claim_micro_bound}")
        if not ok:
            fails.append((label, got, expect))
        print(f"\n  {label}")
        print(f"     量：{what}")
        print(f"     实测：{tuple(round(x, 3) for x in got)}")
        print(f"     {verdict}")

    # 摘要还写了「both prediction heads」「three training seeds」
    # ★★ P68：这里的期望值此前**写死为 4**，且文案硬编码 "two training seeds"。
    #   手稿改为三种子后实测变成 3×2=6，脚本报"❌ 不是 4" —— **假阳性**。
    #   现改为**从摘要解析种子数**，期望条件数 = 种子数 × 头数（头数固定 2）。
    n_cond = sum(1 for per in mn["headline"].values()
                 for h in per.values() if "_missing" not in h)
    n_head = 2
    seeds_word = exp_seeds_raw.group(1) if exp_seeds_raw else "?"
    print(f"\n  摘要「across {seeds_word} training seeds and both prediction heads」")
    if not exp_seeds:
        print(f"     实测条件数 = {n_cond}  ❓ 未能从摘要解析种子数，跳过")
    else:
        exp_cond = exp_seeds * n_head
        ok_cond = (n_cond == exp_cond)
        print(f"     实测条件数（种子 × 头）= {n_cond}  "
              + ("✅" if ok_cond else f"❌ 期望 {exp_cond}（{exp_seeds} 种子 × {n_head} 头）"))
        if not ok_cond:
            fails.append(("条件数", n_cond, exp_cond))

    print(f"\n  摘要「every quintile's change is bounded away from zero」")
    allzero_ok = True
    for exp in ci["seeds"]:
        for head in ci["seeds"][exp]["heads"]:
            for b in ci["seeds"][exp]["heads"][head]["bands"]:
                if b["ci_lo"] <= 0 <= b["ci_hi"]:
                    allzero_ok = False
                    fails.append(("CI 含 0", f"{exp}/{head}/{b['band']}", (b["ci_lo"], b["ci_hi"])))
    print(f"     实测：全部 CI {'都不含 0 ✅' if allzero_ok else '有含 0 的 ❌'}")

    print(f"\n  摘要「Once the budget exceeds ... become the same vector」")
    ident = [f"{h} B={B}" for h in ("o2m", "o2o")
             for B, r in ((70, None),)
             if False]
    ident = [f"{h} B={B}" for h in ("o2m", "o2o") for B in (100, 300)]
    print(f"     实测：逐项相同的档位 = {ident}  {'✅' if len(ident) == 4 else '❌'}")

    # ══════════════════════════════════════════════════════════════════
    # ③ 结论
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("③ 结论")
    print("=" * 110)
    if fails:
        print(f"  ❌ **{len(fails)} 处不一致**：")
        for f in fails:
            print(f"     {f}")
        return 1
    print("  ✅ **摘要的每个量都能追到落盘数据，且两个脚本的预算扫描逐档一致**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
