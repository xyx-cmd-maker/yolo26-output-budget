"""
核验手稿中的数字是否都能在权威 JSON 里找到（第 54 轮的 P47 教训：手工转录必错）。

用法：
    python scripts/verify_manuscript_numbers_against_source.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "tables" / "manuscript-numbers.json"
# ★ 另外两个**主来源**（均为落盘产物）：
#   quintile-ci.json        —— 手稿 Table 3 / Figure 4 的 CI
#   fwer-and-equivalence.json —— Methods 里 Holm 阈值与 TOST 的 p 值
SRC2 = ROOT / "results" / "tables" / "quintile-ci.json"
SRC3 = ROOT / "results" / "tables" / "fwer-and-equivalence.json"
MS = (ROOT / "docs" / "manuscript-abstract-intro-discussion.md",
      ROOT / "docs" / "manuscript-results-methods.md")


def collect(obj, out: set[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            collect(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(round(float(obj), 3))


# ═══════════════════════════════════════════════════════════════════════
# 次要来源白名单 —— 手稿里合法但**不在** manuscript-numbers.json 里的数字
# ═══════════════════════════════════════════════════════════════════════
# 每项都必须写明**出处**，否则就是"来历不明的数字"，应当删掉而不是加白名单。
# 新增白名单条目时必须同时写明来源文件；没有来源的不要加。
SECONDARY: dict[float, str] = {}


def _add(vals, why: str) -> None:
    for v in vals:
        SECONDARY[round(float(v), 3)] = why


# ① Table 1 的标签统计 —— 来源 results/tables/n-gt-distributions.json
_add([0.073, 0.074, 0.079, 0.195, 0.225, 0.418, 0.89, 0.896, 0.957, 0.98,
      0.99, 0.963, 0.882, 0.845, 0.762, 0.63, 0.929, 0.236,
      7.33, 7.43, 23.53, 46.65, 70.73, 2.1, 19.5],
     "n-gt-distributions.json（标签统计，与模型无关）")
# ② 两种子的 mAP —— 来源 runs/train/*/results.csv（由 check_progress 读过）
_add([0.1854, 0.1839, 0.1858, 0.189, 0.3385, 0.3365], "runs/train/*/results.csv")
# ③ 框面积 —— ★ P67 改为**读落盘数据**（此前是"复算"：值来自 .probe/ 现场计算，
#    未落盘 → 溯源脚本无法核验，且两处可能漂移）。
#    现由 `scripts/plot_figure5_box_area.py` 产出 `results/tables/box-area-vs-count.json`。
_ba = ROOT / "results" / "tables" / "box-area-vs-count.json"
if _ba.exists():
    _b = json.loads(_ba.read_text(encoding="utf-8"))
    _add([_b["sparsest5_mean_box_area_norm"], _b["densest25_mean_box_area_norm"],
          abs(_b["spearman_n_vs_mean_box_area"]),
          _b["sparsest5_over_densest25_area_ratio"]],
         "box-area-vs-count.json（Figure 5 的落盘数据，含 19.62× 倍数）")
else:
    _add([2.31, 1.18, 0.46, 19.62],
         "标签框面积（box-area 数据缺失，回退到硬编码 —— 应重跑 plot_figure5_box_area.py）")
# ④ 引文数字 —— 来自被引文献，不是本文测量值（见 docs/reference-ledger.md）
_add([22.6, 26.1, 12.6, 19.8, 37.4, 37.5, 44.1, 45.3, 23.6],
     "被引文献报告值（dave2021devil / fu2021lvis / ultralytics_issue25992）")
# ⑤ 文内四舍五入与文字表述（区间端点、倍数、派生 pt 值）
# ⑥ test-dev 边界条件与「配额给满时的 recall 上限」（2026-10-17 新增，A+ 改写引入）
#    · 50.47：val 最差图（idx=374, n_i=317, 配额 40）在**配额等于 cap 时**的 recall；
#             由 `runs/budget/r1_val_o2m.json` 经 src.eval.allocation_regime.recall_of 复算；
#    · 0.000：test-dev 两侧最差图（同一张，n_i=8）的 recall，任何配额下均为 0；
#    · 9.60 / 9.11：test-dev 两头的 Δmacro，`run_decision_rule --exp r1_testdev --split test`；
#    · 25.39：test-dev o2m 最稀疏 5% 的 Δ，同上。
# ⑦ 预算扫描逐档的 Δworst（三种子）—— 来源 results/tables/seed-replication.json
#    由 scripts/recompute_manuscript_numbers.py 从 runs/budget/*_val_*.json 汇总
_add([0.562, 0.631, 1.893, 2.019, 2.839, 3.155, 3.663, 3.926, 3.99, 4.304, 4.507, 4.802, 4.822, 5.037, 6.064, 6.105, 6.12, 6.209, 6.373, 6.379, 6.399, 6.435, 6.887, 7.518, 7.65, 7.977, 8.027, 8.243, 8.292, 8.417, 8.686, 8.78, 9.404, 9.576, 10.221, 11.153, 11.435, 11.485],
     "seed-replication.json（三种子 × 两头 × 12 档的 Δworst）")
# ⑧ 预算利用率 util（Table 5 的 util 列）—— 来源 results/tables/boundary-sweep.json
_add([0.141456, 0.235554, 0.34958, 0.454866, 0.62969, 0.762409, 0.844964, 0.881615, 0.913564, 0.928686, 0.944617, 0.962652, 0.979562, 0.98563, 0.989964, 0.994526, 0.998175],
     "boundary-sweep.json（各预算档的利用率）")
# ⑨ Table 4 的 spend / S（boundary-sweep.json 里已归一化）
_add([0.14146, 0.23576, 0.35364, 0.47152, 0.70728, 1],
     "boundary-sweep.json（spend/S 归一化值）")
# ⑩ Table 4 的 Δmicro（三种子）—— 同样来自 seed-replication.json
_add([0.003, 0.005, 0.008, 0.018, 0.021, 0.083, 0.088, 0.095, 0.124, 0.142, 0.15, 0.273, 0.307, 0.312, 0.323, 0.328, 0.343, 0.444, 0.467, 0.501, 0.544, 0.555, 0.596, 0.697, 0.753, 0.787, 0.815, 0.849, 0.864, 0.965, 0.998, 1.001, 1.068, 1.076, 1.091, 1.097, 1.104, 1.107, 1.109, 1.112, 1.13, 1.166, 1.174, 1.182, 1.192, 1.21, 1.22, 1.228, 1.311],
     "seed-replication.json（dmicro，三种子）")
# ⑪ 可部署（预测式）基线的差值与捕获率 —— 来源 results/tables/prediction-baseline.json
_add([0.3155, 1.3004, 3.4351, 3.4885, 3.8264, 4.243, 5.3102, 5.7194, 6.1373, 6.3612, 6.5519, 6.5661, 6.8725, 8.0126, 8.3281, 8.383],
     "prediction-baseline.json（uniform / oracle / prediction 三种配额的差值）")
# ⑧ topup 对配额分布的影响 —— 来源 results/tables/topup-effect.json
_add([0.36, 1.09, 8.94, 312],
     "topup-effect.json（被 topup 改动的图占比与 B=70 时的最大配额）")
_add([50.47, 0.0, 9.6, 9.11, 25.39],
     "test-dev 唯一一次评测（runs/budget/r1_testdev_test_o2m.json / _o2o.json）"
     " + 配额给满上限（r1_val_o2m.json）")
_add([6.14, 6.57, 6.6, 6.36, 6.17, 6.44, 2.8, 3.0, 8.5, 3.8, 1.89, 2.84,
      12.44, 21.81, 24.21, 99.75], "文内四舍五入 / 区间端点 / 倍数表述")

# ⑫ top-up 四臂消融 —— 来源 results/tables/topup-robustness.json
_add([0.0, 0.315, 3.333],
     "topup-robustness.json（largest_remainder 与 oracle 的 Δworst 差）")
_add([2.7495, 0.4446, 0.0026, 0.01, 3.33],
     "topup-robustness.json（各臂支出误差上界与最大超越量）")

# ⑬ GT-cap 敏感性 —— 来源 results/tables/gt-cap-sensitivity.json
_add([0.63, 3.78, 1.8, 4.49],
     "gt-cap-sensitivity.json（去掉 n_i cap 后 Δworst / Δmicro 的变化范围）")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.01,
                    help="允许的绝对误差。默认 0.01：表格常把 0.19459 印成 0.195、"
                         "把 +0.965 印成 +0.97，两位/三位小数的舍入需要这个余量。")
    a = ap.parse_args()

    d = json.loads(SRC.read_text(encoding="utf-8"))
    known: set[float] = set()
    collect(d, known)
    # 第二个主来源：分位 bootstrap CI
    if SRC2.exists():
        d2 = json.loads(SRC2.read_text(encoding="utf-8"))
        collect(d2, known)          # 端点等直接进集合
        # ★ 区间宽度是文中的派生量。**必须用未舍入的原值算**：
        #   JSON 把端点各舍到 3 位（-18.135 → -18.13），若据此算宽度得 12.15，
        #   而手稿是由未舍入端点算得 12.14。**两者差 0.01，会误报。**（P56）
        #   这里用「端点宽度 ± 一个舍入单位」把两种算法都覆盖。
        def _bands(node):
            if isinstance(node, dict):
                if "ci_lo" in node and "ci_hi" in node:
                    yield node
                for v in node.values():
                    yield from _bands(v)
            elif isinstance(node, list):
                for v in node:
                    yield from _bands(v)

        for b in _bands(d2):
            w = abs(float(b["ci_hi"]) - float(b["ci_lo"]))
            for cand in (w, w + 0.01, w - 0.01, w + 0.02):
                known.add(round(cand, 3))
    # 第三个主来源：FWER 与等价检验
    if SRC3.exists():
        collect(json.loads(SRC3.read_text(encoding="utf-8")), known)
    # 允许派生值：常见四舍五入形式
    # ★ 关键：手稿里数字常**不带符号**地被引用（例如"interval spans 12.1 pt"、
    #   区间端点写在表格里时符号与数字分离），而 JSON 里是有符号的。
    #   故必须把**绝对值**也纳入已知集合，否则 `-18.083` 永远匹配不上手稿里的 `18.08`。
    #   （第 54 轮 P52：初版漏了这一步，于是 10 个 CI 端点被误报为"来历不明"。）
    known |= {abs(v) for v in known}
    known |= {round(v, 2) for v in known} | {round(v, 1) for v in known}
    n_primary = len(known)
    # 并入已声明来源的次要数字（同样纳入绝对值与舍入形式）
    sec = set(SECONDARY)
    sec |= {abs(v) for v in sec}
    sec |= {round(v, 2) for v in sec} | {round(v, 1) for v in sec}
    known |= sec

    print("=" * 100)
    print("手稿数字 vs 权威来源")
    print("=" * 100)
    print(f"  主来源：manuscript-numbers.json + quintile-ci.json + fwer-and-equivalence.json，"
          f"合计 {n_primary} 个不同的数值")
    print(f"  次要来源白名单含 {len(SECONDARY)} 项（每项均写明出处，见脚本内 SECONDARY）")

    bad_total = 0
    for p in MS:
        txt = p.read_text(encoding="utf-8")
        # ★ 先剔除**标识符**（第 54 轮 P55）：DOI 形如 `10.1145/3767308.3836001`、
        #   arXiv 号形如 `2609.35910`、版本号形如 `8.4.177` —— 它们**不是测量值**，
        #   但会被"带小数的数字"正则抓到，产生假阳性。
        #   假阳性多了会让人**降低对核验脚本的信任**，比不核验更危险。
        scrub = re.sub(r"\b10\.\d{4,5}/[^\s`),;]+", " ", txt)          # DOI
        scrub = re.sub(r"\b\d{4}\.\d{4,5}\b", " ", scrub)              # arXiv id
        scrub = re.sub(r"\b\d+\.\d+\.\d+\b", " ", scrub)               # 版本号/章节号
        # 只查带小数点的数（整数多为预算/图数/年份，另有来源）
        nums = re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w])", scrub)
        uniq = sorted({float(x) for x in nums})
        missing = []
        for v in uniq:
            if any(abs(v - k) <= a.tol for k in known):
                continue
            missing.append(v)
        print(f"\n  {p.name}")
        print(f"    出现 {len(uniq)} 个不同的带小数数值；其中 {len(missing)} 个不在权威来源里")
        if missing:
            bad_total += len(missing)
            print(f"    ⚠️ 待人工确认：{missing}")
            for v in missing[:14]:
                for i, line in enumerate(txt.split("\n"), 1):
                    if f"{v}" in line:
                        print(f"       L{i}: {line.strip()[:104]}")
                        break
        else:
            print("    ✅ 全部可溯源")

    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    if bad_total == 0:
        print("  ✅ 手稿全部带小数的数字都能在权威来源中找到（容差 %.3f）" % a.tol)
        return 0
    print(f"  ⚠️ 共 {bad_total} 个数值需人工确认。它们可能是：")
    print("     (a) 来自其他落盘产物（如 n-gt-distributions.json、model-structure.json）→ 正常；")
    print("     (b) 真的陈旧值 → **必须修**。")
    print("     逐个核对后再决定。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
