#!/usr/bin/env python
"""
scripts/check_all.py —— **一条命令跑完全部投稿前检查**

═══ 为什么需要（P80）═══════════════════════════════════════════════════
本轮新增了 4 个专项检查器（顺序 / 种子数 / 边界条件 / 另加原有若干）。
它们的共同教训是：**我每次都漏跑了其中某一个**，然后由用户发现错误。
「记住要跑」不是可靠机制；**一条命令全跑**才是。

用法：
    python scripts/check_all.py            # 全跑，任一失败即退出码 1
    python scripts/check_all.py --fast     # 跳过耗时的（主张对账 / 数字溯源）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (脚本相对路径, 显示名, 是否耗时)
CHECKS = [
    ("scripts/smoke/check_py_compiles.py", "全部 .py 语法", True),
    ("scripts/smoke/check_no_mojibake.py", "乱码", False),
    ("scripts/smoke/check_quote_nesting.py", "引号嵌套", False),
    ("scripts/smoke/check_stale_numbers.py", "作废口径", False),
    ("scripts/smoke/check_script_entrypoints.py", "脚本入口", False),
    ("scripts/check_section_order.py", "章节/表/图 编号顺序", False),
    ("scripts/check_spend_invariants.py", "spend 不变量", False),
    ("scripts/check_si_items.py", "SI 条目完整性", False)
    ("scripts/check_seed_consistency.py", "种子数一致性", False),
    ("scripts/check_boundary_condition.py", "条件限定完整性", False),
    ("scripts/count_manuscript_words.py", "字数上限", False),
    ("scripts/audit_claims_vs_data.py", "主张 ↔ 数据对账", True),
    ("scripts/verify_manuscript_numbers_against_source.py", "数字溯源", True),
    # ★ P87：PDF 实际渲染版面（本轮的 4 个缺陷全都只有它能发现）
    ("scripts/check_lesson_ids.py", "教训账编号一致性", False),
    # ★ P132：TOST 检验统计量曾算错 N 倍（最严重的缺陷）
    ("scripts/check_statistic_units.py", "统计量量纲", True),
    # ★ P131：JSON 配置字段不一致（s43 的 n_boot 还是 2000）
    ("scripts/check_config_consistency.py", "配置一致性", True),
    # ★ P130：列名口径未说明（Table 1 的 Images 实为有标注图数）
    ("scripts/check_column_definitions.py", "表头口径", True),
    # ★ P129：图注是独立于代码的叙述，会与实现脱节
    ("scripts/check_caption_claims.py", "图注断言 vs 实现", True),
    # ★ P127：百分比/整数断言曾全部从检查里漏过去
    ("scripts/check_percent_claims.py", "百分比断言登记册", True),
    # ★ P124：Table 5 的布尔列曾是手填的，与数据脱节而自动检查全部放行
    ("scripts/check_table5_consistency.py", "Table 5 一致性", True),
    # ★ P106：按条目清点（能发现「某段内容整块消失」，墨迹统计发现不了）
    ("scripts/check_pdf_items.py", "PDF 条目清点", True),
    ("scripts/check_pdf_render.py", "PDF 渲染版面", True),
]


def run(rel: str) -> tuple[bool, str]:
    p = ROOT / rel
    if not p.exists():
        return False, f"缺文件 {rel}"
    r = subprocess.run([PY, str(p)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过耗时检查")
    args = ap.parse_args()

    print("=" * 100)
    print("投稿前全套检查")
    print("=" * 100)

    results = []
    for rel, name, slow in CHECKS:
        if args.fast and slow:
            print(f"  ⊘ {name:<24} （--fast 跳过）")
            continue
        ok, out = run(rel)
        results.append((name, ok, out))
        print(f"  {'✅' if ok else '❌'} {name:<24} {'通过' if ok else '失败'}")

    fails = [(n, o) for n, ok, o in results if not ok]
    print()
    print("=" * 100)
    if fails:
        print(f"  ❌ {len(fails)} / {len(results)} 项失败 —— 逐项输出如下")
        print("=" * 100)
        for name, out in fails:
            print(f"\n### {name}")
            tail = [l for l in out.split("\n") if l.strip()][-14:]
            for l in tail:
                print(f"  {l}")
        print(f"\n  ★ 先修失败项，再重建交付物（build_submission_pdf / build_latex_submission）。")
        return 1
    print(f"  ✅ 全部 {len(results)} 项通过 —— 可以重建交付物")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
