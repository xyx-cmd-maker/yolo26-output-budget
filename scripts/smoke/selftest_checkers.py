"""
scripts/smoke/selftest_checkers.py
==================================
**校验器的校验器**：对每个 smoke 校验器跑**阳性对照 + 阴性对照**。

为什么必须存在（教训 P15）：
    我两次把"校验器没报错"当成"校验通过"，而真相是**测试根本没跑**——
    一次是目录没建成（空目录扫描当然干净），一次是样本被豁免规则吃掉。
    "没有报错" ≠ "测过了"。**校验器本身必须先被证明会失败。**

做法：为每个校验器构造
    · 阳性样本：**已知应当被判为有问题**的输入 → 断言校验器**返回非 0**
    · 阴性样本：**已知干净**的输入 → 断言校验器**返回 0**
两向都过，该校验器才算可信。

用法：
    python scripts/smoke/selftest_checkers.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def case(name: str, expect_rc_nonzero: bool, cmd: list[str], cwd: Path) -> tuple[str, bool, str]:
    rc, out = run(cmd, cwd)
    got_nonzero = rc != 0
    ok = got_nonzero == expect_rc_nonzero
    kind = "阳性" if expect_rc_nonzero else "阴性"
    return name, ok, f"[{kind}] rc={rc} 期望{'非0' if expect_rc_nonzero else '0'} → {'✅' if ok else '❌'}\n{out.strip()[-600:]}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="selftest_"))
    results: list[tuple[str, bool, str]] = []
    try:
        # ── 1. check_stale_numbers：阳性 / 阴性 ──────────────────────────
        pos = tmp / "stale_pos"
        pos.mkdir()
        (pos / "a.md").write_text("本机实测 20 秒每 epoch 即可跑完。\n"
                                  "本项目 3 类、目标为交通标志与行人。\n"
                                  "峰值显存只有 1.33 GB。\n", encoding="utf-8")
        results.append(case(
            "check_stale_numbers 阳性（3 个已作废口径）", True,
            [PY, "scripts/smoke/check_stale_numbers.py", "--root", str(pos)], PROJECT_ROOT))

        neg = tmp / "stale_neg"
        neg.mkdir()
        (neg / "a.md").write_text("现行吞吐为 238 s/epoch，数据集 VisDrone2019-DET 共 10 类。\n",
                                  encoding="utf-8")
        results.append(case(
            "check_stale_numbers 阴性（全部现行值）", False,
            [PY, "scripts/smoke/check_stale_numbers.py", "--root", str(neg)], PROJECT_ROOT))

        # ── 2. check_no_mojibake：阳性 / 阴性 ────────────────────────────
        mb_pos = tmp / "mb_pos"
        mb_pos.mkdir()
        (mb_pos / "a.md").write_text("# 鏁版嵁闆嗗凡钀藉湴骞惰浆鎹㈠畬鎴愶紝璁粌姝ｅ湪杩涜涓紝"
                                     "棰勮鎬绘椂闀垮叚灏忔椂\n", encoding="utf-8")
        results.append(case(
            "check_no_mojibake 阳性（真实乱码）", True,
            [PY, "scripts/smoke/check_no_mojibake.py", str(mb_pos)], PROJECT_ROOT))

        mb_neg = tmp / "mb_neg"
        mb_neg.mkdir()
        (mb_neg / "a.md").write_text("# 这是混淆矩阵分析，没有 4×4 混淆矩阵，需要补充。\n"
                                     'ap = argparse.ArgumentParser(description="生成合成夹具数据集")\n',
                                     encoding="utf-8")
        results.append(case(
            "check_no_mojibake 阴性（正常中文，含历史误报句）", False,
            [PY, "scripts/smoke/check_no_mojibake.py", str(mb_neg)], PROJECT_ROOT))

        # ── 3. check_stale_numbers：自扫描必须干净（阴性）───────────────
        results.append(case(
            "check_stale_numbers 阴性（项目自身）", False,
            [PY, "scripts/smoke/check_stale_numbers.py"], PROJECT_ROOT))
        results.append(case(
            "check_no_mojibake 阴性（项目自身）", False,
            [PY, "scripts/smoke/check_no_mojibake.py"], PROJECT_ROOT))
        results.append(case(
            "check_configs 阴性（全部 expcfg dry-run）", False,
            [PY, "scripts/smoke/check_configs.py"], PROJECT_ROOT))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 78)
    print("校验器的校验（阳性必须报错、阴性必须通过）")
    print("=" * 78)
    n_fail = 0
    for name, ok, detail in results:
        print(f"\n{'✅' if ok else '❌'} {name}")
        if not ok:
            n_fail += 1
            print("   " + detail.replace("\n", "\n   "))

    print("\n" + "=" * 78)
    if n_fail:
        print(f"❌ {n_fail}/{len(results)} 项未通过 —— **校验器本身不可信，先修它**")
        return 1
    print(f"✅ 全部 {len(results)} 项通过：三个校验器的阳性/阴性对照均成立")
    print("   → 它们报「通过」时才有意义（见 AGENTS-LESSONS P15）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
