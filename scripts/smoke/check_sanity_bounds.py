"""
scripts/smoke/check_sanity_bounds.py
====================================
**机器校验器**：脚本输出里出现"超过物理上限的增益/比例"时报警。

═══════════════════════════════════════════════════════════════════════════
为什么必须存在（教训 P23）
═══════════════════════════════════════════════════════════════════════════
我写 `k3_rule_check.py` 比较"cap=B"与"cap=300"的可达召回，输出
`cap=300 有 **5911%** 的增益` —— **荒谬到应该立刻警觉**，但我把它当成
"规则更强"的证据，还准备写成部署建议。

**根因**：我拿一个**违反约束**的量（`c > B` 时 `Σmin(c,nᵢ) > B·N`，即超预算）
去和**满足约束**的基线比。

**类比**：任何"精度提升 5000%""覆盖率 300%"这类数字，都是量纲或约束出错的信号。

═══════════════════════════════════════════════════════════════════════════
判据（对**脚本源码里的输出字面量**做静态检查，不执行脚本）
═══════════════════════════════════════════════════════════════════════════
扫描 `scripts/` 与 `src/` 下的 `.py`，找**同时**满足以下条件的行：
  1. 含 `%` 格式化输出（`{...:.1f}%` / `%` 等）
  2. 附近（同一行或前后 3 行）出现"增益/占比/比例/提升/覆盖率"等**有物理上限的语义词**
  3. 该处**没有**显式的上界钳制（如 `min(...)`、`clip`）或"⚠️超预算"类警示标注

并额外检查：项目输出里是否出现过**已知的荒谬数字**（如 5911、1869、4411 这类）。

退出码：0 = 干净；1 = 发现可疑。

★ 注意：本校验器是**启发式**的，会有误报。**误报要逐条给豁免理由**（同 `check_stale_numbers.py` 的做法）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 有物理上限的语义词（其"增益/比例"不可能超过 100%）
BOUNDED_WORDS = ("增益", "占比", "覆盖率", "比例", "提升", "召回率", "精度", "fraction",
                 "share", "coverage", "gain", "ratio")

# 已知的荒谬数字（我实际输出过的）
KNOWN_ABSURD = ("5911", "5917", "1869.2", "4411.9", "5032.1", "1580.1", "1919.4")

# 豁免：显式做了钳制或标注的行
SAFE_MARKERS = ("min(", "clip(", "np.minimum", "超预算", "上限", "⚠️", "cap", "不可能",
                "clip", "bound", "saturat", "≤", "<=")

SCAN_DIRS = ("scripts", "src")
SKIP_PARTS = {".probe", "__pycache__", "check_sanity_bounds.py"}


def main() -> int:
    ap = argparse.ArgumentParser(description="检出超过物理上限的增益/比例输出")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    a = ap.parse_args()

    files: list[Path] = []
    for d in SCAN_DIRS:
        p = a.root / d
        if p.exists():
            files += [f for f in p.rglob("*.py") if not any(s in f.parts for s in SKIP_PARTS)]

    absurd_hits, risky_hits = [], []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        # ★ 假阳性修正：跳过模块/函数 docstring 与注释块 ——
        #   本次实测：`k3_budget_correction.py` 的**模块 docstring** 里写了
        #   "`cap = 300` 相对 `cap = B` 有 5911% 的增益"（用于解释这个错误本身），
        #   属于**合法**的历史留痕，不是"仍在用荒谬数字"。
        #   与 `check_stale_numbers.py` 的"留痕"判据一致。
        in_doc = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 跟踪三引号块
            if stripped.startswith('"""') or stripped.startswith("'''"):
                q = stripped[:3]
                if stripped.count(q) == 1:
                    in_doc = not in_doc
                    continue
                if stripped.count(q) >= 2:
                    continue          # 单行 docstring
            if in_doc or stripped.startswith("#"):
                continue
            # (a) 已知荒谬数字（仅在**可执行代码**里才算问题）
            for token in KNOWN_ABSURD:
                if token in line:
                    absurd_hits.append((f, i + 1, token, line.strip()[:100]))
            # (b) 有界语义词 + % 输出，且无安全标记
            if "%" in line and any(w in line for w in BOUNDED_WORDS):
                window = " ".join(lines[max(0, i - 3):i + 4])
                if not any(m in window for m in SAFE_MARKERS):
                    risky_hits.append((f, i + 1, line.strip()[:100]))

    print("=" * 84)
    print("物理上限健全性检查（教训 P23）")
    print("=" * 84)
    print(f"扫描 {len(files)} 个 .py（{'、'.join(SCAN_DIRS)}）")
    if not files:
        print("❌ 扫到 0 个文件 —— 视为未验证")
        return 1

    if absurd_hits:
        print(f"\n❌ 发现 {len(absurd_hits)} 处**已知荒谬数字**（我实际输出过、应已修正）：\n")
        for f, n, tok, s in absurd_hits:
            print(f"   {f.relative_to(a.root)}:{n}   含 '{tok}'")
            print(f"      {s}")
        print("\n修法：确认该量是否违反了约束（如超预算）—— 见 docs/finding-k3-calibration-verdict.md §1。")
        return 1

    if risky_hits:
        print(f"\n⚠️ 发现 {len(risky_hits)} 处**待人工确认**（有界语义 + % 输出，未见显式钳制）：\n")
        for f, n, s in risky_hits[:15]:
            print(f"   {f.relative_to(a.root)}:{n}   {s}")
        if len(risky_hits) > 15:
            print(f"   …另有 {len(risky_hits) - 15} 处")
        print("\n判读：并非全是问题（有界量也可能是 0–100% 内的正常输出）。")
        print("      **需要逐条确认**：该数字是否可能超过 100%？若可能 → 查约束是否被违反。")
        print("      ⚠️ 本项**不判失败**（启发式，误报多）；但**必须人工过一遍**。")
    else:
        print("\n✅ 未发现可疑的有界量输出")

    print("\n" + "=" * 84)
    print("✅ 通过（未发现已知荒谬数字）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
