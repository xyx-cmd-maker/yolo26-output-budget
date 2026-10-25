"""
scripts/smoke/check_script_entrypoints.py
=========================================
**机器校验器**：`scripts/` 与 `src/` 下的可执行脚本必须有 `if __name__ == "__main__"` 入口。

═══════════════════════════════════════════════════════════════════════════
为什么存在（第 32 轮发现）
═══════════════════════════════════════════════════════════════════════════
做交接文档时我写了一批"可直接复制执行"的命令，于是写了个核验器查这些命令的参数是否真实存在。
结果查出**两类问题**：
  1. `resolution_probe.py` 的参数是 `--combos imgsz:batch`，**我却写了 `--imgsz/--batch`**
  2. **`waterfill_dp_verify.py` 与 `prereg_decision_rule.py` 把整段逻辑写在模块层**
     → 导入即执行、无入口、不符合项目其它脚本的一致约定

问题 2 是一条**可维护的通用规则**，所以固化为校验器。

判据
----
凡满足以下任一条的文件，必须有 `__name__ == "__main__"`：
  · 路径在 `scripts/` 下
  · 路径以 `src/eval/` 开头
  · 文件末尾 3 行内出现 `print(` 或 `sys.exit(`（即模块层"会执行"）

豁免：`__init__.py`、`_TEMPLATE`、以及显式的 `EXEMPT` 集合（须逐条给理由）。

退出码：0 = 全部合规；1 = 有缺失。
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SKIP_DIRS = {".git", ".probe", "runs", "data", "weights", "__pycache__",
             "node_modules", "ultralytics-main"}

# 豁免必须逐条给理由（与 check_no_mojibake 的纪律一致）
EXEMPT: dict[str, str] = {
    # 纯库模块：只提供函数/常量，模块层不执行任何东西
    "src/eval/proxy_density.py": "库模块：只定义函数，模块层无副作用",
    "src/eval/stats.py": "库模块：只定义函数，模块层无副作用",
    "src/utils/preflight.py": "库模块：由 run.py 导入调用",
    "src/utils/console.py": "库模块：被导入即生效的 console 设置（有意为之）",
    "src/utils/results_io.py": "库模块：只定义函数",
    "src/utils/repro.py": "库模块：只定义函数",
    "src/utils/data_card.py": "库模块：只定义函数",
    "src/datasets/temporal_split.py": "库模块：只定义函数",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="检查脚本是否有 __main__ 入口")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    a = ap.parse_args()

    files: list[Path] = []
    for d in ("scripts", "src"):
        p = a.root / d
        if p.exists():
            files += [f for f in p.rglob("*.py") if not any(s in f.parts for s in SKIP_DIRS)]

    missing, exempted, ok = [], [], 0
    for f in files:
        rel = f.relative_to(a.root).as_posix()
        if f.name == "__init__.py" or "_TEMPLATE" in f.name:
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        has_main = any('__name__ == "__main__"' in ln for ln in lines)
        # 是否需要入口
        tail = "\n".join(lines[-3:])
        needs = (rel.startswith("scripts/") or rel.startswith("src/eval/")
                 or "print(" in tail or "sys.exit(" in tail)
        if not needs:
            continue
        if has_main:
            ok += 1
        elif rel in EXEMPT:
            exempted.append((rel, EXEMPT[rel]))
        else:
            missing.append(rel)

    print(f"扫描 {len(files)} 个 .py（{'、'.join(['scripts', 'src'])}）")
    print(f"判据：scripts/ 或 src/eval/ 下的文件、或末尾有 print()/sys.exit() 的文件，"
          f"必须有 __main__ 入口")
    if not files:
        print("❌ 扫到 0 个文件 —— 视为未验证")
        return 1

    if exempted:
        print(f"\n豁免 {len(exempted)} 个（逐条给理由）：")
        for rel, why in exempted:
            print(f"   · {rel}  —— {why}")

    if missing:
        print(f"\n❌ **{len(missing)} 个文件缺 `if __name__ == \"__main__\"` 入口**：")
        for rel in missing:
            print(f"   · {rel}")
        print("""
修法：把模块层代码包进 `def main() -> int:`，并在文件末尾加
    if __name__ == "__main__":
        raise SystemExit(main())
理由：模块层直接执行会导致**导入即运行**（无法被测试或复用），
      且与项目其它脚本的约定不一致。
""")
        return 1

    print(f"\n✅ {ok} 个需要入口的文件全部合规")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
