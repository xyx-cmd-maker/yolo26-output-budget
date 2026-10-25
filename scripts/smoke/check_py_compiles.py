"""
scripts/smoke/check_py_compiles.py
=================================
**写文件后立刻编译检查** —— 把"引号嵌套 / 语法错误"在进入下一步之前拦住。

═══════════════════════════════════════════════════════════════════════════
为什么需要（我犯过 6 次同类错误）
═══════════════════════════════════════════════════════════════════════════
我在中文 `print("...")` / 文档字符串里嵌英文双引号，反复造成 `SyntaxError`：
    `print("  → 与"分配是否有意义"的排序一致")`
**每次都靠跑脚本才发现，浪费了一轮工具调用。**
已有 `check_quote_nesting.py` 抓这一类，但它是**事后**的、且只抓特定模式。

**本校验器更彻底**：对**所有** `scripts/**/*.py` 与 `src/**/*.py` 与 `tests/*.py`
做真正的 `compile()`，**任何语法错误立即报出并给行号**。
→ 于是"写完就编译"成为**机器动作**，不依赖我的记性。

用法：
    python scripts/smoke/check_py_compiles.py            # 全量
    python scripts/smoke/check_py_compiles.py f1.py f2.py  # 只查指定文件
退出码：0 = 全部可编译；1 = 有语法错误。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".probe", "runs", "data", "weights", "__pycache__",
             "node_modules", "ultralytics-main"}


def collect(argv: list[str]) -> list[Path]:
    if argv:
        out = []
        for a in argv:
            p = Path(a)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.is_file() and p.suffix == ".py":
                out.append(p)
            elif p.is_dir():
                out += [f for f in p.rglob("*.py")
                        if not any(s in f.parts for s in SKIP_DIRS)]
        return out
    files: list[Path] = []
    for d in ("scripts", "src", "tests"):
        root = PROJECT_ROOT / d
        if root.exists():
            files += [f for f in root.rglob("*.py")
                      if not any(s in f.parts for s in SKIP_DIRS)]
    return files


def main() -> int:
    files = collect(sys.argv[1:])
    if not files:
        print("❌ 没找到任何 .py —— 视为未验证")
        return 1

    bad: list[tuple[Path, SyntaxError]] = []
    bom_ok = 0
    for f in files:
        try:
            raw = f.read_bytes()
        except OSError as e:
            print(f"  ⚠️ 读不了 {f.relative_to(PROJECT_ROOT)}：{e}")
            continue
        # ★ 关键：**必须对 bytes 调 compile()**。
        #   带 UTF-8 BOM（EF BB BF）的 .py 是**合法**的（PEP 263），Python 3 能直接 import。
        #   但若先 `read_text(encoding="utf-8")` 再 compile，BOM 会变成源码首字符
        #   → 误报 `invalid non-printable character U+FEFF`（**实测确证**，见 P40）。
        #   compile(bytes) 会自行处理编码声明与 BOM。
        if raw[:3] == b"\xef\xbb\xbf":
            bom_ok += 1
        try:
            compile(raw, str(f), "exec")
        except SyntaxError as e:
            bad.append((f, e))
        except ValueError as e:
            # 源文件含 NUL 等；报出来但不算语法错误
            print(f"  ⚠️ {f.relative_to(PROJECT_ROOT)}：{e}")

    print(f"编译检查：{len(files)} 个 .py（其中 {bom_ok} 个带 UTF-8 BOM —— 合法，见 P40）")
    if bad:
        print(f"\n❌ **{len(bad)} 个文件有语法错误**：")
        for f, e in bad:
            rel = f.relative_to(PROJECT_ROOT)
            print(f"\n   {rel}:{e.lineno}  {e.msg}")
            if e.text:
                print(f"      {e.text.rstrip()}")
                if e.offset:
                    print(f"      {' ' * (e.offset - 1)}^")
        print("""
常见原因（我犯过 6 次）：**在中文双引号字符串里再嵌英文双引号**。
    ✗  print("  → 与"分配是否有意义"的排序一致")
    ✓  print("  → 与「分配是否有意义」的排序一致")
或改用 f-string 的另一种引号 / 三引号。
""")
        return 1
    print(f"\n✅ 全部 {len(files)} 个文件语法正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
