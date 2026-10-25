"""
scripts/smoke/check_quote_nesting.py
===================================
**机器校验器**：检出 Python 里"双引号字符串中又嵌套中文引号"的语法错误。

═══════════════════════════════════════════════════════════════════════════
为什么必须存在（教训 P23 —— 我在同一类错误上犯了 **10 次**）
═══════════════════════════════════════════════════════════════════════════
`print("…这是"关键"的…")` —— 中文引号 `"` `"` 与 ASCII 双引号 `"` 视觉相近，
我在写中文注释/输出时反复把中文引号打进 ASCII 字符串里，导致 `SyntaxError`。

**已发生 10 次**（grep 记录见 `AGENTS-LESSONS.md` P23），每次都靠 `py_compile` 报错才发现，
每次都要重新定位—修改—重编译。**用校验器可以在写完后一次扫出全部。**

判据（正则）：在**以 `print(` 或 `f"` 或 `= "` 开头的行**里，
出现 `中文 + ASCII双引号 + 中文`（即中文引号夹在 ASCII 双引号串中间）。
排除三引号块与纯注释行。

★ 本文件自己在写上面这句时就踩了一次：docstring 里直接写三个 ASCII 双引号
  **会提前终止 docstring**（语法错误）。所以此处改用文字描述。
  这恰好印证了这一类错误的普遍性。见 `AGENTS-LESSONS.md` **P23**。

退出码：0 = 干净；1 = 发现疑似嵌套。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 「中文字符 + ASCII 双引号 + 中文字符」—— 正常代码里几乎不可能出现
NEST_RE = re.compile(r"[\u4e00-\u9fff][\"][\u4e00-\u9fff]")
# 也检反向： ASCII 引号出现在中文串中间且两侧都是中文
SKIP_DIRS = {".git", ".probe", "runs", "data", "weights", "__pycache__", "node_modules",
             "ultralytics-main"}


def main() -> int:
    ap = argparse.ArgumentParser(description="检出 Python 中文字符串里的引号嵌套")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    a = ap.parse_args()

    py = [p for p in a.root.rglob("*.py")
          if not any(part in SKIP_DIRS for part in p.parts)]
    hits = []
    for p in py:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        in_triple = False
        for n, line in enumerate(lines, 1):
            # 跟踪三引号块（粗略但够用）
            cnt = line.count('"""')
            if in_triple:
                if cnt:
                    in_triple = False
                continue
            if cnt % 2 == 1:
                in_triple = True
                continue
            s = line.strip()
            if s.startswith("#"):
                continue
            # ★ 假阳性修正 1（实测 2026-10-14）：**单行 docstring** 里的中文引号是合法的。
            #   例："""每图恰好 B 个输出（不含"用不满"机制）时的最坏图召回。"""
            #   该行以三引号开头且以三引号结尾 → 整行都在字符串里 → 跳过。
            if (s.startswith('"""') and s.endswith('"""') and len(s) >= 6) or \
               (s.startswith("'''") and s.endswith("'''") and len(s) >= 6):
                continue
            # 行尾的三引号闭合（多行 docstring 的最后一行）
            if s.endswith('"""') or s.endswith("'''"):
                continue
            # ★ 假阳性修正 2：剥掉行内注释再判（注释里的中文引号合法）。
            #   只有**字符串字面量**里的嵌套才会导致 SyntaxError。
            #   简化：取第一个 `#` 之前的部分。字符串内含 `#` 的情形极罕见，可接受。
            code = s.split("#", 1)[0]
            if NEST_RE.search(code):
                hits.append((p, n, s[:110]))

    print(f"扫描 {len(py)} 个 .py 文件（跳过 {sorted(SKIP_DIRS)}）")
    print(f"判据：ASCII 双引号两侧均为中文字符 —— {NEST_RE.pattern}")
    if not py:
        print("❌ 扫到 0 个文件 —— 视为未验证")
        return 1
    if hits:
        print(f"\n❌ 发现 {len(hits)} 处疑似引号嵌套（会导致 SyntaxError）：\n")
        for p, n, s in hits:
            print(f"   {p.relative_to(a.root)}:{n}")
            print(f"      {s}")
        print("\n修法：把中文引号 \" \" 换成 「 」（推荐），或改用单引号包字符串。")
        return 1
    print("\n✅ 未发现引号嵌套")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
