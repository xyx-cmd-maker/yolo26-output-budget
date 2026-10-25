#!/usr/bin/env python
"""
scripts/check_pdf_items.py —— **按条目清点 PDF 内容**（P106）

═══ 为什么需要 ═══════════════════════════════════════════════════════════
P104 的教训：Figure 3/4/5 的图注**整块消失**，而 `check_pdf_render.py` 只看墨迹——
少三条图注不会让任何一页变成空白，所以它完全没报警。

本检查换一个维度：**不统计像素，而是统计"画上去的文字条目"**。
做法：给 `Page` 打桩，记录每一次 `fig.text()` 的字符串，
然后与源文件里应当出现的条目（每张图/表的图注、每个章节标题）逐一比对。
**任何应出现而未出现的条目 → 失败。**

用法：python scripts/check_pdf_items.py
退出码：0 = 条目齐全；1 = 有缺失
"""
from __future__ import annotations

import importlib.util
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_submission_pdf.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("bp", BUILDER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expected_items(bp) -> dict[str, list[str]]:
    """从源文件解析出**应当出现**的条目。"""
    out: dict[str, list[str]] = {"figure": [], "table": [], "heading": []}
    for fn in ("docs/manuscript-results-methods.md",
               "docs/manuscript-abstract-intro-discussion.md"):
        t = (ROOT / fn).read_text(encoding="utf-8")
        out["figure"] += re.findall(r"\*\*(Figure\s+\d+\.)", t)
        out["table"] += re.findall(r"\*\*(Table\s+\d+\.)", t)
        out["heading"] += [l[3:].strip() for l in t.split("\n") if l.startswith("## ")]
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def main() -> int:
    bp = load_builder()

    # ── 打桩：记录所有画上去的文字 ──────────────────────────────
    drawn: list[str] = []
    _orig_text = None

    class Recorder:
        pass

    import matplotlib.pyplot as plt
    _orig = plt.Figure.text

    def _patched(self, x, y, s, *a, **k):
        drawn.append(str(s))
        return _orig(self, x, y, s, *a, **k)

    plt.Figure.text = _patched
    buf, sys.stdout = sys.stdout, io.StringIO()
    try:
        bp.main()
    finally:
        sys.stdout = buf
        plt.Figure.text = _orig

    all_text = " ".join(drawn)
    exp = expected_items(bp)

    # ★★ P108：**markdown 标记泄漏检查**。
    #   `strip_markdown` 只处理 `# ` 与 `## `，任何 `###` 及更深的标题
    #   会被当成普通段落**原样输出**。
    #   实测：我加的 `### Deployable variant ...` 连三个井号一起印进了 PDF。
    MARKDOWN_LEAK = [
        (r"^#{1,6}\s", "行首井号（标题未被识别）"),
        (r"\*\*", "双星号（粗体未被剥离）"),
        (r"`", "反引号（行内代码未被剥离）"),
        (r"\]\(http", "markdown 链接"),
    ]
    leaks = []
    for pat, why in MARKDOWN_LEAK:
        for s_ in drawn:
            if re.search(pat, str(s_)):
                leaks.append(f"{why}：{str(s_)[:80]!r}")
                break

    print("=" * 96)
    print("PDF 条目清点（按「画上去的文字」而非像素）")
    print("=" * 96)
    print(f"  记录到 {len(drawn)} 段文字")

    missing: list[str] = []
    for kind, label in (("figure", "图注"), ("table", "表注"), ("heading", "章节标题")):
        items = exp[kind]
        miss = [x for x in items if x not in all_text]
        print(f"\n  {label}：应出现 {len(items)} 条，缺失 {len(miss)} 条")
        for x in items:
            mark = "❌" if x in miss else "✅"
            print(f"     {mark} {x}")
        missing += miss

    print()
    if leaks:
        print(f"  ❌ 发现 {len(leaks)} 类 markdown 标记泄漏：")
        for x in leaks:
            print(f"     {x}")
        print("\n  ★ 处置：源文件里的标记未被 strip_markdown 处理，"
              "应改写源文件或扩展 strip_markdown。")
        return 1
    if missing:
        print(f"  ❌ {len(missing)} 条内容未出现在 PDF 中：")
        for x in missing:
            print(f"     {x}")
        print("\n  ★ 处置：检查主循环是否提前 break、是否有页未被 savefig 保存。")
        return 1
    print("  ✅ 全部图注 / 表注 / 章节标题都已出现在 PDF 中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
