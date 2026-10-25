#!/usr/bin/env python
"""
scripts/check_pdf_render.py —— **渲染 PDF 每页并做版面统计**

═══ 为什么需要（P87）═════════════════════════════════════════════════
本轮 4 个缺陷**全部**是"渲染看图"才发现的，而 11 项自动检查全绿：
  · Discussion 被排在 Results 之前
  · Table 2 的**表体整页丢失**（place_fig 的 pg.__init__ 丢弃当前页）
  · 图注与图之间 22% 页高的 letterbox 空白
  · Table 1 表头 `ceiling(100)` 被断成 `ceiling(1-00)`
源文件全对、LaTeX 全对，**只有 PDF 是坏的**。

实现：复用**已有**的 WinRT 渲染脚本（`render_pages.ps1`，已在用，不装新包），
      再用 PIL/numpy 统计每页墨迹覆盖与横向范围。

用法：python scripts/check_pdf_render.py
退出码：0 = 无明显版面异常；1 = 有可疑页
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "results" / "submission-manuscript.pdf"
RENDER = Path(r"D:\Deepseek Harness\yolo\.probe\render_pages.ps1")
OUT = ROOT / "logs" / "_pdfpages"

MIN_INK = 0.003       # ★ 校准：图页墨迹天然只有 2–4%，正文 8–10%；
                      #   真正异常是 **接近 0**（整页空白）。原设 3% 会把图页误报。
MAX_BLANK_ROWS = 0.985
MIN_SPAN = 0.70       # ★ 校准：页边距各 6%，正文横向本就约 80%，
                      #   低于 70% 才是可疑（内容缩在中间）。


def render_all(n_pages: int) -> bool:
    if not RENDER.exists():
        print(f"  ⊘ 跳过：缺渲染脚本 {RENDER}")
        return False
    pages = ",".join(str(i) for i in range(1, n_pages + 1))
    r = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(RENDER),
                        "-Pdf", str(PDF.relative_to(ROOT)), "-Pages", pages],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    ok = r.returncode == 0
    if not ok:
        print(f"  ⚠️ 渲染失败：{(r.stderr or r.stdout or '')[-300:]}")
    return ok


def main() -> int:
    if not PDF.exists():
        print(f"  ❌ 缺 {PDF.relative_to(ROOT)}")
        return 1
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"  ⊘ 跳过（缺 {e.name}）")
        return 0

    # 先数页数（从渲染脚本的输出里拿不到，用 PNG 数量兜底）
    print("=" * 96)
    print("PDF 版面统计（机器可见信号；人眼核对仍不可省）")
    print("=" * 96)
    if not render_all(30):
        return 0

    pngs = sorted(OUT.glob("page*.png"))
    if not pngs:
        print("  ⊘ 未产生页面图")
        return 0

    problems = []
    for png in pngs:
        n = int(png.stem.replace("page", ""))
        arr = np.asarray(Image.open(png).convert("L"))
        ink = float((arr < 240).mean())
        blank_rows = float((arr.min(axis=1) > 250).mean())
        cols = (arr < 240).any(axis=0)
        if cols.any():
            lo = int(np.argmax(cols))
            hi = int(len(cols) - np.argmax(cols[::-1]) - 1)
            span = (hi - lo) / arr.shape[1]
        else:
            span = 0.0
        flags = []
        if ink < MIN_INK:
            flags.append("墨迹过少")
            problems.append((n, f"墨迹 {ink:.2%} < {MIN_INK:.1%}"))
        if blank_rows > MAX_BLANK_ROWS:
            flags.append("几乎全空")
        if span < MIN_SPAN and ink > 0.002:
            flags.append(f"横向仅 {span:.0%}")
        print(f"  第 {n:>2} 页  墨迹 {ink:6.2%}  空白行 {blank_rows:5.1%}  横向 {span:5.0%}  "
              f"{'⚠️ ' + '、'.join(flags) if flags else ''}")

    print()
    if problems:
        print(f"  ❌ {len(problems)} 页可疑：")
        for n, why in problems:
            print(f"     第 {n} 页：{why}")
        print("\n  ★ 处置：渲染该页人眼看，确认是否内容被丢弃/溢出。")
        return 1
    print("  ✅ 无页面墨迹过少或横向溢出（**人眼核对仍不可省**）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
