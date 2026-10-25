"""
yolo26-detection / src/utils/console.py
======================================
Windows 控制台 UTF-8 修复。

为什么需要（本项目实测踩到的坑）：
    本机 Windows 控制台默认代码页是 936 (GBK)。Python 3.12 在 Windows 上
    默认用 **locale 编码**写 stdout，于是任何 emoji（⚠️ ✅ ❌ 🚀）或非 GBK 字符
    都会让 print() 抛 UnicodeEncodeError 并**直接终止训练脚本**。
    （实测：src/train/run.py 打印 "⚠️ tier=smoke…" 时崩在 py3.12 的 gbk 编码器上。）

用法：在任何会 print 中文/emoji 的入口**第一行**调用
    from src.utils.console import setup_console
    setup_console()
"""

from __future__ import annotations

import os
import sys

__all__ = ["setup_console"]

_done = False


def setup_console(force_utf8_env: bool = True) -> None:
    """把 stdout/stderr 切到 UTF-8 且 errors='replace'，失败时静默降级。

    幂等：重复调用无副作用。
    """
    global _done
    if _done:
        return
    _done = True

    # 让子进程（dataloader worker 等）也用 UTF-8
    if force_utf8_env:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("PYTHONUTF8", "1")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            # 不是 TextIOWrapper（例如被重定向到管道且已包装），或平台不支持
            pass

    # Windows 控制台代码页：尽力切到 65001，失败不影响
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
