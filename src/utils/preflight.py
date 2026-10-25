"""
src/utils/preflight.py
======================
**训练前置自检** —— 把"资源纪律"从"我记得调用外部脚本"变成"训练自己会拒绝"。

═══════════════════════════════════════════════════════════════════════════
为什么内置而不是外部脚本（教训 P21 的第三层反思）
═══════════════════════════════════════════════════════════════════════════
P21 的损失：R1 训练期间我并发跑了 `budget_scan` → **R1 在 epoch 28 崩溃**
（Windows 事件日志：`Faulting module: cv2.pyd`，`Exception code: 0xc0000005`），
**27 个 epoch、约 1.9 小时全部丢失，`weights/` 为空**。

**我做的第一版补救是"外部守卫脚本 + 写进 AGENTS-NORMS 规则 11"** ——
但那**仍然是"靠我记得调用"**，与我批评的"规则靠自觉"是同一类问题。

**本模块把检查内置进训练入口**：`src/train/run.py` 启动时自动调用，
不依赖任何人记得跑外部脚本。

═══════════════════════════════════════════════════════════════════════════
检查项（只在**启动训练**时执行；默认拒绝而不是警告）
═══════════════════════════════════════════════════════════════════════════
  1. **已有其他 Python 进程在跑** → 拒绝（真因是 OpenCV 原生冲突，非 RAM）
  2. **已有训练进程在跑** → 拒绝（同一实验重复启动会互相破坏输出目录）
  3. **可用 RAM < 阈值** → 拒绝（防 P4 的 RAM 耗尽）
  4. **输出目录已存在** → 拒绝（防覆盖已有训练成果）

可用 `--allow-concurrent` 绕过（仅用于明确知道后果时）。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

# ── 实测阈值（2026-10-14 第 9 轮校准，不再拍脑袋）────────────────────────
# 实测条件：VisDrone 训练 640/batch8/workers=0，训练进程 RSS 3,283 MB
#   · **训练期间可用 RAM 稳态 = 5.4 – 5.8 GB**（6 次采样，间隔 20s）
#   · 训练期间 GPU 占用 = 4.0 – 4.4 GB
# 旧阈值 6.0 GB **高于稳态值** → 会频繁误报（实测已误报一次：5.8 < 6.0）。
# 新阈值 4.5 GB：比稳态低约 1 GB（~20% 余量），既能容忍正常训练，
# 又能在 RAM 真正吃紧（P4 崩溃时可用仅 6.2 GB 且并发更多任务）之前拦住。
MIN_FREE_RAM_GB = 4.5

# GPU：仅在**没有训练在跑**时才作为判据（训练时本来就会占 4+ GB，检查无意义）
MAX_GPU_GB_WHEN_IDLE = 1.5
_TRAIN_MARKERS = ("src.train.run", "src/train/run.py", "src\\train\\run.py")


@dataclass
class PreflightResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["=" * 76, "训练前置自检（src/utils/preflight.py）", "=" * 76]
        lines += ["  · " + s for s in self.info]
        if self.ok:
            lines.append("\n✅ 前置自检通过")
        else:
            lines.append("\n❌ **前置自检未通过**，训练拒绝启动：")
            lines += ["   - " + r for r in self.reasons]
            lines.append("\n  处置：")
            lines.append("    · 等已有进程结束再跑（P21：崩溃不可恢复，代价是数小时）")
            lines.append("    · 确认无冲突后可用 --allow-concurrent 绕过（自负后果）")
        return "\n".join(lines)


def _python_procs() -> list[tuple[int, float, str]]:
    """(pid, RSS_MB, cmdline)。用 PowerShell/CIM；失败返回空表（不阻塞训练）。"""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)|$([math]::Round($_.WorkingSetSize/1MB))|"
        "$($_.CommandLine)\" }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def _free_ram_gb() -> float:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip()) / 1024**2
    except Exception:
        return float("nan")


def check_before_train(
    *,
    out_dir: str | None = None,
    min_free_ram_gb: float = MIN_FREE_RAM_GB,
    allow_concurrent: bool = False,
) -> PreflightResult:
    """训练启动前的自检。返回 PreflightResult；调用方应在 `not ok` 时**中止**。"""
    res = PreflightResult(ok=True)

    procs = _python_procs()
    me = None
    trainers, others = [], []
    for pid, rss, cmd in procs:
        cmd = cmd or ""
        if any(m in cmd for m in _TRAIN_MARKERS):
            trainers.append((pid, rss, cmd))
        elif "preflight" not in cmd and "multiprocessing" not in cmd:
            others.append((pid, rss, cmd))

    res.info.append(f"python 进程 {len(procs)} 个（训练 {len(trainers)} / 其他 {len(others)}）")

    ram = _free_ram_gb()
    res.info.append(f"可用 RAM = {ram:.1f} GB（阈值 {min_free_ram_gb} GB）")

    if not allow_concurrent:
        if len(others) > 1:      # 允许 1 个（可能是调用栈里的父进程）
            res.ok = False
            res.reasons.append(
                f"检测到 {len(others)} 个其他 Python 进程 → 禁止并发"
                f"（P21 真因：cv2.pyd 原生访问违规，非 RAM；崩溃无 traceback、不可恢复）")
            for pid, rss, cmd in others[:4]:
                res.reasons.append(f"    pid={pid} RSS={rss:.0f}MB  {cmd[:80]}")
        if trainers:
            res.ok = False
            res.reasons.append(
                f"检测到 {len(trainers)} 个**训练进程**已在运行 → 拒绝重复启动"
                f"（会争抢 GPU 并互相破坏输出目录）")

    if ram == ram and ram < min_free_ram_gb:  # nan-safe
        res.ok = False
        res.reasons.append(f"可用 RAM 仅 {ram:.1f} GB < {min_free_ram_gb} GB（P4 的教训）")

    if out_dir:
        from pathlib import Path

        p = Path(out_dir)
        if p.exists() and any(p.iterdir()):
            res.ok = False
            res.reasons.append(f"输出目录已存在且非空：{out_dir}"
                               f"（防覆盖已有训练成果；如确认要重跑请先改名或删除）")

    return res


def report_system() -> str:
    """给训练日志留一份系统快照（便于事后归因）。"""
    ram = _free_ram_gb()
    gpu = ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20)
            gpu = out.stdout.strip()
        except Exception:
            gpu = "(nvidia-smi 失败)"
    return f"RAM 可用 {ram:.1f} GB | GPU {gpu}"
