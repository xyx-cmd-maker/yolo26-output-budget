"""
scripts/guard_no_concurrent.py
==============================
**资源纪律的机器强制**：在启动任何 GPU / 内存密集任务前，检查是否有训练在跑。

═══════════════════════════════════════════════════════════════════════════
★★ 真因修正（第 8 轮，来自 Windows 事件日志的直接证据）
═══════════════════════════════════════════════════════════════════════════
我最初推断是"RAM 耗尽导致系统杀进程"。**事件日志证明不是**：

    2026/10/14 23:00:36 · Application Error · Event ID 1000
    Faulting application: python.exe
    Faulting module: cv2.pyd            ← **OpenCV 原生库**
    Exception code: 0xc0000005

→ **是 OpenCV 原生库的访问违规。** 我在 R1 训练期间并发跑了 `budget_scan`
  （大量 `cv2` 图像解码），**两个进程同时使用 OpenCV 原生代码 → 崩溃**。

**为什么这个修正很重要**：
  · RAM 耗尽可能只给出可恢复的 `MemoryError`（我在自己的进程里确实看到了）；
  · **原生访问违规直接杀进程、无 traceback、不可恢复**；
  · 所以"**只看 RAM**"的守卫**会漏掉真因** —— 崩溃时 RAM 未必耗尽。

**因此本守卫的判据以"训练期间禁止任何其他 Python 进程"为主**，
RAM / GPU 检查为辅。见 `AGENTS-LESSONS` **P21**。

═══════════════════════════════════════════════════════════════════════════
另有早先的 P4（RAM 耗尽，四个种子全挂）—— 同一类错误，已犯两次
═══════════════════════════════════════════════════════════════════════════

用法：
    # 在启动密集任务前
    python scripts/guard_no_concurrent.py || exit 1

    # 或包在命令前
    python scripts/guard_no_concurrent.py --run "python -m src.eval.budget_scan ..."
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

# ── 实测阈值（2026-10-14 第 9 轮校准，不再拍脑袋）────────────────────────
# 实测（VisDrone 640/batch8/workers=0，训练进程 RSS 3,283 MB）：
#   · 训练期间可用 RAM 稳态 = **5.4 – 5.8 GB**（6 次采样）
#   · 训练期间 GPU 占用 = **4.0 – 4.4 GB**
# 旧阈值 6.0 GB **高于稳态值** → 实测误报 6/6 次。
MIN_FREE_RAM_GB = 4.5              # 比稳态低约 1 GB（~17% 余量）
MAX_GPU_USED_GB_FOR_OTHER_TASKS = 1.5   # **仅在无训练时**作为判据

# ★ 重要的诚实声明：**RAM 判据拦不住 P4**。
#   P4 崩溃时可用 RAM = **6.2 GB**，高于本阈值也高于旧阈值 6.0。
#   真正拦住它的是下面的主判据「训练期间禁止任何并发 Python 进程」。


def free_ram_gb() -> float:
    try:
        import psutil  # 可能未安装

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        pass
    # 回退：用 Windows 的 wmic / PowerShell
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=20)
        kb = float(out.stdout.strip())
        return kb / 1024**2
    except Exception:
        return float("nan")


def gpu_used_gb() -> float:
    if not shutil.which("nvidia-smi"):
        return float("nan")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return float("nan")


def python_procs() -> list[tuple[int, float, str]]:
    """返回 (pid, RSS_MB, 命令行片段)。用 PowerShell 取命令行。"""
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


def main() -> int:
    ap = argparse.ArgumentParser(description="资源纪律守卫：禁止与训练并发")
    ap.add_argument("--min-free-ram-gb", type=float, default=MIN_FREE_RAM_GB)
    ap.add_argument("--max-gpu-gb", type=float, default=MAX_GPU_USED_GB_FOR_OTHER_TASKS)
    ap.add_argument("--max-other-procs", type=int, default=1,
                    help="无训练时，允许的其他 Python 进程上限（默认 1 = 只有本守卫自己）")
    ap.add_argument("--run", default=None, help="守卫通过后要执行的命令")
    ap.add_argument("--allow-training", action="store_true",
                    help="用于**启动训练本身**时绕过'有训练在跑'这一条")
    a = ap.parse_args()

    reasons: list[str] = []
    info: list[str] = []

    ram = free_ram_gb()
    info.append(f"可用 RAM = {ram:.1f} GB（阈值 {a.min_free_ram_gb} GB）")
    if ram == ram and ram < a.min_free_ram_gb:  # nan-safe
        reasons.append(f"可用 RAM 仅 {ram:.1f} GB < {a.min_free_ram_gb} GB")

    gpu = gpu_used_gb()
    # ★ 校准（第 9 轮）：GPU 判据**只在没有训练时**有意义 ——
    #   训练本来就会占 4.0–4.4 GB，训练期间的 GPU 检查纯属噪声。
    has_trainer = any("src.train" in (p[2] or "") for p in python_procs())
    if has_trainer:
        info.append(f"GPU 已用 = {gpu:.2f} GB（训练在跑，此项不作判据）")
    else:
        info.append(f"GPU 已用 = {gpu:.2f} GB（空闲态上限 {a.max_gpu_gb} GB）")
        if gpu == gpu and gpu > a.max_gpu_gb:
            reasons.append(f"无训练但 GPU 已用 {gpu:.2f} GB > {a.max_gpu_gb} GB"
                           f"（疑似残留进程占着显存——**这比 RAM 更容易被忽略**）")

    procs = python_procs()
    trainers = [p for p in procs if "src.train.run" in (p[2] or "") or "src.train" in (p[2] or "")]
    others = [p for p in procs if p not in trainers and "guard_no_concurrent" not in (p[2] or "")]
    info.append(f"python 进程 = {len(procs)} 个，其中训练进程 = {len(trainers)} 个，"
                f"其他 = {len(others)} 个")
    # ★★ 主判据（第 8 轮修正）：真因是 OpenCV 原生崩溃，不是 RAM。
    #   所以只要**有训练在跑**，就拒绝**任何**其他 Python 进程 —— 不管内存多宽裕。
    if trainers and not a.allow_training:
        reasons.append(
            f"检测到 {len(trainers)} 个训练进程在跑 → **禁止任何并发 Python 进程**"
            f"（P21：真因是 cv2.pyd 访问违规，非 RAM；崩溃不可恢复）")
        for pid, rss, cmd in others[:5]:
            reasons.append(f"    其他进程 pid={pid} RSS={rss:.0f}MB  {(cmd or '')[:90]}")
    if others and not trainers and len(others) > a.max_other_procs:
        reasons.append(f"已有 {len(others)} 个其他 Python 进程 > {a.max_other_procs}"
                       f"（会争抢 RAM 与 OpenCV 原生资源）")

    print("=" * 76)
    print("资源纪律守卫（AGENTS-LESSONS P4 / P21 的机器强制）")
    print("=" * 76)
    for line in info:
        print("  · " + line)

    if reasons:
        print("\n❌ **拒绝执行**，原因：")
        for r in reasons:
            print("   - " + r)
        print("\n  处置：")
        print("    · 若确有训练在跑 → **等它结束**，不要并发（P21 的代价是 27 个 epoch 全丢）")
        print("    · 若只是想跑小任务 → 先用 --limit 参数把规模压到安全范围")
        print("    · 若确认可安全执行 → 手工复核后加 --allow-training（仅启动训练时用）")
        return 1

    print("\n✅ 守卫通过，可以执行")
    if a.run:
        print(f"   执行：{a.run}")
        rc = subprocess.call(a.run, shell=True)
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
