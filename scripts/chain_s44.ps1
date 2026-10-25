<#
.SYNOPSIS
  等 s43 结束后自动启动 s44（多种子补全，消解"单种子"这一最致命缺口）

.DESCRIPTION
  为什么需要它：每种子的 100 epoch 训练约 6.6 小时，而 __main__ 会话无法一直等着。
  本脚本以**后台作业**形式运行：轮询 s43 的 python 进程，退出后启动 s44。

  安全护栏（本项目已因并发崩溃丢过 27 个 epoch，见 AGENTS-LESSONS **P21**）：
    1. 启动 s44 前确认**没有任何 python 进程**（除本脚本自身）
    2. 确认 s44 的输出目录**不存在**（防覆盖；`run.py` 的 preflight 也会拦）
    3. s44 用**修正后**的 `run.py`（`--seeds` 优先于 expcfg，见 **P34**）
    4. 全程**串行**，绝不并发

  用法（作为后台作业）：
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/chain_s44.ps1
#>

$ErrorActionPreference = 'Stop'
$proj = 'D:\Deepseek Harness\yolo\yolo26-detection'
$py = 'G:\conda-envs\yolo26train\python.exe'
$log = Join-Path $proj 'logs\chain_s44.log'

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Log "=== 串接脚本启动：等待 s43 结束，然后启动 s44 ==="

# ── ① 等 s43 结束 ─────────────────────────────────────────────
$waited = 0
while ($true) {
    $procs = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited += 60
    if ($waited % 1800 -eq 0) {
        $cpus = ($procs | ForEach-Object { [math]::Round($_.CPU) }) -join ','
        Log "  仍在运行：$($procs.Count) 个 python 进程（CPU 秒 $cpus），已等 $([math]::Round($waited/60)) 分钟"
    }
}
Log "s43 已结束（等了 $([math]::Round($waited/60)) 分钟）"

# ── ② 护栏：确认无残留 + s44 目录不存在 ──────────────────────
Start-Sleep -Seconds 30
$left = @(Get-Process python -ErrorAction SilentlyContinue)
if ($left.Count -gt 0) {
    Log "❌ 仍有 $($left.Count) 个 python 进程残留，**拒绝启动 s44**（P21 教训）"
    exit 1
}
$s44dir = Join-Path $proj 'runs\train\r1_yolo26n_visdrone_640__s44'
if (Test-Path $s44dir) {
    Log "❌ s44 目录已存在：$s44dir —— **拒绝启动**（防覆盖）"
    exit 1
}

# ── ③ 核对 s43 是否真的产出了权重 ────────────────────────────
$s43best = Join-Path $proj 'runs\train\r1_yolo26n_visdrone_640__s43\weights\best.pt'
if (-not (Test-Path $s43best)) {
    Log "⚠️ s43 的 best.pt 不存在 —— s43 可能失败。**仍启动 s44**（多种子需要它），但请在结果里核查 s43"
} else {
    Log "✅ s43 best.pt 存在（$([math]::Round((Get-Item $s43best).Length/1MB,1)) MB）"
}

# ── ④ 启动 s44 ────────────────────────────────────────────────
Log "启动 s44 …"
$out = Join-Path $proj 'logs\r1_s44.log'
$err = Join-Path $proj 'logs\r1_s44.err.log'
$p = Start-Process -FilePath $py `
    -ArgumentList '-m','src.train.run','--expcfg','src/configs/experiments/r1_yolo26n_visdrone_640.yaml','--seeds','44','--allow-concurrent' `
    -WorkingDirectory $proj `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -PassThru
Log "s44 已启动，pid=$($p.Id)"

# ── ⑤ 确认它真的在跑（60 秒后）──────────────────────────────
Start-Sleep -Seconds 60
$now = @(Get-Process python -ErrorAction SilentlyContinue)
if ($now.Count -eq 0) {
    Log "❌ s44 启动后 60 秒内退出 —— 见 $err"
    if (Test-Path $err) { Get-Content $err -Encoding UTF8 -Tail 20 | ForEach-Object { Log "  ERR: $_" } }
    exit 1
}
Log "✅ s44 正在运行（$($now.Count) 个 python 进程）"
Log "=== 串接完成，本脚本退出 ==="
