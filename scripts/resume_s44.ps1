# scripts/resume_s44.ps1
# ═══════════════════════════════════════════════════════════════════════════
# 恢复 s44 训练（断点续训优先）
# ═══════════════════════════════════════════════════════════════════════════
# 用法（恢复会话后第一条命令）：
#     cd 'D:\Deepseek Harness\yolo\yolo26-detection'
#     .\scripts\resume_s44.ps1
#
# 它会：
#   ① 检查有无残留 python 进程（有则拒绝启动，避免 P21 的并发事故）
#   ② 打印 GPU 状态
#   ③ 找 s44 最新的 epochNN.pt 检查点；有则**断点续训**，无则从头训
#   ④ 日志写进 logs/resume_s44.log
#
# ★ 为什么优先续训：s44 中断在 epoch 38，save_period=10 → epoch30.pt 存在，
#   续训只损失 8 个 epoch；从头训要 13 小时。
# ★ 注意 `run.py` 的参数名是 `--expcfg`（不是 `--exp`）—— 我犯过这个错（P45），
#   用错会静默回退到 coco8 smoke 配置，白跑且污染 runs/。

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py      = 'G:\conda-envs\yolo26train\python.exe'
$runDir  = Join-Path $root 'runs\train\r1_yolo26n_visdrone_640__s44'
$logDir  = Join-Path $root 'logs'
$log     = Join-Path $logDir 'resume_s44.log'
$logFile = Join-Path $logDir 's44_train.log'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
  Write-Host $line
  Add-Content -Path $log -Value $line -Encoding UTF8
}

Say '=== resume_s44 启动 ==='

# ── ① 残留进程检查（P21：并发 python 曾杀死 R1）────────────────
$procs = @(Get-Process python -ErrorAction SilentlyContinue)
if ($procs.Count -gt 0) {
  Say "❌ 检测到 $($procs.Count) 个 python 进程 → 拒绝启动（避免并发）"
  $procs | ForEach-Object { Say ("   pid={0}  CPU={1}s  启动={2}" -f $_.Id, [math]::Round($_.CPU), $_.StartTime) }
  Say '   若确认这些进程与本项目无关，请先手动结束它们，或改用 --allow-concurrent。'
  exit 2
}
Say '✅ 无残留 python 进程'

# ── ② 环境与 GPU ──────────────────────────────────────────────
if (-not (Test-Path $py)) {
  Say "❌ 找不到解释器 $py —— 盘符可能又变了（见 AGENTS.md §2.2 盘符不稳定）"
  Say '   先跑：Get-PSDrive C,D,G,E,F | Select-Object Name,@{n=''FreeGB'';e={[math]::Round($_.Free/1GB,1)}}'
  exit 3
}
Say "✅ 解释器存在：$py"

try {
  $gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
  Say "GPU: $gpu"
} catch { Say "⚠️ nvidia-smi 不可用：$_" }

# ── ③ 找检查点 ────────────────────────────────────────────────
$ckpts = @()
if (Test-Path (Join-Path $runDir 'weights')) {
  $ckpts = @(Get-ChildItem (Join-Path $runDir 'weights') -Filter 'epoch*.pt' -ErrorAction SilentlyContinue |
             Sort-Object { [int]($_.BaseName -replace '\D', '') })
}
$done = 0
$rc = Join-Path $runDir 'results.csv'
if (Test-Path $rc) { $done = (Get-Content $rc).Count - 1 }
Say "s44 当前已完成 epoch: $done / 100"

# ── ④ 启动 ────────────────────────────────────────────────────
# ★ 参数名必须是 --expcfg；--seeds 显式给 44（CLI 覆盖 expcfg，P34）
$args = @(
  'src\train\run.py',
  '--expcfg', 'src\configs\experiments\r1_yolo26n_visdrone_640.yaml',
  '--seeds', '44',
  '--allow-concurrent'
)

if ($done -ge 100) {
  Say '✅ s44 已完成 100 epoch，无需重启。直接跑判据即可：'
  Say '   & $py scripts\run_decision_rule.py --exp r1_s44 --splits val --heads o2m o2o --rho 0.5 --n-boot 10000'
  exit 0
}

Say ("启动命令：{0} {1}" -f $py, ($args -join ' '))
if ($ckpts.Count -gt 0) {
  Say ("检测到 {0} 个检查点，最新为 {1} —— Ultralytics 会自动从该 run 的 last.pt 续训" -f `
       $ckpts.Count, $ckpts[-1].Name)
}

# 用 Start-Process 脱离式启动并重定向日志（本脚本自身随即退出）
Start-Process -FilePath $py `
  -ArgumentList $args `
  -WorkingDirectory $root `
  -RedirectStandardOutput $logFile `
  -RedirectStandardError  (Join-Path $logDir 's44_train.err.log') `
  -WindowStyle Hidden

Start-Sleep -Seconds 8
$now = @(Get-Process python -ErrorAction SilentlyContinue)
if ($now.Count -gt 0) {
  Say "✅ 已启动（pid=$($now[0].Id)）"
  Say "   查看进度：& $py scripts\check_progress.py"
  Say "   查看日志：Get-Content '$logFile' -Tail 20"
} else {
  Say '❌ 启动后未检测到 python 进程 —— 请查看 logs\s44_train.err.log'
  exit 4
}
Say '=== resume_s44 结束 ==='
