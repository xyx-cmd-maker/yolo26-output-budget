# test-dev 唯一一次评测（G-1 护栏）—— 两条头 × 11 档预算，chunk=150
#
# 官方纪律：`test` **只在定稿时跑一次**，且**不得用于挑 checkpoint 或调超参**。
# 本脚本只做**评测**，不动权重、不选 epoch。
#
# ⚠️ 规模：test-dev = **1,610 图**（val 548 的 2.94 倍）。
#    val 的单头扫描约 33 分钟 → 本脚本预计 **3–4.5 小时**。
#    因此**必须由计划任务脱离会话运行**（见 scripts\run_testdev_launcher.cmd）。
#
# ⚠️ 历史坑（本项目亲历）：
#   · 后台作业在本环境约 **5 分钟上限**，会被连坐杀掉 → 必须脱离进程树；
#   · `.ps1` 必须存成 **UTF-8 with BOM**，否则中文按 GBK 误读、语法报错；
#   · PowerShell **变量名不区分大小写**：`$py` 会覆盖 `$PY`（P68）；
#   · **`$args` 是自动变量，不可赋值**（P68）；
#   · `Set-Location` **不改变 .NET 的 CWD**，`[IO.File]::` 只认绝对路径；
#   · **PowerShell 5.1 没有三元 `?:`**。

$ErrorActionPreference = 'Continue'
$proj = 'D:\Deepseek Harness\yolo\yolo26-detection'
Set-Location $proj
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$PyExe = 'G:\conda-envs\yolo26train\python.exe'   # ★ 不叫 $py（会与 $PY 同名）
$Log = Join-Path $proj 'logs\testdev_run.log'
New-Item -ItemType Directory -Force -Path (Join-Path $proj 'logs') | Out-Null

function Say($m) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

Say '════ test-dev 唯一一次评测（G-1 护栏）开始 ════'

# ── 前置检查 ────────────────────────────────────────────────────────
if (-not (Test-Path $PyExe)) { Say "❌ 找不到解释器 $PyExe（G: 盘可能未挂载）"; exit 2 }
Say "✅ 解释器在位"

$busy = Get-Process python -ErrorAction SilentlyContinue
if ($busy) {
    $n = ($busy | Measure-Object).Count
    Say "⚠️ 检测到 $n 个 python 进程；并发会让 OpenCV 原生崩溃（P21）。等待 60 秒后复查…"
    Start-Sleep -Seconds 60
    $busy2 = Get-Process python -ErrorAction SilentlyContinue
    if ($busy2) { Say "❌ 仍有 python 进程，中止"; exit 3 }
}
Say "✅ 无其他 python 进程"

# ★ 权重：**用已固定的 best.pt，不得重新选择**（test 不得用于挑 checkpoint）
$Ckpt = Join-Path $proj 'runs\train\r1_yolo26n_visdrone_640__s42\weights\best.pt'
if (-not (Test-Path $Ckpt)) { Say "❌ 找不到 s42 的 best.pt: $Ckpt"; exit 4 }
Say "✅ 权重 = $Ckpt（s42 best.pt，已固定，不重选）"

# ★ 记录环境指纹与权重哈希（可复现性）
$hash = (Get-FileHash $Ckpt -Algorithm SHA256).Hash
Say "   best.pt SHA256 = $hash"
Add-Content -Path $Log -Value "   best.pt SHA256 = $hash" -Encoding UTF8

$Budgets = @(10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000)
$TestDir = Join-Path $proj 'data\processed\visdrone\labels\test'
$nTest = (Get-ChildItem $TestDir -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count
Say "   test-dev 标签数 = $nTest（官方 1,610）"
if ($nTest -ne 1610) { Say "⚠️ 标签数不等于 1,610，请核对 data yaml 的 test 指向" }

# ── 逐头扫描（chunk=150，避免 CUDA OOM）────────────────────────────
foreach ($head in @('o2m', 'o2o')) {
    Say "── head = $head 开始 ──"
    $scanArgs = @(
        'scripts\budget_scan_chunked.py',
        '--exp', 'r1_testdev',
        '--head', $head,
        '--split', 'test',
        '--chunk', '150',
        '--weights', $Ckpt,
        '--budgets'
    ) + ($Budgets | ForEach-Object { "$_" })

    & $PyExe @scanArgs 2>&1 | Tee-Object -FilePath $Log -Append
    $rc = $LASTEXITCODE
    Say "   head=$head 退出码 = $rc"
    if ($rc -ne 0) { Say "❌ head=$head 扫描失败，中止"; exit 5 }
}
Say '── 两条头扫描完成 ──'

# ── 重算（把 test-dev 纳入对账脚本的第 8 项检查）──────────────────
Say '── 重算手稿数字 ──'
& $PyExe 'scripts\recompute_manuscript_numbers.py' 2>&1 | Add-Content -Path $Log -Encoding UTF8
Say "   重算退出码 = $LASTEXITCODE"

Say '── 主张对账（含第 8 项：test-dev）──'
& $PyExe 'scripts\audit_claims_vs_data.py' 2>&1 | Add-Content -Path $Log -Encoding UTF8
Say "   对账退出码 = $LASTEXITCODE"

Say '════ 完成：请查看 logs\testdev_run.log ════'
Say '下一步（人工）：核对 test-dev 的 Δ 是否与 val 同号；若不同号，**必须如实报告**，不得用 val 结论覆盖（G-1 护栏）。'
