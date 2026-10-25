# ══════════════════════════════════════════════════════════════════════════════
# scripts/run_s44_scan.ps1 —— s44 训练完成后的**判据扫描**（一键，后台）
# ══════════════════════════════════════════════════════════════════════════════
# 用法（在项目根目录）：
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s44_scan.ps1
#
# ★ 为什么必须显式传 --budgets：
#   `budget_scan_chunked.py` 的默认 BUDGETS 只有 6 档 [10,20,30,50,100,300]，
#   而现有 val 曲线（runs/budget/r1_val_*.json）用的是 **11 档**
#       [10,20,30,50,70,100,150,200,300,500,1000]
#   若用默认值，s44 的曲线与 s42/s43 **不可比** → 跨种子对照全部作废。
#   本脚本把 11 档写死在参数里，避免遗漏。
#
# ★ 为什么 chunk=150：
#   `--limit 548 --budgets ... 500` 在本机曾触发 CUDA OOM（需 2.51 GiB，总 7.96 GiB）。
#   150 图/块是本机实测的安全值。
#
# ★ 前置条件（脚本会检查）：
#   ① 没有 python 训练进程在跑（否则抢显存，见 P21）
#   ② G: 盘在位（环境与数据在 G:）
#   ③ runs/train/r1_yolo26n_visdrone_640__s44/weights/best.pt 存在

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PY     = 'G:\conda-envs\yolo26train\python.exe'
$Run    = 'runs\train\r1_yolo26n_visdrone_640__s44'
$Ckpt   = Join-Path $Run 'weights\best.pt'
$Log    = 'logs\s44_scan.log'
$Budgets = @(10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000)

Write-Host '=== s44 判据扫描（两条头，11 档预算，chunk=150）==='
Write-Host "  项目根    = $Root"
Write-Host "  解释器    = $PY"
Write-Host "  权重      = $Ckpt"
Write-Host "  日志      = $Log"
Write-Host ''

# ── 前置检查 ────────────────────────────────────────────────────────────
Write-Host '[1/4] 前置检查'

# ★★★ P68 的核心 bug：变量名**不能叫 `$py`** —— PowerShell 变量名**不区分大小写**，
#     `$py` 与 `$PY`（L27 的解释器路径）是**同一个变量**。
#     这里把 `$PY` 覆盖成一个 Process 对象，于是 L88 的 `& $PY @scanArgs`
#     变成"拿 Process 对象当命令" → 
#     "The expression after '&' in a pipeline element produced an object that was not valid"。
#     改名 `$pyProcs`。
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pyProcs) {
    $n = ($pyProcs | Measure-Object).Count
    Write-Host "  ⚠️ 检测到 $n 个 python 进程（可能有训练在跑）。"
    Write-Host "     P21 的教训：并发会让 OpenCV 原生崩溃并杀死训练。"
    Write-Host "     若确认训练已结束，请手动结束这些进程后重跑本脚本。"
    exit 2
}
Write-Host '  ✅ 无 python 进程'

if (-not (Test-Path 'G:\conda-envs\yolo26train\python.exe')) {
    Write-Host '  ❌ 找不到解释器 —— G: 盘可能未挂载（外接盘，历史掉过两次）。'
    exit 3
}
Write-Host '  ✅ G: 盘在位'

if (-not (Test-Path $Ckpt)) {
    Write-Host "  ❌ 找不到权重 $Ckpt —— s44 还没产出 best.pt。"
    Write-Host '     用 python scripts\check_progress.py --run r1_yolo26n_visdrone_640__s44 看进度。'
    exit 4
}
$sz = [math]::Round((Get-Item $Ckpt).Length / 1MB, 1)
Write-Host "  ✅ 权重存在（$sz MB）"

# ── 逐头扫描 ────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[2/4] 开始扫描（每条头约 3–5 分钟，共 11 块 × 2 头）'
$all = @()
foreach ($head in @('o2m', 'o2o')) {
    Write-Host ''
    Write-Host "  ── head = $head ──"
    #  ★★ P68：变量名**不能叫 `$args`** —— 它是 PowerShell 的自动变量（保留变量），
    #     对它的赋值会被忽略，于是 `@args` 展开为空，命令变成 `& $PY`（无参数）
    #     并报 "The expression after '&' in a pipeline element produced an object
    #     that was not valid"。改用 `$scanArgs`。
    $scanArgs = @(
        'scripts\budget_scan_chunked.py',
        '--exp', 'r1_s44',
        '--head', $head,
        '--split', 'val',
        '--chunk', '150',
        '--weights', $Ckpt,
        '--budgets'
    ) + ($Budgets | ForEach-Object { "$_" })

    & $PY @scanArgs 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ head=$head 扫描失败（exit $LASTEXITCODE），日志见 $Log"
        exit 5
    }
    $out = "runs\budget\r1_s44_val_$head.json"
    if (Test-Path $out) {
        $kb = [math]::Round((Get-Item $out).Length / 1KB)
        Write-Host "  ✅ 产出 $out ($kb KB)"
        $all += $out
    } else {
        Write-Host "  ❌ 未产出 $out"
        exit 6
    }
}

# ── 核验 ────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[3/4] 核验产出'
& $PY -c @"
import json, sys
from pathlib import Path
ok = True
EXPECT = [10,20,30,50,70,100,150,200,300,500,1000]
for head in ('o2m','o2o'):
    p = Path(f'runs/budget/r1_s44_val_{head}.json')
    d = json.loads(p.read_text(encoding='utf-8'))
    n = d.get('n_images'); pi = d.get('per_image', []); b = d.get('budgets')
    good = (n == 548 and len(pi) == 548 and b == EXPECT)
    print(f'  {p.name}: n_images={n}  per_image={len(pi)}  budgets={\"OK\" if b==EXPECT else b}')
    if not good: ok = False
print('  ' + ('✅ 两条头均完整（548 图 / 11 档）' if ok else '❌ 有缺失，不要继续'))
sys.exit(0 if ok else 1)
"@
if ($LASTEXITCODE -ne 0) { exit 7 }

# ── 跨种子复现 ──────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[4/4] 跨种子复现检验（三个种子）'
& $PY scripts\verify_seed_replication.py --seeds r1 r1_s43 r1_s44 2>&1 | Tee-Object -FilePath $Log -Append

Write-Host ''
Write-Host '✅ 完成。下一步：'
Write-Host '   python scripts\recompute_manuscript_numbers.py      # 重算手稿数字（含三列）'
Write-Host '   python scripts\bootstrap_quintile_ci.py            # 分位 CI'
Write-Host '   python scripts\plot_figure4_quintile_cost.py --grayscale'
Write-Host '   python scripts\build_submission_pdf.py             # 重建投稿 PDF'
