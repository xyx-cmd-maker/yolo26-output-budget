<#
================================================================================
 run_matrix.ps1 —— yolo26-detection 实验矩阵串行执行器
================================================================================
 作用
   读一个"实验清单"文本文件，按顺序调用唯一训练入口 src\train\run.py；
   全程 tee 到 logs\matrix_<timestamp>.log；任一条失败即停并打印错误摘要；
   全部成功后自动对 paper 档实验跑聚合与出表（src\utils\results_io.py）。

 为什么必须是 powershell 而不是 pwsh
   本机没有 pwsh，只有 Windows PowerShell 5.1。
   本文件必须以 UTF-8 **with BOM** 保存，否则 PS 5.1 按 ANSI 读取，中文注释与提示全部乱码。

 调用方式（在项目根或任意目录都可以，脚本自己会切到项目根）
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_matrix.ps1 -Manifest scripts\matrix\e02_e04.txt
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_matrix.ps1 -Manifest ... -DryRun

 退出码
   0 = 全部成功（且聚合出表全部成功）
   1 = 某个实验失败，或聚合失败
   2 = 前置检查未通过（解释器/F 盘/GPU/磁盘/项目文件/清单）

 实验清单格式（UTF-8 文本，一行一条命令）
   * 空行与以 # 开头的行被忽略。
   * 每行 = 传给 `python -m src.train.run` 的参数，写法与命令行完全一致，例如：
       --expcfg src/configs/experiments/e02_yolo26n_baseline.yaml
       --expcfg src/configs/experiments/e04_yolo11_parity.yaml
       --exp e03_scale_sweep --data visdrone --model yolo26s.pt --tier paper --epochs 100 --imgsz 640 --seeds 0 1 2
   * 含空格的参数用双引号包起来（例如 --model "some dir/x.pt"）。
   * 清单里的路径请写成**相对项目根**的相对路径：项目根含空格（D:\Deepseek Harness\...），
     相对路径可以完全避开 PowerShell 的引号与转义问题。
   * 只有 tier 解析为 paper 的实验会在结束时被聚合（非 paper 档按 AGENTS.md §2 不得进对外文本，
     所以脚本会显式跳过并说明）。tier 的来源与 run.py 一致：--expcfg 里的 `tier:` 覆盖 --tier，
     两者都没有时取 run.py 的默认值 smoke —— 也就是说，**批量正式跑请统一用 --expcfg**。
   * 脚本会按 run.py 的同一套优先级（expcfg 覆盖命令行）反推每个实验的
     exp_id / data / tier，用来决定聚合命令的参数；推导逻辑见 Get-ExpMeta。

 本脚本不做的事
   * 不替你改 tier、不替你加 --overwrite、不替你挑种子。它只负责"按清单顺序跑、记账、停止"。
   * 不在失败后继续跑下一条（除非显式 -AggregateOnFailure，那只影响失败后的聚合行为）。
================================================================================
#>

[CmdletBinding()]
param(
    # 实验清单文件（必填）。相对路径按项目根解析。
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Manifest,

    # 项目根。默认取本脚本上一级目录（本脚本放在 <项目根>\scripts\ 下）。
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    # 解释器绝对路径（AGENTS.md §1 实测事实：conda 环境在 F: 移动硬盘上）。
    [string]$Python = 'G:\conda-envs\yolo26train\python.exe',

    # 项目盘与 F: 各自要求的最小剩余空间（GB）。
    [int]$MinFreeGB = 20,

    # 失败时在日志里回放最后多少行输出。
    [int]$TailLines = 25,

    # 只做演练：给每条命令追加 --dry-run（run.py 会打印执行计划后返回 0，不训练）。
    [switch]$DryRun,

    # 默认失败即停且不聚合；加这个开关表示"失败也把已成功的实验聚合掉"。
    [switch]$AggregateOnFailure
)

# 不要设成 'Stop'：原生命令（python）的 stderr 经 2>&1 会变成 ErrorRecord，
# 在 Stop 下会直接终止脚本，从而掩盖真正的 $LASTEXITCODE。
$ErrorActionPreference = 'Continue'

# 让 python 的 stdout/stderr 实时写进日志（否则训练结束才一次性刷出），并避免控制台代码页乱码。
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

# ------------------------------------------------------------------ 日志工具

$script:LogPath = $null

function Write-Raw {
    # 原样写一行（用于转发 python 的输出），不额外加时间戳。
    param([string]$Text)
    Add-Content -LiteralPath $script:LogPath -Value $Text -Encoding UTF8
    Write-Host $Text
}

function Write-Log {
    # 脚本自己的提示行，带时间戳。
    param([string]$Message, [string]$Color = 'Gray')
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message
    Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
    Write-Host $line -ForegroundColor $Color
}

function ConvertTo-ArgArray {
    # 把清单里的一行拆成参数数组；支持用双引号包住含空格的参数。
    param([string]$Line)
    $out = @()
    foreach ($m in [regex]::Matches($Line, '"([^"]*)"|(\S+)')) {
        if ($m.Groups[1].Success) { $out += $m.Groups[1].Value } else { $out += $m.Groups[2].Value }
    }
    return , $out
}

function Get-IsRooted {
    # 替代 [System.IO.Path]::IsPathRooted，避免依赖 .NET 静态调用。
    param([string]$Path)
    return ($Path -match '^([A-Za-z]:|\\\\|/)')
}

function Resolve-AgainstRoot {
    param([string]$Path)
    if (Get-IsRooted -Path $Path) { return $Path }
    return (Join-Path $ProjectRoot $Path)
}

function Get-YamlTopLevel {
    # 极简 YAML 顶层键读取（不引入模块依赖，PS 5.1 没有内置 YAML）。
    # 只认**行首无空白**的 key: value；缩进键属于 train_args/val_args，跳过。
    param([string]$Path)
    $map = @{}
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $map }
    foreach ($raw in (Get-Content -LiteralPath $Path -Encoding UTF8)) {
        if ($raw -match '^\s*#' -or $raw.Trim().Length -eq 0) { continue }
        if ($raw -match '^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$') {
            $v = ($Matches[2] -split '\s+#')[0].Trim()      # 去掉行尾注释
            $v = $v.Trim('"').Trim("'")
            $map[$Matches[1]] = $v
        }
    }
    return $map
}

function Get-DatasetName {
    # 复刻 run.py 落盘时的 dataset 字段：Path(data).stem
    param([string]$Data)
    $name = $Data -replace '.*[\\/]', ''
    $name = $name -replace '\.ya?ml$', ''
    return $name
}

function Get-ExpMeta {
    # 按 run.py 的同一优先级（expcfg 覆盖命令行）反推 exp_id / data / tier。
    # 这两个值直接决定聚合命令 `python -m src.utils.results_io <exp_id> <dataset>`。
    param([string[]]$Argv)
    $meta = @{ exp_id = ''; data = 'coco8'; tier = 'smoke'; expcfg = '' }
    for ($i = 0; $i -lt $Argv.Count; $i++) {
        switch ($Argv[$i]) {
            '--exp' { if ($i + 1 -lt $Argv.Count) { $meta.exp_id = $Argv[$i + 1] } }
            '--data' { if ($i + 1 -lt $Argv.Count) { $meta.data = $Argv[$i + 1] } }
            '--tier' { if ($i + 1 -lt $Argv.Count) { $meta.tier = $Argv[$i + 1] } }
            '--expcfg' { if ($i + 1 -lt $Argv.Count) { $meta.expcfg = $Argv[$i + 1] } }
        }
    }
    if ($meta.expcfg) {
        $p = Resolve-AgainstRoot -Path $meta.expcfg
        $y = Get-YamlTopLevel -Path $p
        if ($y.ContainsKey('exp_id')) { $meta.exp_id = $y['exp_id'] }
        if ($y.ContainsKey('data')) { $meta.data = $y['data'] }
        if ($y.ContainsKey('tier')) { $meta.tier = $y['tier'] }
    }
    $meta['dataset'] = Get-DatasetName -Data $meta.data
    return $meta
}

# ------------------------------------------------------------------ 前置检查

function Test-Preflight {
    # 返回问题描述数组；空数组 = 全部通过。
    param([string]$ManifestAbs)

    $fail = @()

    # 1) 解释器（F: 是移动硬盘，拔盘后这个路径直接消失）
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        $fail += "解释器不存在：$Python（F: 移动硬盘是否插好？conda 环境在 G:\conda-envs\yolo26train）"
    }

    # 2) F: 盘在位
    if (-not (Test-Path -LiteralPath 'F:\')) {
        $fail += 'F: 盘不可访问。环境装在 F: 上，拔盘即环境失效（AGENTS.md §1.1 铁律 3）'
    }

    # 3) GPU 可见（跑一次 python 探针；同时验证 torch 能 import）
    if ($fail.Count -eq 0) {
        $probe = & $Python -c 'import torch;print("GPU_OK" if torch.cuda.is_available() else "GPU_FAIL")' 2>&1
        $probeText = ($probe | Out-String)
        if ($LASTEXITCODE -ne 0) {
            $fail += "GPU 探针执行失败（exit=$LASTEXITCODE）：$($probeText.Trim())"
        }
        elseif ($probeText -notmatch 'GPU_OK') {
            $fail += "torch 报告 GPU 不可见。检查：驱动、torch 是否为 +cu130 版本（sm_120 要求 cu128 及以上）"
        }
    }

    # 4) 磁盘空间：项目盘（data\ 与 runs\ 都在这）与 F:（环境）
    $projDrive = ''
    if ($ProjectRoot -match '^([A-Za-z]):') { $projDrive = $Matches[1] }
    foreach ($d in @($projDrive, 'F') | Where-Object { $_ -ne '' } | Sort-Object -Unique) {
        $drv = Get-PSDrive -Name $d -ErrorAction SilentlyContinue
        if ($null -eq $drv) {
            $fail += "取不到盘 $d`: 的信息"
        }
        elseif ($drv.Free -lt ($MinFreeGB * 1GB)) {
            $fail += ("盘 {0}: 剩余 {1:N1} GB < 要求的 {2} GB" -f $d, ($drv.Free / 1GB), $MinFreeGB)
        }
    }

    # 5) 项目关键文件
    foreach ($rel in @('src\train\run.py', 'src\utils\results_io.py', 'src\utils\repro.py', 'src\configs\datasets')) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $rel))) {
            $fail += "缺少项目文件/目录：$rel"
        }
    }

    # 6) 实验清单
    if (-not (Test-Path -LiteralPath $ManifestAbs -PathType Leaf)) {
        $fail += "实验清单不存在：$ManifestAbs"
    }

    return , $fail
}

# ================================================================== 主流程

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Host "项目根不存在：$ProjectRoot" -ForegroundColor Red
    exit 2
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

# 日志先落地，后面所有输出都 tee 进去
$logDir = Join-Path $ProjectRoot 'logs'
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$script:LogPath = Join-Path $logDir ("matrix_{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType File -Force -Path $script:LogPath | Out-Null

$ManifestAbs = Resolve-AgainstRoot -Path $Manifest

Push-Location $ProjectRoot
try {
    Write-Log '=================================================================='
    Write-Log ' yolo26-detection 实验矩阵'
    Write-Log '=================================================================='
    Write-Log ("项目根    ：{0}" -f $ProjectRoot)
    Write-Log ("解释器    ：{0}" -f $Python)
    Write-Log ("清单      ：{0}" -f $ManifestAbs)
    Write-Log ("日志      ：{0}" -f $script:LogPath)
    Write-Log ("DryRun    ：{0}" -f [bool]$DryRun)
    Write-Log ("起始时间  ：{0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))

    # ---------------- 前置检查 ----------------
    Write-Log '---- 前置检查 ----'
    $problems = Test-Preflight -ManifestAbs $ManifestAbs
    if ($problems.Count -gt 0) {
        foreach ($p in $problems) { Write-Log ("❌ " + $p) 'Red' }
        Write-Log '前置检查未通过，未开始任何训练。' 'Red'
        exit 2
    }
    Write-Log '✅ 解释器 / F 盘 / GPU / 磁盘空间 / 项目文件 / 清单 全部通过' 'Green'

    # ---------------- 读清单 ----------------
    $items = @()
    $lineNo = 0
    foreach ($raw in (Get-Content -LiteralPath $ManifestAbs -Encoding UTF8)) {
        $lineNo++
        $line = $raw.Trim()
        if ($line.Length -eq 0) { continue }
        if ($line.StartsWith('#')) { continue }
        $argv = @(ConvertTo-ArgArray -Line $line)
        if ($argv.Count -eq 0) { continue }
        $items += [pscustomobject]@{ LineNo = $lineNo; Args = $argv; Text = $line }
    }
    if ($items.Count -eq 0) {
        Write-Log "清单里没有有效命令（空行与 # 注释行不计）：$ManifestAbs" 'Red'
        exit 2
    }
    Write-Log ("清单解析完成：{0} 条实验" -f $items.Count)

    # ---------------- 逐个跑 ----------------
    $matrixStart = Get-Date
    $failed = $false
    $failedText = ''
    $records = @()
    $idx = 0

    foreach ($it in $items) {
        $idx++
        $argv = @($it.Args)
        if ($DryRun) { $argv += '--dry-run' }

        # 反推 exp_id / dataset / tier，供结束时决定聚合参数；缺 exp_id 不在这里报错，
        # 交给 run.py 自己报（它会 ap.error 退出），这里只在聚合阶段跳过。
        $meta = Get-ExpMeta -Argv $argv

        Write-Log '=================================================================='
        Write-Log ("[{0}/{1}] 清单第 {2} 行：{3}" -f $idx, $items.Count, $it.LineNo, $it.Text) 'Cyan'
        Write-Log ("         exp_id={0}  dataset={1}  tier={2}" -f $meta.exp_id, $meta.dataset, $meta.tier) 'Cyan'

        $t0 = Get-Date
        $tail = @()   # 只保留最后 $TailLines 行，失败时回放

        # 只读转发子进程输出：这里的 ForEach-Object 不影响 $LASTEXITCODE 的传递。
        & $Python -m src.train.run @argv 2>&1 | ForEach-Object {
            $s = [string]$_
            Write-Raw $s
            $tail += $s
            if ($tail.Count -gt $TailLines) {
                $tail = @($tail[($tail.Count - $TailLines)..($tail.Count - 1)])
            }
        }
        $code = $LASTEXITCODE
        $minutes = ((Get-Date) - $t0).TotalMinutes

        $records += [pscustomobject]@{
            Line    = $it.LineNo
            exp_id  = $meta.exp_id
            dataset = $meta.dataset
            tier    = $meta.tier
            exit    = $code
            minutes = $minutes
        }

        if ($code -ne 0) {
            Write-Log ("❌ 第 {0} 条失败：exit={1}，耗时 {2:N1} 分钟" -f $it.LineNo, $code, $minutes) 'Red'
            Write-Log ("   命令：python -m src.train.run {0}" -f ($argv -join ' ')) 'Red'
            Write-Log ("---- 最后 {0} 行输出（完整日志：{1}）----" -f $TailLines, $script:LogPath) 'Red'
            foreach ($l in $tail) { Write-Log ('    ' + $l) 'Red' }
            Write-Log '---- 常见原因与处置（详见 docs\experiment-playbook.md §6）----' 'Red'
            Write-Log '  1) FileExistsError "... 已存在"      → 该 (exp_id,dataset,tier,seed) 已跑过；换 seed 或显式 --overwrite（等于丢一个种子），重跑后必须重跑聚合'
            Write-Log '  2) TierViolation "含非 paper 档"       → 同一 (exp_id,dataset) 下混了 dev 与 paper；给 dev 换一个 exp_id'
            Write-Log '  3) blocked_reasons "种子数 N < 3"       → 种子不足，不出表；按 playbook §2 E5 补跑缺失的种子'
            Write-Log '  4) ValueError "未知档位 ..."           → expcfg 的 tier 拼写/大小写错误（注意这一轮训练已经白跑）'
            Write-Log '  5) TypeError "multiple values for ..." → train_args/val_args 里写了 run.py 的保留键'
            Write-Log '  6) CUDA out of memory                  → 调小 expcfg 的 batch/imgsz；8 GB 单卡只跑一个训练'
            Write-Log '  7) FileNotFoundError "训练未产出 best.pt"→ 训练进程中断；看 runs\train\<exp_id>__s<seed>\'
            Write-Log '  8) FileNotFoundError "images not found"  → 数据集 yaml 的 path 解析问题（playbook §7.1）；先加 --dry-run 看 data 解析结果'
            $failed = $true
            $failedText = ("清单第 {0} 行：{1}" -f $it.LineNo, $it.Text)
            break
        }

        Write-Log ("✅ 第 {0} 条完成：exit=0，耗时 {1:N1} 分钟" -f $it.LineNo, $minutes) 'Green'
    }

    # ---------------- 汇总 ----------------
    $totalMinutes = ((Get-Date) - $matrixStart).TotalMinutes
    Write-Log '=================================================================='
    Write-Log ' 执行汇总'
    Write-Log '=================================================================='
    foreach ($r in $records) {
        Write-Log ("  第 {0,3} 行  exit={1,3}  {2,7:N1} 分钟  tier={3,-5}  {4} / {5}" -f $r.Line, $r.exit, $r.minutes, $r.tier, $r.exp_id, $r.dataset)
    }
    Write-Log ("总耗时：{0:N1} 分钟" -f $totalMinutes)

    if ($failed) {
        Write-Log ("❌ 失败即停：{0}" -f $failedText) 'Red'
        if (-not $AggregateOnFailure) {
            Write-Log '按默认策略不聚合（失败即停）。要聚合已成功的实验，加 -AggregateOnFailure 重跑，或手动执行：' 'Red'
            Write-Log '  python -m src.utils.results_io <exp_id> <dataset>' 'Red'
            Write-Log ("日志：{0}" -f $script:LogPath) 'Red'
            exit 1
        }
        Write-Log '（-AggregateOnFailure 已指定：继续聚合已成功的实验）' 'Yellow'
    }

    # ---------------- 聚合与出表 ----------------
    if ($DryRun) {
        Write-Log '-DryRun：只演练，没有任何记录落盘，跳过聚合与出表。' 'Yellow'
        Write-Log ("日志：{0}" -f $script:LogPath)
        exit 0
    }
    Write-Log '---- 聚合与出表（python -m src.utils.results_io <exp_id> <dataset>）----'
    $aggFail = 0
    $aggSkip = 0
    $seen = @()
    foreach ($r in $records) {
        if ($r.exit -ne 0) { continue }
        $key = "{0}|{1}" -f $r.exp_id, $r.dataset
        if ($seen -contains $key) { continue }        # 同一实验的多行只聚合一次
        $seen += $key

        if (-not $r.exp_id) {
            Write-Log ("跳过：清单第 {0} 行没有可解析的 exp_id（既没有 --exp，expcfg 里也没有 exp_id）" -f $r.Line) 'Yellow'
            $aggSkip++
            continue
        }
        if ($r.tier -ne 'paper') {
            Write-Log ("跳过 {0}/{1}：tier={2} 非 paper。按 AGENTS.md §2，非 paper 档不得聚合进对外文本（要强制需人工加 --force，并会留痕）" -f $r.exp_id, $r.dataset, $r.tier) 'Yellow'
            $aggSkip++
            continue
        }

        Write-Log ("聚合 {0} / {1}" -f $r.exp_id, $r.dataset) 'Cyan'
        & $Python -m src.utils.results_io $r.exp_id $r.dataset 2>&1 | ForEach-Object { Write-Raw ([string]$_) }
        if ($LASTEXITCODE -ne 0) {
            Write-Log ("❌ 聚合失败：{0}/{1}（exit={2}）" -f $r.exp_id, $r.dataset, $LASTEXITCODE) 'Red'
            $aggFail++
        }
        else {
            Write-Log ("✅ 聚合与出表完成：{0}/{1}" -f $r.exp_id, $r.dataset) 'Green'
        }
    }
    Write-Log ("聚合：成功 {0} 个，跳过 {1} 个，失败 {2} 个" -f ($seen.Count - $aggSkip - $aggFail), $aggSkip, $aggFail)
    Write-Log ("日志：{0}" -f $script:LogPath)

    if ($failed -or $aggFail -gt 0) { exit 1 }
    exit 0
}
finally {
    Pop-Location
}
