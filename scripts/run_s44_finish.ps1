# s44 收尾链：等训练进程退出 → 扫描 → 重算 → 自助法 → 对账 → 重建 PDF/LaTeX
# 用法: powershell -File run_s44_finish.ps1
$ErrorActionPreference = 'Continue'
Set-Location 'D:\Deepseek Harness\yolo\yolo26-detection'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$PY = 'G:\conda-envs\yolo26train\python.exe'
$log = 'logs\s44_finish.log'
New-Item -ItemType Directory -Force -Path 'logs' | Out-Null
function Say($m) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $m"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Say '=== 等待 s44 训练进程退出（每 60 秒检查）==='
$waited = 0
while ($true) {
    $p = Get-Process python -ErrorAction SilentlyContinue
    $train = $false
    foreach ($proc in $p) {
        $cl = (Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" -EA SilentlyContinue).CommandLine
        if ($cl -and $cl -match 'src\\train\\run\.py') { $train = $true; break }
    }
    if (-not $train) { Say '训练进程已退出 ✅'; break }
    if ($waited -ge 5400) { Say '等待超时（90 分钟）——中止'; exit 1 }
    Start-Sleep -Seconds 60
    $waited += 60
    if ($waited % 300 -eq 0) { Say "  已等待 $([int]($waited/60)) 分钟…" }
}
Start-Sleep -Seconds 20   # 让 DDP worker 完全释放

Say '=== ① 运行 s44 预算扫描（11 档）==='
& powershell -NoProfile -ExecutionPolicy Bypass -File 'scripts\run_s44_scan.ps1' 2>&1 |
    Tee-Object -FilePath 'logs\s44_scan_out.log' -Append
Say "扫描退出码 = $LASTEXITCODE"

Say '=== ② 重算手稿数字（三种子）==='
& $PY 'scripts\recompute_manuscript_numbers.py' 2>&1 | Add-Content -Path $log -Encoding UTF8
Say "重算退出码 = $LASTEXITCODE"

Say '=== ③ 自助法置信区间（三种子）==='
& $PY 'scripts\bootstrap_quintile_ci.py' --seeds r1 r1_s43 r1_s44 2>&1 | Add-Content -Path $log -Encoding UTF8
Say "自助法退出码 = $LASTEXITCODE"

Say '=== ④ 主张对账 ==='
& $PY 'scripts\audit_claims_vs_data.py' 2>&1 | Add-Content -Path $log -Encoding UTF8
Say "对账退出码 = $LASTEXITCODE"

Say '=== ⑤ 数字溯源 ==='
& $PY 'scripts\verify_manuscript_numbers_against_source.py' 2>&1 | Add-Content -Path $log -Encoding UTF8
Say "溯源退出码 = $LASTEXITCODE"

Say '=== ⑥ 参考文献重排编号 ==='
& $PY 'scripts\number_references.py' 2>&1 | Add-Content -Path $log -Encoding UTF8

Say '=== ⑦ 重建投稿 PDF ==='
& $PY 'scripts\build_submission_pdf.py' 2>&1 | Add-Content -Path $log -Encoding UTF8

Say '=== ⑧ 重建 LaTeX 并编译（加列会改表宽）==='
& $PY 'scripts\build_latex_submission.py' 2>&1 | Add-Content -Path $log -Encoding UTF8
Push-Location 'latex'
Remove-Item 'manuscript.aux','manuscript.log','manuscript.out','manuscript.bbl','manuscript.blg' -Force -EA SilentlyContinue
& pdflatex -interaction=nonstopmode -file-line-error manuscript.tex 2>&1 | Out-Null
& bibtex manuscript 2>&1 | Out-Null
& pdflatex -interaction=nonstopmode -file-line-error manuscript.tex 2>&1 | Out-Null
& pdflatex -interaction=nonstopmode -file-line-error manuscript.tex 2>&1 | Out-Null
Pop-Location
& $PY 'D:\Deepseek Harness\yolo\.probe\check_latex_log.py' 2>&1 | Add-Content -Path $log -Encoding UTF8

Say '=== 完成：请查看 logs\s44_finish.log ==='
