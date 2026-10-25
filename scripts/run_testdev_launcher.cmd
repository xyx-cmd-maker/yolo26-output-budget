@echo off
rem ── test-dev 唯一一次评测（G-1 护栏）──────────────────────────────
rem 由 Windows 计划任务调用；放在**项目内**（此前放在 .probe 下，任务上下文访问不到）。
rem 官方纪律：test 只在定稿时跑一次，且不得用于挑 checkpoint 或调超参。
setlocal
pushd "D:\Deepseek Harness\yolo\yolo26-detection"
if errorlevel 1 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_s44_testdev.ps1"
set RC=%ERRORLEVEL%
popd
exit /b %RC%
