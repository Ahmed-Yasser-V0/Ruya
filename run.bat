@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
rem Use the AI models already downloaded by setup.bat (works without internet)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
rem Close an older copy of this app that is still running (it keeps the camera and run_log.txt busy)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$venv = (Resolve-Path '.venv').Path; $all = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\"; $old = @($all | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venv) -and $_.CommandLine -like '*app.py*' }); $ids = @($old | ForEach-Object { $_.ProcessId }); $kids = @($all | Where-Object { $ids -contains $_.ParentProcessId }); foreach ($p in ($kids + $old)) { Write-Host ('Closing the old Ru''ya server (process ' + $p.ProcessId + ')'); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul
echo Starting Ru'ya smart glasses... (first start can take 1-2 minutes)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\.venv\Scripts\python.exe' -u app.py 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $(try { [IO.File]::Open('run_log.txt','OpenOrCreate','ReadWrite','None').Close(); 'run_log.txt' } catch { 'run_log_' + (Get-Date -Format 'HHmmss') + '.txt' })"
pause
