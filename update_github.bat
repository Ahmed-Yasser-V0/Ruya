@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================================
echo  Upload Smart Glasses v2 (CCTV, Ma'at, plates) to GitHub
echo ==========================================================
where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed. Installing Git for Windows...
    winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
    set "PATH=%PATH%;C:\Program Files\Git\cmd"
)
where git >nul 2>nul
if errorlevel 1 (
    echo Git could not be installed. Install it from https://git-scm.com and run this again.
    pause
    exit /b 1
)
rem The folder was created by the Windows account from before the reinstall: tell Git it is trusted
set "HERE=%CD:\=/%"
git config --global --get-all safe.directory | findstr /x /c:"%HERE%" >nul || git config --global --add safe.directory "%HERE%"
git config user.name >nul 2>nul || git config user.name "Ahmed Yasser"
git config user.email >nul 2>nul || git config user.email "Ahmed-Yasser-V0@users.noreply.github.com"
git remote set-url origin https://github.com/Ahmed-Yasser-V0/SimulatdSmartGlasses.git
echo.
echo Safety check - these private folders are NOT uploaded:
echo   faces\  uploads\  training\plates_new\  .venv\  logs
echo.
rem old cache files, camera frames and test outputs uploaded in 2025 are removed from GitHub (they stay on this computer)
git rm -r -q --cached __pycache__ >nul 2>nul
git rm -r -q --cached uploads >nul 2>nul
git rm -r -q --cached runs >nul 2>nul
git checkout -B v2-cctv-maat
git add -A
git diff --cached --name-only --diff-filter=AM | findstr /c:"faces/" /c:"uploads/" /c:"plates_new" /c:".venv/" >nul && (
    echo STOP: a private file was about to be uploaded. Nothing was sent.
    pause
    exit /b 1
)
git commit -m "v2: audit, Guide mode, Arabic, CCTV console with face lists, working time, mood and age, money, vehicles and Egyptian plate reading (LPR)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GHKTAyrYAoNtVNsK7RwJy3"
echo.
echo Uploading... (a GitHub sign-in window may open the first time - sign in as Ahmed-Yasser-V0)
git push -u origin v2-cctv-maat
if errorlevel 1 (
    echo Upload failed. Check your internet and GitHub sign-in, then run this again.
    pause
    exit /b 1
)
echo.
echo Done. Opening GitHub: press "Create pull request", then "Merge" to put it on main.
start "" "https://github.com/Ahmed-Yasser-V0/SimulatdSmartGlasses/compare/main...v2-cctv-maat?expand=1"
git checkout main >nul 2>nul
git merge --ff-only v2-cctv-maat >nul 2>nul
pause
