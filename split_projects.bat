@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "R=D:\Projects\GitHub\Raqib"
echo =====================================================================
echo  Split into two projects:  Ru'ya (glasses, public)  +  Raqib (CCTV, private)
echo =====================================================================
if not exist "%R%\cctv.py" (
    echo STOP: Raqib's code is not in %R% yet. Nothing was changed.
    pause & exit /b 1
)

echo.
echo 0/4  Restoring the complete new version (the first upload script had switched this folder back to May 2025)...
set "HERE=%CD:\=/%"
git config --global --get-all safe.directory | findstr /x /c:"%HERE%" >nul || git config --global --add safe.directory "%HERE%"
set "BK=D:\Projects\GitHub\SimulatedSmartGlasses\_backup_2025_frames"
if exist uploads robocopy uploads "%BK%\uploads" /E /NJH /NJS /NFL /NDL /NP >nul
if exist runs robocopy runs "%BK%\runs" /E /NJH /NJS /NFL /NDL /NP >nul
git checkout -f v2-cctv-maat || ( echo STOP: could not restore the new version. Nothing else was changed. & pause & exit /b 1 )
if not exist face_function.py ( echo STOP: restore incomplete. & pause & exit /b 1 )
if not exist models\face_recognition_sface_2021dec.onnx ( echo STOP: models missing after restore. & pause & exit /b 1 )
echo      Applying the Ru'ya changes (name, distance in steps)...
xcopy /E /Y /Q _ruya_update\* . >nul
rmdir /s /q _ruya_update
echo      Done. Old 2025 frames are backed up in %BK%

echo.
echo 1/4  Copying models, samples, saved people and the Python environment to Raqib...
robocopy . "%R%" yolov8m.pt currency.pt /NJH /NJS /NFL /NDL /NP >nul
robocopy models "%R%\models" face_detection_yunet_2023mar.onnx face_recognition_sface_2021dec.onnx facial_expression_recognition_mobilefacenet_2022july.onnx age_googlenet.onnx plate_chars.pt plate_detector.pt /NJH /NJS /NFL /NDL /NP >nul
if exist samples robocopy samples "%R%\samples" /E /NJH /NJS /NFL /NDL /NP >nul
if exist faces robocopy faces "%R%\faces" /E /NJH /NJS /NFL /NDL /NP >nul
if exist training\plates_new robocopy training\plates_new "%R%\training\plates_new" /E /NJH /NJS /NFL /NDL /NP >nul
echo      Python environment (a few GB, about a minute)...
if not exist "%R%\.venv\Scripts\python.exe" robocopy .venv "%R%\.venv" /E /NJH /NJS /NFL /NDL /NP /MT:16 >nul
for %%F in (yolov8m.pt currency.pt models\face_recognition_sface_2021dec.onnx models\plate_chars.pt models\plate_detector.pt .venv\Scripts\python.exe) do (
    if not exist "%R%\%%F" ( echo STOP: %%F did not copy to Raqib. Nothing was removed. & pause & exit /b 1 )
)
echo      Done.

echo.
echo 2/4  Removing the CCTV parts from the glasses folder (they are now in Raqib)...
for %%F in (cctv.py activity.py lpr.py templates\cctv.html tests\test_cctv.py tests\test_activity.py tests\test_lpr.py training\train_plate_chars.py training\train_plate_chars.bat models\plate_chars.pt models\plate_detector.pt faces\plates.json faces\lists.json) do if exist "%%F" del /q "%%F"
for %%D in (samples tests\data uploads\cctv training\plates_new) do if exist "%%D" rmdir /s /q "%%D"
echo      Done.

echo.
echo 3/4  Uploading Ru'ya (public)...
set "HERE=%CD:\=/%"
git config --global --get-all safe.directory | findstr /x /c:"%HERE%" >nul || git config --global --add safe.directory "%HERE%"
git remote set-url origin https://github.com/Ahmed-Yasser-V0/Ruya.git
git checkout -B ruya-v2
git reset --soft origin/main
git rm -r -q --cached __pycache__ >nul 2>nul
git rm -r -q --cached uploads >nul 2>nul
git rm -r -q --cached runs >nul 2>nul
git add -A
git diff --cached --name-only --diff-filter=AM | findstr /c:"faces/" /c:"uploads/" /c:"plates_new" /c:".venv/" /c:"cctv.py" /c:"lpr.py" >nul && (
    echo STOP: a private or Raqib file was about to be uploaded to the public repo. Nothing was sent.
    pause & exit /b 1
)
git commit -q -m "Ru'ya v2: audit, Guide mode, Arabic, find anything, mood and age, distance in steps" -m "Renamed from SimulatdSmartGlasses. CCTV and security features moved to a separate private project." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GHKTAyrYAoNtVNsK7RwJy3"
git push -u origin ruya-v2 || ( echo Upload of Ru'ya failed. & pause & exit /b 1 )
echo      Removing the old public branch that contained the CCTV code...
git push origin --delete v2-cctv-maat
git branch -D v2-cctv-maat >nul 2>nul
rem the local "main" was moved onto the old CCTV commit by the first upload script: put it back
git branch -f main origin/main >nul 2>nul

echo.
echo 4/4  Uploading Raqib (private)...
cd /d "%R%"
set "RQ=%CD:\=/%"
git config --global --get-all safe.directory | findstr /x /c:"%RQ%" >nul || git config --global --add safe.directory "%RQ%"
if not exist .git git init -q -b main
git config user.name >nul 2>nul || git config user.name "Ahmed Yasser"
git config user.email >nul 2>nul || git config user.email "Ahmed-Yasser-V0@users.noreply.github.com"
git add -A
git diff --cached --name-only --diff-filter=AM | findstr /c:"faces/" /c:"uploads/" /c:"plates_new" /c:".venv/" >nul && (
    echo STOP: a private file was about to be uploaded. Nothing was sent.
    pause & exit /b 1
)
git commit -q -m "Raqib: CCTV analytics - face lists and alarms, Egyptian plate reading, working time, money, vehicles" -m "Split from the Ru'ya smart-glasses project." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GHKTAyrYAoNtVNsK7RwJy3"
git remote get-url origin >nul 2>nul || git remote add origin https://github.com/Ahmed-Yasser-V0/Raqib.git
git push -u origin main || ( echo Upload of Raqib failed. & pause & exit /b 1 )

echo.
echo =====================================================================
echo  DONE. Ru'ya: press "Create pull request" then "Merge" in the page that opens.
echo        Raqib: https://github.com/Ahmed-Yasser-V0/Raqib  (private)
echo =====================================================================
start "" "https://github.com/Ahmed-Yasser-V0/Ruya/compare/main...ruya-v2?expand=1"
pause
