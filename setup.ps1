# Smart Glasses - one-time setup for Windows (run via setup.bat)
# Creates a .venv inside the project and installs everything the app needs.
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "setup_log.txt"
"Setup started $(Get-Date)" | Out-File $log -Encoding utf8

function Say($text, $color = "Cyan") {
    Write-Host ""
    Write-Host "==== $text" -ForegroundColor $color
    "==== $text" | Out-File $log -Append -Encoding utf8
}
function Run($exe, [string[]]$arguments) {
    & $exe @arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    return $LASTEXITCODE
}

# ---------- 1. Find Python ----------
Say "1/5  Looking for Python"
$py = $null
foreach ($c in @("C:\Python314\python.exe",
                 "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
                 "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
                 "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) { $cmd = Get-Command python -ErrorAction SilentlyContinue; if ($cmd) { $py = $cmd.Source } }
if (-not $py) {
    Say "Python was not found. Install it from python.org and run setup.bat again." "Red"
    Read-Host "Press Enter to close"; exit 1
}
Say "Using $py" "Green"

# ---------- 2. Virtual environment ----------
Say "2/5  Creating the project environment (.venv)"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { $null = Run $py @("-m", "venv", ".venv") }
if (-not (Test-Path $venvPy)) { Say "Could not create .venv" "Red"; Read-Host "Press Enter to close"; exit 1 }
$null = Run $venvPy @("-m", "pip", "install", "--upgrade", "pip")

# ---------- 3. Core (face recognition, detection, currency, finder) ----------
Say "3/5  Installing the main libraries (about 1 GB, takes a while)"
$code = Run $venvPy @("-m", "pip", "install", "flask", "waitress", "ultralytics",
                      "opencv-python", "num2words", "pillow", "numpy")
if ($code -ne 0) {
    Say "Main libraries FAILED to install. Send setup_log.txt to Claude." "Red"
    Read-Host "Press Enter to close"; exit 1
}
Say "Main libraries installed" "Green"

# ---------- 4. Optional modes: describe, text (OCR), summarize ----------
Say "4/5  Installing description / text-reading / summary libraries"
$descOk = (Run $venvPy @("-m", "pip", "install", "transformers", "accelerate")) -eq 0
$ocrOk  = (Run $venvPy @("-m", "pip", "install", "pytesseract", "pyttsx3")) -eq 0
$sumOk  = (Run $venvPy @("-m", "pip", "install", "txtai")) -eq 0

$tesseract = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (-not (Test-Path $tesseract)) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Say "Installing Tesseract OCR (Windows may ask for permission - click Yes)"
        $null = Run "winget" @("install", "--id", "UB-Mannheim.TesseractOCR", "-e", "--silent",
                       "--accept-package-agreements", "--accept-source-agreements")
    }
}

# ---------- 5. Download AI models now (so tomorrow needs no big download) ----------
Say "5/5  Downloading AI models (description ~1.8 GB, summary ~1.2 GB)"
$null = Run $venvPy @("-c", "from ultralytics import YOLO; YOLO('yolov8m.pt'); print('YOLO ok')")
if ($descOk) {
    $null = Run $venvPy @("-c", "from transformers import BlipProcessor, BlipForConditionalGeneration as B; n='Salesforce/blip-image-captioning-large'; BlipProcessor.from_pretrained(n); B.from_pretrained(n); print('BLIP ok')")
}
if ($sumOk) {
    $null = Run $venvPy @("-c", "from txtai.pipeline import Summary; Summary(); print('Summary ok')")
}

# ---------- Summary ----------
Say "SETUP FINISHED" "Green"
Write-Host ("  Face recognition / detection / currency : OK") -ForegroundColor Green
Write-Host ("  Description mode : " + $(if ($descOk) {"OK"} else {"not installed"}))
Write-Host ("  Text (OCR) mode  : " + $(if ($ocrOk -and (Test-Path $tesseract)) {"OK"} else {"needs Tesseract"}))
Write-Host ("  Summarize mode   : " + $(if ($sumOk) {"OK"} else {"not installed"}))
Write-Host ""
Write-Host "Now double-click run.bat" -ForegroundColor Yellow
"Setup finished $(Get-Date) desc=$descOk ocr=$ocrOk sum=$sumOk tesseract=$(Test-Path $tesseract)" | Out-File $log -Append -Encoding utf8
Read-Host "Press Enter to close"
