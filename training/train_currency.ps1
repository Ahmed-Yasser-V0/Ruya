# Retrain the currency model on this laptop's NVIDIA GPU.
# 1) Put the Roboflow dataset zips (YOLOv8 format) in training\datasets\
# 2) Double-click training\train_currency.bat
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "training_log.txt"
"Started $(Get-Date)" | Out-File $log -Encoding utf8
function Run($exe, [string[]]$a) { & $exe @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host; return $LASTEXITCODE }
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "1/3  Installing PyTorch for the NVIDIA GPU (about 2.5 GB, one time only)" -ForegroundColor Cyan
$cuda = & $py -c "import torch;print(torch.cuda.is_available())" 2>$null
if ($cuda -ne "True") {
    $null = Run $py @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")
}
$cuda = & $py -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>$null
"GPU: $cuda" | Tee-Object -FilePath $log -Append

Write-Host "2/3  Unpacking datasets" -ForegroundColor Cyan
$ds = Join-Path $PSScriptRoot "datasets"
New-Item -ItemType Directory -Force -Path $ds | Out-Null
Get-ChildItem $ds -Filter *.zip | ForEach-Object {
    $dest = Join-Path $ds $_.BaseName
    if (-not (Test-Path $dest)) { Expand-Archive $_.FullName -DestinationPath $dest -Force }
}
$folders = Get-ChildItem $ds -Directory | Where-Object { Test-Path (Join-Path $_.FullName "data.yaml") } | ForEach-Object { $_.FullName }
if (-not $folders) { Write-Host "No datasets found in training\datasets" -ForegroundColor Red; Read-Host "Press Enter"; exit 1 }
"Datasets: $($folders -join ', ')" | Tee-Object -FilePath $log -Append

Write-Host "3/3  Training (about 30-60 minutes on the GPU). Keep the laptop plugged in." -ForegroundColor Cyan
$env:PYTHONUTF8 = "1"
$null = Run $py (@("-u", "training\train_currency.py") + $folders)
Write-Host "Finished. See training\training_log.txt" -ForegroundColor Green
Read-Host "Press Enter to close"
