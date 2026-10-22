# tools/retrain.ps1 -- one-shot retraining cycle for the clubhead detector.
#
#   powershell -ExecutionPolicy Bypass -File tools\retrain.ps1 [-Dataset datasets\clubhead] [-ValRatio 0.2] [-Epochs 60]
#                                                               [-TorchPython ..\golf-autotrace\.venv\Scripts\python.exe]
#                                                               [-Split holdout|tune|all] [-SkipEval] [-Name <model name>]
#
# Steps (all outputs under build\retrain\<stamp>\ plus models\<name>.onnx):
#   1. inventory  : tools\dataset_inventory.py  (counts, sources, cameras, sizes, holdout check -> labels.json "excluded", anomalies)
#   2. train      : tools\train_clubhead.py --val-ratio (clip-level split stratified by camera) -> models\clubhead_<date>_<n>.onnx
#                   The current models\clubhead.onnx is NOT overwritten.
#   3. evaluate   : tools\eval_club.py --split <Split> --detector both --model <new>  (legacy / current / new in one table)
#   4. report     : tools\retrain_report.py -> build\retrain\<stamp>\report.md  (recommendation guide + swap procedure)
#
# Requirements: .venv (app deps), a torch-enabled venv for training (TorchPython), labels in the dataset folder.
# ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less files as ANSI). Run from cmd.exe or Git Bash.

param(
    [string]$Dataset = "datasets\clubhead",
    [double]$ValRatio = 0.2,
    [int]$Epochs = 60,
    [string]$TorchPython = "..\golf-autotrace\.venv\Scripts\python.exe",
    [string]$Split = "holdout",
    [string]$Name = "",
    [switch]$SkipEval
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found: $py" }
if (-not (Test-Path $TorchPython)) { throw "torch python not found: $TorchPython (training venv)" }
if (-not (Test-Path $Dataset)) { throw "dataset folder not found: $Dataset" }
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$work = Join-Path $root ("build\retrain\" + $stamp)
New-Item -ItemType Directory -Force $work | Out-Null
function Step($msg) { Write-Host ("`n=== " + $msg) -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. inventory
Step "1. inventory ($Dataset)"
$invTxt = Join-Path $work "inventory.txt"
$invJson = Join-Path $work "inventory.json"
& $py tools\dataset_inventory.py --dataset $Dataset --json $invJson --exclude-holdout | Tee-Object -FilePath $invTxt
if ($LASTEXITCODE -ne 0) { throw "inventory failed" }
$inv = Get-Content $invJson -Raw -Encoding UTF8 | ConvertFrom-Json
if ($inv.n_strong_usable -lt 20) { throw "not enough usable strong labels ($($inv.n_strong_usable)); collect more before retraining" }

# ---------------------------------------------------------------- 2. train
Step "2. train (val-ratio $ValRatio, epochs $Epochs)"
$trainLog = Join-Path $work "train.log"
$before = Get-ChildItem models -Filter "clubhead_*.onnx" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$targs = @("tools\train_clubhead.py", "--dataset", $Dataset, "--val-ratio", "$ValRatio", "--epochs", "$Epochs")
if ($Name) { $targs += @("--name", $Name) }
cmd.exe /c ('"' + $TorchPython + '" ' + ($targs -join ' ') + ' > "' + $trainLog + '" 2>&1')
if ($LASTEXITCODE -ne 0) { Get-Content $trainLog -Tail 20; throw "training failed (see $trainLog)" }
$after = Get-ChildItem models -Filter "clubhead_*.onnx" | Where-Object { $before -notcontains $_.FullName } | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $after) { $after = Get-ChildItem models -Filter "clubhead_*.onnx" | Sort-Object LastWriteTime | Select-Object -Last 1 }
if (-not $after) { throw "no model produced" }
$model = $after.FullName
$trainJson = [IO.Path]::ChangeExtension($model, ".train.json")
Write-Host "model: $model"
Get-Content $trainLog | Select-String -Pattern "^epoch|device=|pck" | ForEach-Object { Write-Host ("  " + $_.Line.Substring(0, [Math]::Min(160, $_.Line.Length))) }

# ---------------------------------------------------------------- 3. evaluate
$resultMd = $null
if (-not $SkipEval) {
    Step "3. evaluate on $Split : legacy / current clubhead.onnx / $($after.BaseName)"
    $evalLog = Join-Path $work "eval.log"
    $tag = "retrain_" + $stamp
    cmd.exe /c ('"' + $py + '" tools\eval_club.py --all --split ' + $Split + ' --tag ' + $tag + ' --detector both --model "' + $model + '" > "' + $evalLog + '" 2>&1')
    if ($LASTEXITCODE -ne 0) { Get-Content $evalLog -Tail 20; throw "evaluation failed (see $evalLog)" }
    $resultMd = Get-ChildItem tests\real\results -Filter ("*_" + $tag + "_*.md") | Sort-Object LastWriteTime | Select-Object -Last 1
    Write-Host ("result: " + $resultMd.FullName)
}

# ---------------------------------------------------------------- 4. report
Step "4. report"
$report = Join-Path $work "report.md"
$rargs = @("tools\retrain_report.py", "--inventory", $invJson, "--inventory-txt", $invTxt, "--train", $trainJson, "--out", $report)
if ($resultMd) { $rargs += @("--result", $resultMd.FullName) }
& $py @rargs
if ($LASTEXITCODE -ne 0) { throw "report failed" }
Write-Host "`nreport : $report"
Write-Host "model  : $model (current models\clubhead.onnx untouched)"
