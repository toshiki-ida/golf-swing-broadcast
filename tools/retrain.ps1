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
    [string]$Previous = "",
    [switch]$Curve,
    [switch]$SkipEval
)
# -Previous <inventory.json> : show the delta against a previous inventory
# -Curve                     : also train with 25/50/75% of the training clips (learning curve) and evaluate all models
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
# written by python as UTF-8 (Tee-Object would write UTF-16, which the report step cannot read)
$invArgs = ' tools\dataset_inventory.py --dataset "' + $Dataset + '" --json "' + $invJson + '" --exclude-holdout --estimate-camera'
if ($Previous -and (Test-Path $Previous)) { $invArgs += ' --previous "' + (Resolve-Path $Previous).Path + '"' }
cmd.exe /c ('"' + $py + '"' + $invArgs + ' > "' + $invTxt + '" 2>&1')
if ($LASTEXITCODE -ne 0) { Get-Content $invTxt -Encoding UTF8 -Tail 20; throw "inventory failed" }
Get-Content $invTxt -Encoding UTF8
$inv = Get-Content $invJson -Raw -Encoding UTF8 | ConvertFrom-Json
if ($inv.n_strong_usable -lt 20) { throw "not enough usable strong labels ($($inv.n_strong_usable)); collect more before retraining" }

# ---------------------------------------------------------------- 1b. trajectory prior (template for filling unobserved spans)
Step "1b. trajectory prior (models	rajectory_prior_<angle>.json)"
cmd.exe /c ('"' + $py + '" toolsuild_prior.py --dataset "' + $Dataset + '" > "' + (Join-Path $work "prior.txt") + '" 2>&1')
if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $work "prior.txt") -Encoding UTF8 -Tail 10; throw "build_prior failed" }
Get-Content (Join-Path $work "prior.txt") -Encoding UTF8 | Select-String -Pattern "^(face_on|dtl|unknown)" | ForEach-Object { Write-Host ("  " + $_.Line) }

# ---------------------------------------------------------------- 2. train
Step "2. train (val-ratio $ValRatio, epochs $Epochs)"
$trainLog = Join-Path $work "train.log"
$fractions = @(1.0)
if ($Curve) { $fractions = @(0.25, 0.5, 0.75, 1.0) }
$models = @()
$trainJsons = @()
foreach ($frac in $fractions) {
    $before = Get-ChildItem models -Filter "clubhead_*.onnx" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $log = Join-Path $work ("train_f" + [int]($frac * 100) + ".log")
    $targs = @("tools\train_clubhead.py", "--dataset", $Dataset, "--val-ratio", "$ValRatio", "--epochs", "$Epochs", "--train-fraction", "$frac")
    if ($Name) { $targs += @("--name", $Name) }
    Write-Host ("  training with " + [int]($frac * 100) + "% of the training clips ...")
    cmd.exe /c ('"' + $TorchPython + '" ' + ($targs -join ' ') + ' > "' + $log + '" 2>&1')
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 20; throw "training failed (see $log)" }
    $after = Get-ChildItem models -Filter "clubhead_*.onnx" | Where-Object { $before -notcontains $_.FullName } | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $after) { throw "no model produced for fraction $frac" }
    $models += $after.FullName
    $trainJsons += [IO.Path]::ChangeExtension($after.FullName, ".train.json")
    Get-Content $log | Select-String -Pattern "^epoch|device=|fraction" | Select-Object -Last 3 | ForEach-Object { Write-Host ("  " + $_.Line.Substring(0, [Math]::Min(160, $_.Line.Length))) }
}
$model = $models[-1]
$trainJson = $trainJsons[-1]
$trainLog = Join-Path $work "train_f100.log"
Write-Host "model: $model"

# ---------------------------------------------------------------- 3. evaluate
$resultMd = $null
if (-not $SkipEval) {
    Step "3. evaluate on $Split : legacy / current clubhead.onnx / $($after.BaseName)"
    $evalLog = Join-Path $work "eval.log"
    $tag = "retrain_" + $stamp
    $modelArgs = ($models | ForEach-Object { '--model "' + $_ + '"' }) -join ' '
    cmd.exe /c ('"' + $py + '" tools\eval_club.py --all --split ' + $Split + ' --tag ' + $tag + ' --detector both ' + $modelArgs + ' --no-run > "' + $evalLog + '" 2>&1')
    if ($LASTEXITCODE -ne 0) { Get-Content $evalLog -Tail 20; throw "evaluation failed (see $evalLog)" }
    $resultMd = Get-ChildItem tests\real\results -Filter ("*_" + $tag + "_*.md") | Sort-Object LastWriteTime | Select-Object -Last 1
    Write-Host ("result: " + $resultMd.FullName)
}

# ---------------------------------------------------------------- 4. report
Step "4. report"
$report = Join-Path $work "report.md"
$rargs = @("tools\retrain_report.py", "--inventory", $invJson, "--inventory-txt", $invTxt, "--out", $report)
foreach ($tj in $trainJsons) { $rargs += @("--train", $tj) }
if ($resultMd) { $rargs += @("--result", $resultMd.FullName) }
& $py @rargs
if ($LASTEXITCODE -ne 0) { throw "report failed" }
Write-Host "`nreport : $report"
Write-Host "model  : $model (current models\clubhead.onnx untouched)"
