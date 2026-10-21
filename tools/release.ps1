# tools/release.ps1 -- build the onedir exe, run the AI selftest, zip, replace the draft GitHub release asset,
# update handoff/VERSION.txt and the release notes.
#
#   powershell -ExecutionPolicy Bypass -File tools\release.ps1 [-Tag v0.2.0-ai-preview] [-SkipBuild] [-NoUpload]
#
# Requirements: clean working tree (the commit id goes into the exe name), .venv with pyinstaller,
# ffmpeg in C:\ffmpeg\bin or on PATH (bundled by the spec), GitHub CLI (token from Git Credential Manager).
# Release notes body = docs/RELEASE_NOTES.md (UTF-8). The script only prepends the build header.
# The script is ASCII-only on purpose (Windows PowerShell 5.1 reads BOM-less files as ANSI).

param(
    [string]$Tag = "v0.2.0-ai-preview",
    [string]$Sample = "handoff\samples\rec_0611_131029_trim.mp4",
    [switch]$SkipBuild,
    [switch]$NoUpload
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$gh = "C:\Program Files\GitHub CLI\gh.exe"

function Step($msg) { Write-Host ("`n=== " + $msg) -ForegroundColor Cyan }

# ---------------------------------------------------------------- 0. preconditions
Step "preconditions"
$dirty = git status --porcelain
if ($dirty) { throw "Working tree is not clean. Commit first:`n$dirty" }
$sha = (git rev-parse --short HEAD).Trim()
$fullSha = (git rev-parse HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -eq "main") { throw "Do not release from main; use a feature branch" }
$media = git ls-files | Select-String -Pattern '\.(mp4|mov|jpg|jpeg|png|onnx|pt|pth|ttf|otf)$'
if ($media) { throw "Tracked media/model/font files found (public repo):`n$($media -join "`n")" }
if (-not (Test-Path $py)) { throw "venv python not found: $py" }
if (-not (Test-Path $Sample)) { throw "sample video not found: $Sample" }
Write-Host "branch=$branch commit=$sha"

# ---------------------------------------------------------------- 1. build
$dist = Join-Path $root "dist\GolfSwingBroadcast"
$exe = Join-Path $dist "GolfSwingBroadcast.exe"
if (-not $SkipBuild) {
    Step "build (PyInstaller onedir)"
    Get-Process GolfSwingBroadcast -ErrorAction SilentlyContinue | Stop-Process -Force
    if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
    & $py -m PyInstaller GolfSwingBroadcast.spec --noconfirm --distpath (Join-Path $root "dist") --workpath (Join-Path $root "build\pyinstaller")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
}
if (-not (Test-Path $exe)) { throw "exe not found: $exe" }

Step "bundle check"
$required = @(
    "_internal\ffmpeg.exe", "_internal\ffprobe.exe",
    "_internal\models\pose_landmarker_heavy.task",
    "_internal\config\rules.yaml",
    "_internal\assets\fonts"
)
foreach ($r in $required) {
    if (-not (Test-Path (Join-Path $dist $r))) { throw "missing in bundle: $r" }
    Write-Host ("  ok  " + $r)
}
$ort = Get-ChildItem -Path (Join-Path $dist "_internal") -Recurse -Filter "onnxruntime*.dll" | Select-Object -First 1
if (-not $ort) { throw "missing in bundle: onnxruntime DLL" }
Write-Host ("  ok  " + $ort.FullName.Substring($dist.Length + 1))
$clubOnnx = Join-Path $dist "_internal\models\clubhead.onnx"
$hasDetector = Test-Path $clubOnnx
if ($hasDetector) { Write-Host "  ok  _internal\models\clubhead.onnx (detector bundled)" }
else { Write-Host "  --  _internal\models\clubhead.onnx not present (detector disabled, legacy tracking)" -ForegroundColor Yellow }
$exeFolderMb = [math]::Round((Get-ChildItem $dist -Recurse | Measure-Object Length -Sum).Sum / 1MB)

# ---------------------------------------------------------------- 2. selftest (frozen exe)
Step "selftest"
$st = Join-Path $root ("build\selftest_" + $sha)
if (Test-Path $st) { Remove-Item -Recurse -Force $st }
$t0 = Get-Date
$p = Start-Process -FilePath $exe -ArgumentList @("--ai-selftest", (Resolve-Path $Sample).Path, $st) -Wait -PassThru -NoNewWindow
$selftestSec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
$resultPath = Join-Path $st "selftest_result.json"
if (-not (Test-Path $resultPath)) { throw "selftest produced no result (exit $($p.ExitCode))" }
$result = Get-Content $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($result.status -ne "OK") { throw "selftest failed: $($result.status) $($result.error)" }
Write-Host ("  status=OK frozen=" + $result.frozen + " seconds=" + $selftestSec)
$labelsLine = "n/a"
if ($result.labels) {
    $labelsLine = "dir=" + $result.labels.dir + " clips=" + $result.labels.n_clips + " strong=" + $result.labels.n_strong + " weak=" + $result.labels.n_weak
    if ($result.labels.dir -like "$dist*") { throw "labels were saved inside the exe folder: $($result.labels.dir)" }
}
Write-Host ("  labels: " + $labelsLine)
if ($result.club_detector) { Write-Host ("  club_detector available=" + $result.club_detector.available) }

# ---------------------------------------------------------------- 3. zip + handoff folder
Step "zip"
$handoff = Join-Path $root "handoff"
New-Item -ItemType Directory -Force $handoff | Out-Null
Get-ChildItem $handoff -Filter "GolfSwingBroadcast-ai-*.zip" | Remove-Item -Force
$zipName = "GolfSwingBroadcast-ai-$sha-win64.zip"
$zip = Join-Path $handoff $zipName
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($dist, $zip, [IO.Compression.CompressionLevel]::Optimal, $true)
$zipMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "  $zipName  $zipMb MB"
# expanded copy for USB
$hcopy = Join-Path $handoff "GolfSwingBroadcast"
robocopy $dist $hcopy /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
Write-Host "  handoff\GolfSwingBroadcast refreshed"

# ---------------------------------------------------------------- 4. VERSION.txt
Step "VERSION.txt"
$pyver = (& $py -c "import sys; print(sys.version.split()[0])").Trim()
$pkgs = (& $py -c "import importlib.metadata as m; print(', '.join(f'{n} {m.version(n)}' for n in ['pyinstaller','mediapipe','onnxruntime','opencv-contrib-python','customtkinter']))").Trim()
$ffver = (& (Join-Path $dist "_internal\ffmpeg.exe") -version | Select-Object -First 1) -replace '^ffmpeg version (\S+).*', '$1'
$built = Get-Date -Format "yyyy-MM-dd HH:mm"
$detLine = "none (legacy tracking)"
if ($hasDetector) { $detLine = "models/clubhead.onnx bundled (experimental; ai.club_detector=off to disable)" }
$version = @(
    "GolfSwingBroadcast ($branch)",
    "commit: $sha",
    "built: $built",
    "build PC: Windows 11, Python $pyver, $pkgs, ffmpeg $ffver",
    "exe folder: $exeFolderMb MB, zip: $zipName ($zipMb MB)",
    "selftest: OK ($(Split-Path -Leaf $Sample), $selftestSec s, frozen=$($result.frozen))",
    "labels: $labelsLine",
    "club detector: $detLine"
) -join "`r`n"
[IO.File]::WriteAllText((Join-Path $handoff "VERSION.txt"), $version + "`r`n", (New-Object System.Text.UTF8Encoding $false))
Write-Host $version

# ---------------------------------------------------------------- 5. release notes + GitHub draft release
$notesSrc = Join-Path $root "docs\RELEASE_NOTES.md"
if (-not (Test-Path $notesSrc)) { throw "docs\RELEASE_NOTES.md not found" }
$header = "**Draft (private).** feature branch ``$branch`` @ ``$sha`` -- onedir build, built $built. Asset: ``$zipName`` ($zipMb MB, $exeFolderMb MB expanded). Selftest OK ($selftestSec s, frozen exe)."
$notes = $header + "`r`n`r`n" + (Get-Content $notesSrc -Raw -Encoding UTF8)
$notesTmp = Join-Path $root "build\release_notes_$sha.md"
[IO.File]::WriteAllText($notesTmp, $notes, (New-Object System.Text.UTF8Encoding $false))

if ($NoUpload) {
    Step "NoUpload: skipping push and GitHub release"
    Write-Host "zip: $zip ($zipMb MB)  commit: $sha"
    exit 0
}

Step "push branch"
git push origin $branch
if ($LASTEXITCODE -ne 0) { throw "git push failed" }

Step "GitHub draft release $Tag"
$cred = "protocol=https`nhost=github.com`n`n" | git credential fill
$tok = ($cred | Where-Object { $_ -like "password=*" } | Select-Object -First 1)
if (-not $tok) { throw "no GitHub token from git credential fill" }
$env:GH_TOKEN = $tok.Substring(9)
$assets = & $gh release view $Tag --json assets --jq ".assets[].name"
if ($LASTEXITCODE -ne 0) { throw "release $Tag not found" }
foreach ($a in $assets) {
    if ($a) { Write-Host "  delete old asset $a"; & $gh release delete-asset $Tag $a -y }
}
& $gh release upload $Tag $zip
if ($LASTEXITCODE -ne 0) { throw "asset upload failed" }
& $gh release edit $Tag --draft --target $fullSha --title "AI preview ($branch @ $sha) [draft]" --notes-file $notesTmp
if ($LASTEXITCODE -ne 0) { throw "release edit failed" }
$url = (& $gh release view $Tag --json url --jq ".url").Trim()
$remoteSize = (& $gh release view $Tag --json assets --jq ".assets[0].size").Trim()

Step "done"
Write-Host "release : $url"
Write-Host "asset   : $zipName  $zipMb MB (remote $remoteSize bytes)"
Write-Host "commit  : $sha ($branch)"
Write-Host "VERSION : handoff\VERSION.txt"
