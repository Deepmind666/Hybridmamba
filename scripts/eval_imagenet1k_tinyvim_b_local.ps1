param(
    [string]$DataRoot = "C:\mamba\data\imagenet",
    [string]$CheckpointPath = "C:\mamba\weights\tinyvim\tinyvim_b_300e.pth",
    [string]$RunId = "",
    [int]$BatchSize = 256,
    [int]$NumWorkers = 8,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$LocalRoot = "C:\mamba"
$Micromamba = "/home/lkr/.local/bin/micromamba"
$EnvPath = "/mnt/c/mamba/.mamba-env-cu128"

function Convert-WindowsPathToWsl {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert Windows path to WSL path: $Path"
}

function Assert-ImageNetLayout {
    param([Parameter(Mandatory = $true)][string]$Root)
    $train = Join-Path $Root "train"
    $val = Join-Path $Root "val"
    if (-not (Test-Path $train) -or -not (Test-Path $val)) {
        throw "ImageNet train/val folders are missing under $Root"
    }
}

if (-not $RunId) {
    $RunId = "eval_tinyvim_b_imagenet1k_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

Assert-ImageNetLayout $DataRoot
if (-not (Test-Path $CheckpointPath)) {
    throw "Missing TinyViM-B checkpoint: $CheckpointPath"
}

$active = wsl -d Ubuntu-24.04 -- bash -lc "ps -eo args | grep -Ei 'code/tinyvim/main.py|TinyViM_B' | grep -v grep || true"
if ($active -and -not $Force) {
    Write-Warning "Active TinyViM process observed. Dry-run is allowed; actual eval will require -Force."
    if (-not $DryRun) {
        throw "Active TinyViM process observed. This script will not steal resources unless you pass -Force."
    }
}

$runRootWin = Join-Path $LocalRoot "artifacts\eval\$RunId"
$runRootWsl = Convert-WindowsPathToWsl $runRootWin
$dataWsl = Convert-WindowsPathToWsl $DataRoot
$ckptWsl = Convert-WindowsPathToWsl $CheckpointPath
$launcherWin = Join-Path $runRootWin "eval_tinyvim_b.sh"
$launcherWsl = Convert-WindowsPathToWsl $launcherWin

New-Item -ItemType Directory -Force -Path $runRootWin | Out-Null

$launcher = @"
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/mamba
mkdir -p "$runRootWsl"
exec "$Micromamba" run -p "$EnvPath" python code/tinyvim/main.py \
  --eval \
  --model TinyViM_B \
  --batch-size $BatchSize \
  --num_workers $NumWorkers \
  --input-size 224 \
  --data-set IMNET \
  --data-path "$dataWsl" \
  --resume "$ckptWsl" \
  --distillation-type none \
  --output_dir "$runRootWsl" \
  > "$runRootWsl/eval.log" 2>&1
"@
[System.IO.File]::WriteAllText($launcherWin, ($launcher -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)

Write-Host "TinyViM-B quick reproduction eval"
Write-Host ("Run id       {0}" -f $RunId)
Write-Host ("Run root     {0}" -f $runRootWin)
Write-Host ("Checkpoint   {0}" -f $CheckpointPath)
Write-Host ("Dataset      {0}" -f $DataRoot)
Write-Host ("Expected     Top-1 about 81.2 for the official 300e TinyViM-B checkpoint")
Write-Host "ETA          20-45 min when run without concurrent training; longer if -Force is used during training"
if ($DryRun) {
    Write-Host ("Launcher     {0}" -f $launcherWin)
    return
}

wsl -d Ubuntu-24.04 -- bash $launcherWsl
Write-Host ("Done. Log: {0}" -f (Join-Path $runRootWin "eval.log"))
