param(
    [string]$SourceRoot = "C:\mamba\data\imagenet",
    [string]$TargetRootWsl = "/home/lkr/data/imagenet",
    [string]$RunId = "",
    [switch]$Background,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = "C:\mamba"

function Convert-WindowsPathStringToWsl {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert Windows path string to WSL form: $Path"
}

if (-not $RunId) {
    $RunId = "local_imagenet_ext4_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

$train = Join-Path $SourceRoot "train"
$val = Join-Path $SourceRoot "val"
if (-not (Test-Path $train) -or -not (Test-Path $val)) {
    throw "Missing ImageNet train/val folders under $SourceRoot"
}

$sourceWsl = Convert-WindowsPathStringToWsl (Resolve-Path $SourceRoot).Path
$runRoot = Join-Path $repoRoot "artifacts\data_prep\$RunId"
$launcherWin = Join-Path $runRoot "sync_imagenet_ext4.sh"
$launcherWsl = Convert-WindowsPathStringToWsl $launcherWin
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$launcher = @"
#!/usr/bin/env bash
set -euo pipefail
src="$sourceWsl"
dst="$TargetRootWsl"
mkdir -p "`$dst"
echo "[`$(date '+%F %T')] sync train -> `$dst/train"
rsync -a --info=progress2 --partial "`$src/train/" "`$dst/train/"
echo "[`$(date '+%F %T')] sync val -> `$dst/val"
rsync -a --info=progress2 --partial "`$src/val/" "`$dst/val/"
train_classes=`$(find "`$dst/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
val_classes=`$(find "`$dst/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "train_classes=`$train_classes val_classes=`$val_classes"
if [[ "`$train_classes" -ne 1000 || "`$val_classes" -ne 1000 ]]; then
  echo "ImageNet ext4 copy incomplete" >&2
  exit 2
fi
echo "[`$(date '+%F %T')] ready: `$dst" | tee "`$dst/READY.txt"
"@
[System.IO.File]::WriteAllText($launcherWin, ($launcher -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)

Write-Host "Local ImageNet ext4 preparation"
Write-Host ("Source        {0}" -f $SourceRoot)
Write-Host ("Target WSL    {0}" -f $TargetRootWsl)
Write-Host ("Run root      {0}" -f $runRoot)
Write-Host "ETA           first full sync usually 1-4 hours on this dataset; later rsync resumes are incremental"
Write-Host "Resource note CPU/disk busy during sync; GPU training can continue but may see slower filesystem access"
if ($DryRun) {
    Write-Host ("Launcher      {0}" -f $launcherWin)
    return
}

$args = @("-d", "Ubuntu-24.04", "--", "bash", $launcherWsl)
if ($Background) {
    Start-Process -FilePath "wsl.exe" -ArgumentList $args -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot "sync.log") -RedirectStandardError (Join-Path $runRoot "sync.err") | Out-Null
    Write-Host "Started in background."
    Write-Host ("Log           {0}" -f (Join-Path $runRoot "sync.log"))
} else {
    wsl @args
}
