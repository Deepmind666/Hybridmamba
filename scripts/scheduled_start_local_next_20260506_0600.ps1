param(
    [string]$RunId = "local_imagenet1k_tinyvim_b_100e_20260506_0600",
    [string]$DatasetRoot = "C:\mamba\data\imagenet",
    [string]$LogRoot = "C:\mamba\artifacts\scheduled"
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
$log = Join-Path $LogRoot "local_next_20260506_0600.log"

function Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message
    Add-Content -Path $log -Value $line
    Write-Output $line
}

function Test-ImageNetReady {
    param([string]$Root)
    $train = Join-Path $Root "train"
    $val = Join-Path $Root "val"
    $trainTar = Join-Path $Root "train.tar"
    $valTar = Join-Path $Root "val.tar"
    return (((Test-Path $train) -and (Test-Path $val)) -or ((Test-Path $trainTar) -and (Test-Path $valTar)))
}

Log "Scheduled local next experiment guard started."
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\mamba\scripts\preflight_imagenet1k_next.ps1 -SkipFat *> (Join-Path $LogRoot "local_next_20260506_0600_preflight.log")

$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match "local_visdrone2019_tinyvim_b_100e_stable_mem22_20260504_2233|start_imagenet1k_tinyvim_local|code/tinyvim/main.py" -and
    $_.CommandLine -notmatch "scheduled_start_local_next_20260506_0600"
})
if ($active.Count -gt 0) {
    Log ("SKIP: observed active local training-related processes: {0}" -f (($active | Select-Object -First 5 -ExpandProperty ProcessId) -join ","))
    exit 0
}

if (-not (Test-ImageNetReady $DatasetRoot)) {
    Log ("SKIP: ImageNet-1K data is not ready under {0}. Need train/val folders or train.tar/val.tar." -f $DatasetRoot)
    exit 0
}

Log "Launching local TinyViM-B ImageNet-1K run."
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\mamba\scripts\start_imagenet1k_tinyvim_local.ps1 -RunId $RunId -DatasetRoot $DatasetRoot *> (Join-Path $LogRoot "local_next_20260506_0600_launch.log")
Log "Launch command finished."
