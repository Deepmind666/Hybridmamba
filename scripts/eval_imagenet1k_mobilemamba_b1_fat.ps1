param(
    [string]$DatasetRoot = "C:\Users\sshuser\data\imagenet",
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$CheckpointPath = "C:\Users\sshuser\codex_runs\hybrid-mamba\external\mobilemamba\weights\MobileMamba_B1\mobilemamba_b1.pth",
    [string]$RunId = "",
    [string]$ConfigPath = "configs/mobilemamba/mobilemamba_b1.py",
    [int]$TestBatchSize = 125,
    [int]$NumWorkersPerGpu = 4,
    [int]$TorchNumThreads = 2,
    [int]$InteropThreads = 1,
    [int]$Niceness = 5,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$localRoot = "C:\mamba"
$remoteRootUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba"
$remoteEnv = "$remoteRootUnix/.mamba-env-cu128"
$remoteMicromamba = "$remoteRootUnix/artifacts/tools/micromamba"

function Convert-WindowsPathStringToWsl {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert Windows path string to WSL form: $Path"
}

function Quote-CfgValue {
    param([Parameter(Mandatory = $true)]$Value)
    if ($Value -is [string]) {
        $escaped = $Value.Replace("'", "''")
        return "'$escaped'"
    }
    return [string]$Value
}

function Invoke-FatPowerShellJson {
    param([Parameter(Mandatory = $true)][string]$Script)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    $raw = ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded"
    if ($LASTEXITCODE -ne 0) {
        throw "FatMachine PowerShell command failed."
    }
    $json = ($raw -split "`r?`n" | Where-Object { $_ -match '^\{' } | Select-Object -First 1)
    if (-not $json) {
        throw "FatMachine command returned no JSON."
    }
    return $json | ConvertFrom-Json
}

function Get-RemoteReadiness {
    param(
        [Parameter(Mandatory = $true)][string]$Dataset,
        [Parameter(Mandatory = $true)][string]$Checkpoint
    )
    $remoteScript = @"
`$ProgressPreference = 'SilentlyContinue'
`$dataset = '$Dataset'
`$checkpoint = '$Checkpoint'
`$train = Join-Path `$dataset 'train'
`$val = Join-Path `$dataset 'val'
`$trainClasses = if (Test-Path `$train) { @(Get-ChildItem -Path `$train -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
`$valClasses = if (Test-Path `$val) { @(Get-ChildItem -Path `$val -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
`$psLines = & wsl.exe -d Ubuntu-24.04 -u ns3user -- bash -lc "ps -eo pid,etimes,pcpu,pmem,args | grep -E 'external/mobilemamba|run.py' | grep -v grep || true"
[pscustomobject]@{
  DatasetReady = ((Test-Path `$train) -and (Test-Path `$val))
  TrainClasses = `$trainClasses
  ValClasses = `$valClasses
  CheckpointReady = (Test-Path `$checkpoint)
  ActiveTrain = [bool]((`$psLines -join "`n") -match 'run.py.*-m train')
  ActiveLines = @(`$psLines)
} | ConvertTo-Json -Compress
"@
    return Invoke-FatPowerShellJson $remoteScript
}

if (-not (Test-Path (Join-Path $localRoot "external\mobilemamba\run.py"))) {
    throw "Missing MobileMamba entrypoint under $localRoot"
}

if (-not $RunId) {
    $RunId = "eval_mobilemamba_b1_imagenet1k_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

$readiness = Get-RemoteReadiness -Dataset $DatasetRoot -Checkpoint $CheckpointPath
if (-not $readiness.DatasetReady) {
    throw "Missing ImageNet-1K train/val folders on Fat: $DatasetRoot"
}
if (-not $readiness.CheckpointReady) {
    throw "Missing MobileMamba-B1 checkpoint on Fat: $CheckpointPath"
}
if ($readiness.ActiveTrain -and -not $Force) {
    Write-Warning "Active MobileMamba training process observed on Fat. Dry-run is allowed; actual eval will require -Force."
    if (-not $DryRun) {
        throw "Active MobileMamba training process observed on Fat. This script will not steal resources unless you pass -Force."
    }
}

$runRootWin = Join-Path $RemoteRoot "artifacts\eval\$RunId"
$runRootWsl = Convert-WindowsPathStringToWsl $runRootWin
$datasetWsl = Convert-WindowsPathStringToWsl $DatasetRoot
$ckptWsl = Convert-WindowsPathStringToWsl $CheckpointPath
$remoteTempRootWin = "C:\Users\sshuser\AppData\Local\Temp"
$remoteLauncherSh = Join-Path $remoteTempRootWin ("eval_mobilemamba_b1_{0}.sh" -f $RunId)
$localLauncherSh = Join-Path $env:TEMP ("eval_mobilemamba_b1_{0}.sh" -f $RunId)
$launcherShWsl = Convert-WindowsPathStringToWsl $remoteLauncherSh

$optArgs = @(
    "data.type=$(Quote-CfgValue 'DefaultCLS')",
    "data.root=$(Quote-CfgValue $datasetWsl)",
    "trainer.checkpoint=$(Quote-CfgValue $runRootWsl)",
    "trainer.resume_dir=$(Quote-CfgValue '')",
    "trainer.data.batch_size_test=$(Quote-CfgValue $TestBatchSize)",
    "trainer.data.num_workers_per_gpu=$(Quote-CfgValue $NumWorkersPerGpu)",
    "model.model_kwargs.checkpoint_path=$(Quote-CfgValue $ckptWsl)"
)
$optArgLine = $optArgs -join " "

$launcherShTemplate = @'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAMBA_TRITON_INTERPRET=0
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=__TORCH_THREADS__
export MKL_NUM_THREADS=__TORCH_THREADS__
export OPENBLAS_NUM_THREADS=__TORCH_THREADS__
export TORCH_NUM_THREADS=__TORCH_THREADS__
export TORCH_NUM_INTEROP_THREADS=__INTEROP_THREADS__
mkdir -p "__RUN_ROOT__"
cd "__REMOTE_ROOT__/external/mobilemamba"
exec nice -n __NICENESS__ "__REMOTE_MICROMAMBA__" run -p "__REMOTE_ENV__" python run.py -c "__CFG_PATH__" -m test __OPT_ARGS__ > "__RUN_ROOT__/eval.log" 2>&1
'@

$launcherShContent = $launcherShTemplate
foreach ($replacement in @(
    @{ Old = '__TORCH_THREADS__'; New = "$TorchNumThreads" },
    @{ Old = '__INTEROP_THREADS__'; New = "$InteropThreads" },
    @{ Old = '__RUN_ROOT__'; New = $runRootWsl },
    @{ Old = '__REMOTE_ROOT__'; New = $remoteRootUnix },
    @{ Old = '__NICENESS__'; New = "$Niceness" },
    @{ Old = '__REMOTE_MICROMAMBA__'; New = $remoteMicromamba },
    @{ Old = '__REMOTE_ENV__'; New = $remoteEnv },
    @{ Old = '__CFG_PATH__'; New = $ConfigPath },
    @{ Old = '__OPT_ARGS__'; New = $optArgLine }
)) {
    $launcherShContent = $launcherShContent.Replace($replacement.Old, [string]$replacement.New)
}

[System.IO.File]::WriteAllText($localLauncherSh, ($launcherShContent -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)

Write-Host "MobileMamba-B1 quick reproduction eval on Fat"
Write-Host ("Run id        {0}" -f $RunId)
Write-Host ("Run root      {0}" -f $runRootWin)
Write-Host ("Checkpoint    {0}" -f $CheckpointPath)
Write-Host ("Dataset       {0}" -f $DatasetRoot)
Write-Host ("Dataset check train_classes={0} val_classes={1}" -f $readiness.TrainClasses, $readiness.ValClasses)
Write-Host "Expected      Top-1 79.948, Top-5 94.924 for the official MobileMamba-B1 checkpoint"
Write-Host "ETA           15-45 min when run without concurrent training; longer if -Force is used during training"
if ($DryRun) {
    Write-Host ("Local staged  {0}" -f $localLauncherSh)
    Write-Host ("Remote script {0}" -f $remoteLauncherSh)
    return
}

$remoteShScp = "FatMachine:" + $remoteLauncherSh.Replace("\", "/")
scp $localLauncherSh $remoteShScp | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy launcher shell script to Fat: $remoteLauncherSh"
}

$remoteLaunch = @"
`$session = 'eval_$RunId'
`$sh = '$launcherShWsl'
& wsl.exe -d Ubuntu-24.04 -u ns3user -- tmux new-session -d -s `$session "bash '`$sh'"
if (`$LASTEXITCODE -ne 0) { throw "Failed to create tmux session: `$session" }
"@
$encodedLaunch = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteLaunch))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encodedLaunch" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start Fat MobileMamba eval launcher."
}

Write-Host ("Launched on Fat. Log: {0}" -f (Join-Path $runRootWin "eval.log"))
