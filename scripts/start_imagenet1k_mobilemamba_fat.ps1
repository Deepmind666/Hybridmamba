param(
    [string]$DatasetRoot = "C:\Users\sshuser\data\imagenet",
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$RunId = "",
    [string]$ConfigPath = "configs/mobilemamba/mobilemamba_b1.py",
    [int]$BatchSize = 256,
    [int]$Epochs = 100,
    [int]$NumWorkersPerGpu = 8,
    [int]$TestBatchSize = 125,
    [int]$TorchNumThreads = 2,
    [int]$InteropThreads = 1,
    [int]$Niceness = 5,
    [switch]$Background,
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
    if ($null -eq $Value) {
        return "''"
    }
    if ($Value -is [string]) {
        $escaped = $Value.Replace("'", "''")
        return "'$escaped'"
    }
    if ($Value -is [bool]) {
        return $(if ($Value) { "True" } else { "False" })
    }
    return [string]$Value
}

function Get-RemoteImageNetLayout {
    param([string]$Root)
    $remoteScript = @"
`$ProgressPreference = 'SilentlyContinue'
`$root = '$Root'
`$train = Join-Path `$root 'train'
`$val = Join-Path `$root 'val'
`$hasFolders = (Test-Path `$train) -and (Test-Path `$val)
`$trainClasses = if (Test-Path `$train) { @(Get-ChildItem -Path `$train -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
`$valClasses = if (Test-Path `$val) { @(Get-ChildItem -Path `$val -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
[pscustomobject]@{
  RootExists = (Test-Path `$root)
  HasFolders = `$hasFolders
  Ready = `$hasFolders
  TrainClasses = `$trainClasses
  ValClasses = `$valClasses
  Detail = ("folders={0} train_classes={1} val_classes={2}" -f `$hasFolders, `$trainClasses, `$valClasses)
} | ConvertTo-Json -Compress
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
    $json = ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded"
    $json = ($json -split "`r?`n" | Where-Object { $_ -match '^\{' } | Select-Object -First 1)
    if (-not $json) {
        throw "Could not read Fat ImageNet layout for $Root"
    }
    return $json | ConvertFrom-Json
}

if (-not (Test-Path (Join-Path $localRoot "external\mobilemamba\run.py"))) {
    throw "Missing MobileMamba entrypoint under $localRoot"
}

if (-not $RunId) {
    $RunId = "mobilemamba_b1_imagenet1k_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

$runRootWin = Join-Path $RemoteRoot "artifacts\runs\$RunId"
$runRootWsl = Convert-WindowsPathStringToWsl $runRootWin
$datasetWsl = Convert-WindowsPathStringToWsl $DatasetRoot
$datasetLayout = Get-RemoteImageNetLayout $DatasetRoot
if (-not $datasetLayout.Ready -and -not $DryRun) {
    throw "Missing usable ImageNet-1K train/val folders on Fat: $DatasetRoot. Current: $($datasetLayout.Detail)"
}

$epochFull = $Epochs
$warmupEpochs = if ($Epochs -ge 300) { 20 } else { [Math]::Max(5, [Math]::Round($Epochs * 0.07)) }
$testStartEpoch = if ($Epochs -ge 300) { 200 } else { 1 }
$testPerEpoch = if ($Epochs -ge 300) { 5 } else { 1 }
$savePerEpoch = if ($Epochs -ge 300) { 15 } else { 5 }

$optArgs = @(
    "data.type=$(Quote-CfgValue 'DefaultCLS')",
    "data.root=$(Quote-CfgValue $datasetWsl)",
    "batch_size=$(Quote-CfgValue $BatchSize)",
    "trainer.data.batch_size=$(Quote-CfgValue $BatchSize)",
    "trainer.data.batch_size_test=$(Quote-CfgValue $TestBatchSize)",
    "trainer.data.num_workers_per_gpu=$(Quote-CfgValue $NumWorkersPerGpu)",
    "epoch_full=$(Quote-CfgValue $epochFull)",
    "trainer.epoch_full=$(Quote-CfgValue $epochFull)",
    "warmup_epochs=$(Quote-CfgValue $warmupEpochs)",
    "trainer.scheduler_kwargs.warmup_epochs=$(Quote-CfgValue $warmupEpochs)",
    "trainer.test_start_epoch=$(Quote-CfgValue $testStartEpoch)",
    "trainer.test_per_epoch=$(Quote-CfgValue $testPerEpoch)",
    "trainer.save_per_epoch=$(Quote-CfgValue $savePerEpoch)",
    "trainer.checkpoint=$(Quote-CfgValue $runRootWsl)",
    "trainer.resume_dir=$(Quote-CfgValue '')"
)

$optArgLine = $optArgs -join " "
$effectiveLr = [Math]::Round(0.0015 * $BatchSize / 512.0, 6)

$remoteTempRootWin = "C:\Users\sshuser\AppData\Local\Temp"
$remoteLauncherSh = Join-Path $remoteTempRootWin ("mobilemamba_imagenet1k_{0}.sh" -f $RunId)
$remoteLauncherCmd = Join-Path $remoteTempRootWin ("mobilemamba_imagenet1k_{0}.cmd" -f $RunId)
$localLauncherSh = Join-Path $env:TEMP ("mobilemamba_imagenet1k_{0}.sh" -f $RunId)
$localLauncherCmd = Join-Path $env:TEMP ("mobilemamba_imagenet1k_{0}.cmd" -f $RunId)
$launcherShWsl = Convert-WindowsPathStringToWsl $remoteLauncherSh

$launcherShTemplate = @'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=__TORCH_THREADS__
export MKL_NUM_THREADS=__TORCH_THREADS__
export OPENBLAS_NUM_THREADS=__TORCH_THREADS__
export TORCH_NUM_THREADS=__TORCH_THREADS__
export TORCH_NUM_INTEROP_THREADS=__INTEROP_THREADS__
mkdir -p "__RUN_ROOT__"
cd "__REMOTE_ROOT__/external/mobilemamba"
exec nice -n __NICENESS__ "__REMOTE_MICROMAMBA__" run -p "__REMOTE_ENV__" python run.py -c "__CFG_PATH__" -m train __OPT_ARGS__ > "__RUN_ROOT__/launcher.log" 2>&1
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

$launcherCmdContent = @"
@echo off
wsl -d Ubuntu-24.04 -u ns3user -- bash "$launcherShWsl"
"@

[System.IO.File]::WriteAllText($localLauncherSh, ($launcherShContent -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)
[System.IO.File]::WriteAllText($localLauncherCmd, $launcherCmdContent, [System.Text.Encoding]::ASCII)

if ($DryRun) {
    Write-Host ("Run id            {0}" -f $RunId)
    Write-Host ("Run root          {0}" -f $runRootWin)
    Write-Host ("Dataset root      {0}" -f $DatasetRoot)
    Write-Host ("Dataset layout    {0}" -f $datasetLayout.Detail)
    Write-Host ("Batch size        {0}" -f $BatchSize)
    Write-Host ("Epochs            {0}" -f $Epochs)
    Write-Host ("Warmup epochs     {0}" -f $warmupEpochs)
    Write-Host ("Test start epoch  {0}" -f $testStartEpoch)
    Write-Host ("Effective lr      {0}" -f $effectiveLr)
    Write-Host ("Launcher script   {0}" -f $remoteLauncherCmd)
    Write-Host ("Local staging     {0}" -f $localLauncherCmd)
    return
}

$remoteShScp = "FatMachine:" + $remoteLauncherSh.Replace("\", "/")
$remoteCmdScp = "FatMachine:" + $remoteLauncherCmd.Replace("\", "/")
scp $localLauncherSh $remoteShScp | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy launcher shell script to Fat: $remoteLauncherSh"
}
scp $localLauncherCmd $remoteCmdScp | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy launcher cmd script to Fat: $remoteLauncherCmd"
}

$remoteLaunch = @"
`$session = 'mamba_$RunId'
`$sh = '$launcherShWsl'
& wsl.exe -d Ubuntu-24.04 -u ns3user -- tmux new-session -d -s `$session "bash '`$sh'"
if (`$LASTEXITCODE -ne 0) { throw "Failed to create tmux session: `$session" }
"@
$encodedLaunch = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteLaunch))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encodedLaunch" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start Fat launcher via SSH: $remoteLauncherCmd"
}

Write-Host ("Run id            {0}" -f $RunId)
Write-Host ("Run root          {0}" -f $runRootWin)
Write-Host ("Dataset root      {0}" -f $DatasetRoot)
Write-Host ("Batch size        {0}" -f $BatchSize)
Write-Host ("Epochs            {0}" -f $Epochs)
Write-Host ("Warmup epochs     {0}" -f $warmupEpochs)
Write-Host ("Test start epoch  {0}" -f $testStartEpoch)
Write-Host ("Effective lr      {0}" -f $effectiveLr)
Write-Host ("Launcher script   {0}" -f $remoteLauncherCmd)
Write-Host "Fat MobileMamba ImageNet-1K run launched."
