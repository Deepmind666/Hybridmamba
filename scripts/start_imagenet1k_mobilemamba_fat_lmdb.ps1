param(
    [string]$DataRootWsl = "/home/ns3user/data/imagenet_lmdb",
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$RunId = "",
    [string]$ConfigPath = "configs/mobilemamba/mobilemamba_b1.py",
    [int]$BatchSize = 256,
    [int]$Epochs = 100,
    [int]$NumWorkersPerGpu = 2,
    [int]$TestBatchSize = 125,
    [int]$TorchNumThreads = 2,
    [int]$InteropThreads = 1,
    [int]$Niceness = 5,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
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
        return "'" + $Value.Replace("'", "''") + "'"
    }
    if ($Value -is [bool]) {
        return $(if ($Value) { "True" } else { "False" })
    }
    return [string]$Value
}

if (-not $RunId) {
    $RunId = "mobilemamba_b1_imagenet1k_lmdb_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

$readyScript = "test -f '$DataRootWsl/train.lmdb/data.mdb' && test -f '$DataRootWsl/val.lmdb/data.mdb' && echo ready || echo missing"
$ready = ssh FatMachine "wsl -d Ubuntu-24.04 -u ns3user -- bash -lc `"$readyScript`""
if (($ready -notmatch "ready") -and -not $DryRun) {
    throw "LMDB data is not ready on Fat: $DataRootWsl. Run scripts\prepare_imagenet_lmdb_fat.ps1 first."
}

$runRootWin = Join-Path $RemoteRoot "artifacts\runs\$RunId"
$runRootWsl = Convert-WindowsPathStringToWsl $runRootWin
$epochFull = $Epochs
$warmupEpochs = if ($Epochs -ge 300) { 20 } else { [Math]::Max(5, [Math]::Round($Epochs * 0.07)) }
$testStartEpoch = if ($Epochs -ge 300) { 200 } else { 1 }
$testPerEpoch = if ($Epochs -ge 300) { 5 } else { 1 }
$savePerEpoch = if ($Epochs -ge 300) { 15 } else { 5 }
$effectiveLr = [Math]::Round(0.0015 * $BatchSize / 512.0, 6)

$optArgs = @(
    "data.type=$(Quote-CfgValue 'ImageFolderLMDB')",
    "data.root=$(Quote-CfgValue $DataRootWsl)",
    "batch_size=$(Quote-CfgValue $BatchSize)",
    "trainer.data.batch_size=$(Quote-CfgValue $BatchSize)",
    "trainer.data.batch_size_test=$(Quote-CfgValue $TestBatchSize)",
    "trainer.data.num_workers_per_gpu=$(Quote-CfgValue $NumWorkersPerGpu)",
    "trainer.data.persistent_workers=$(Quote-CfgValue $true)",
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

$remoteLauncherSh = "C:\Users\sshuser\AppData\Local\Temp\mobilemamba_lmdb_$RunId.sh"
$localLauncherSh = Join-Path $env:TEMP ("mobilemamba_lmdb_$RunId.sh")
$launcherShWsl = Convert-WindowsPathStringToWsl $remoteLauncherSh
$launcher = @"
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAMBA_TRITON_INTERPRET=0
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=$TorchNumThreads
export MKL_NUM_THREADS=$TorchNumThreads
export OPENBLAS_NUM_THREADS=$TorchNumThreads
export TORCH_NUM_THREADS=$TorchNumThreads
export TORCH_NUM_INTEROP_THREADS=$InteropThreads
mkdir -p "$runRootWsl"
cd "$remoteRootUnix/external/mobilemamba"
exec nice -n $Niceness "$remoteMicromamba" run -p "$remoteEnv" python run.py -c "$ConfigPath" -m train $optArgLine > "$runRootWsl/launcher.log" 2>&1
"@
[System.IO.File]::WriteAllText($localLauncherSh, ($launcher -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)

Write-Host "MobileMamba-B1 LMDB accelerated training on Fat"
Write-Host ("Run id        {0}" -f $RunId)
Write-Host ("Run root      {0}" -f $runRootWin)
Write-Host ("Data WSL      {0}" -f $DataRootWsl)
Write-Host ("Batch size    {0}" -f $BatchSize)
Write-Host ("Workers       {0}" -f $NumWorkersPerGpu)
Write-Host ("Effective lr  {0}" -f $effectiveLr)
Write-Host "ETA           after LMDB is ready, expect data wait to drop substantially; first epoch target under about 1 hour if I/O is fixed"
if ($DryRun) {
    Write-Host ("Local staged  {0}" -f $localLauncherSh)
    Write-Host ("Remote script {0}" -f $remoteLauncherSh)
    return
}

scp $localLauncherSh ("FatMachine:" + $remoteLauncherSh.Replace("\", "/")) | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy MobileMamba LMDB launcher to Fat"
}
$remoteLaunch = @"
`$session = 'mamba_$RunId'
& wsl.exe -d Ubuntu-24.04 -u ns3user -- tmux new-session -d -s `$session "bash '$launcherShWsl'"
if (`$LASTEXITCODE -ne 0) { throw "Failed to create tmux session: `$session" }
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteLaunch))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded" | Out-Null
Write-Host "Launched Fat LMDB training."
