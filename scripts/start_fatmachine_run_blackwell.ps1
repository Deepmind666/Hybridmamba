param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [int]$GpuMemGb = 24,
    [int]$TorchNumThreads = 4,
    [int]$InteropThreads = 1,
    [string]$ResumeFrom = "",
    [switch]$AdaptiveGuard,
    [int]$GuardGpuUtilPct = 80,
    [int]$GuardCpuUtilPct = 80,
    [int]$GuardResumeUtilPct = 70,
    [int]$GuardTempC = 78,
    [int]$GuardMemoryPct = 92,
    [int]$GuardCheckIntervalSec = 2,
    [int]$GuardCooldownSec = 20,
    [switch]$Background
)

$localRoot = "C:\mamba"
$remoteRunDir = Join-Path $RemoteRoot "artifacts\runs\$RunId"
$relativeConfig = Resolve-Path $ConfigPath | ForEach-Object { $_.Path.Replace($localRoot, "").TrimStart('\') }
$remoteConfigUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/" + ($relativeConfig.Replace("\", "/"))
$remoteResumeArg = if ($ResumeFrom) { " --resume-from $ResumeFrom" } else { "" }
$remoteGuardArg = if ($AdaptiveGuard) {
    " --adaptive-guard --guard-gpu-util-pct $GuardGpuUtilPct --guard-cpu-util-pct $GuardCpuUtilPct --guard-resume-util-pct $GuardResumeUtilPct --guard-temp-c $GuardTempC --guard-memory-pct $GuardMemoryPct --guard-check-interval-sec $GuardCheckIntervalSec --guard-cooldown-sec $GuardCooldownSec"
} else {
    ""
}

$launcher = Join-Path $env:TEMP "hybrid_mamba_run_blackwell_$RunId.cmd"
$launcherSh = Join-Path $env:TEMP "hybrid_mamba_run_blackwell_$RunId.sh"
$remoteLauncherShWin = "C:\Users\sshuser\codex_runs\hybrid-mamba\run_$RunId.sh"
$remoteLauncherShUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/run_$RunId.sh"
$remoteRunDirUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/runs/$RunId"

$launcherShContent = @"
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$remoteRunDirUnix"
export MAMBA_LOCAL_SAFE_MODE=1
export GPU_MEM_GB=$GpuMemGb
export OMP_NUM_THREADS=$TorchNumThreads
export MKL_NUM_THREADS=$TorchNumThreads
export OPENBLAS_NUM_THREADS=$TorchNumThreads
export TORCH_NUM_THREADS=$TorchNumThreads
export TORCH_NUM_INTEROP_THREADS=$InteropThreads
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTORCH_NVML_BASED_CUDA_CHECK=1
cd /mnt/c/Users/sshuser/codex_runs/hybrid-mamba
exec /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/tools/micromamba run -p /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/.mamba-env-cu128 python scripts/run_train_mmdet3_manual.py $remoteConfigUnix --work-dir "$remoteRunDirUnix" --gpu-mem-gb $GpuMemGb --torch-num-threads $TorchNumThreads --torch-num-interop-threads $InteropThreads$remoteGuardArg$remoteResumeArg > "$remoteRunDirUnix/launcher.log" 2>&1
"@
$launcherShLf = $launcherShContent -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($launcherSh, $launcherShLf, [System.Text.Encoding]::ASCII)

$launcherContent = @"
@echo off
mkdir "$remoteRunDir" 2>nul
wsl -d Ubuntu-24.04 -u ns3user -- bash "$remoteLauncherShUnix"
"@
Set-Content -Path $launcher -Value $launcherContent -Encoding ASCII

scp $launcherSh "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/run_$RunId.sh"
scp $launcher "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/run_$RunId.cmd"
$remoteLauncher = "C:\Users\sshuser\codex_runs\hybrid-mamba\run_$RunId.cmd"
if ($Background) {
    Start-Process -FilePath "ssh" -ArgumentList @("FatMachine", "cmd /c $remoteLauncher") -WindowStyle Hidden
    Write-Host ("Started FatMachine run in background: {0}" -f $RunId)
    Write-Host ("Remote run dir: {0}" -f $remoteRunDir)
} else {
    ssh FatMachine "cmd /c $remoteLauncher"
}
