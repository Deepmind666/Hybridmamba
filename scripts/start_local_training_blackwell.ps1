param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$RunId,
    [string]$ResumeFrom = "",
    [int]$GpuMemGb = 24,
    [int]$TorchNumThreads = 4,
    [int]$InteropThreads = 1,
    [switch]$AdaptiveGuard,
    [int]$GuardGpuUtilPct = 80,
    [int]$GuardCpuUtilPct = 80,
    [int]$GuardResumeUtilPct = 70,
    [int]$GuardTempC = 78,
    [int]$GuardMemoryPct = 92,
    [int]$GuardCheckIntervalSec = 2,
    [int]$GuardCooldownSec = 20,
    [int]$Niceness = 5,
    [string]$CpuCoreList = "",
    [int]$GpuPowerLimitW = 0,
    [string]$Distro = "Ubuntu-24.04",
    [switch]$UseGameTurbo,
    [switch]$StopDynamicTuning,
    [switch]$SlowMode
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$resolvedConfig = Resolve-Path $ConfigPath

if ($SlowMode) {
    if (-not $PSBoundParameters.ContainsKey("GpuMemGb")) {
        $GpuMemGb = 24
    }
    if (-not $PSBoundParameters.ContainsKey("TorchNumThreads")) {
        $TorchNumThreads = 4
    }
    if (-not $PSBoundParameters.ContainsKey("InteropThreads")) {
        $InteropThreads = 1
    }
    if (-not $PSBoundParameters.ContainsKey("Niceness")) {
        $Niceness = 10
    }
}

if ($UseGameTurbo -or $StopDynamicTuning) {
    & (Join-Path $repoRoot "scripts\prepare_local_training_host.ps1") `
        -UseGameTurbo:$UseGameTurbo `
        -StopDynamicTuning:$StopDynamicTuning
}

if ($GpuPowerLimitW -gt 0 -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    try {
        & nvidia-smi -pl $GpuPowerLimitW | Out-Null
    } catch {
        Write-Host ("GPU power limit   failed to set {0} W" -f $GpuPowerLimitW)
    }
}

if (-not $RunId) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmm"
    $baseName = [IO.Path]::GetFileNameWithoutExtension($resolvedConfig.Path)
    $RunId = "{0}_{1}" -f $baseName, $stamp
}

$runDirWin = Join-Path $repoRoot "artifacts\runs\$RunId"
if (Test-Path $runDirWin) {
    throw "Run directory already exists: $runDirWin"
}
New-Item -ItemType Directory -Force $runDirWin | Out-Null

$relativeConfig = $resolvedConfig.Path.Replace($repoRoot, "").TrimStart("\")
$configUnix = "/mnt/c/mamba/" + ($relativeConfig.Replace("\", "/"))
$runDirUnix = "/mnt/c/mamba/artifacts/runs/$RunId"
$launcher = Join-Path $env:TEMP "hybrid_mamba_local_blackwell_$RunId.cmd"
$resumeArg = ""
if ($ResumeFrom) {
    $resolvedResume = Resolve-Path $ResumeFrom
    $relativeResume = $resolvedResume.Path.Replace($repoRoot, "").TrimStart("\")
    $resumeUnix = "/mnt/c/mamba/" + ($relativeResume.Replace("\", "/"))
    $resumeArg = "--resume-from $resumeUnix"
}
$adaptiveGuardFlag = if ($AdaptiveGuard) { 1 } else { 0 }

$launcherContent = @"
@echo off
mkdir "$runDirWin" 2>nul
wsl -d $Distro bash -lc "export MAMBA_LOCAL_SAFE_MODE=1; export MAMBA_TRAIN_ENTRY=scripts/run_train_mmdet3_manual.py; export GPU_MEM_GB=$GpuMemGb; export OMP_NUM_THREADS=$TorchNumThreads; export MKL_NUM_THREADS=$TorchNumThreads; export OPENBLAS_NUM_THREADS=$TorchNumThreads; export TORCH_NUM_THREADS=$TorchNumThreads; export TORCH_NUM_INTEROP_THREADS=$InteropThreads; export CPU_CORE_LIST=$CpuCoreList; export MAMBA_ADAPTIVE_GUARD=$adaptiveGuardFlag; export MAMBA_GUARD_GPU_UTIL_PCT=$GuardGpuUtilPct; export MAMBA_GUARD_CPU_UTIL_PCT=$GuardCpuUtilPct; export MAMBA_GUARD_RESUME_UTIL_PCT=$GuardResumeUtilPct; export MAMBA_GUARD_TEMP_C=$GuardTempC; export MAMBA_GUARD_MEMORY_PCT=$GuardMemoryPct; export MAMBA_GUARD_CHECK_INTERVAL_SEC=$GuardCheckIntervalSec; export MAMBA_GUARD_COOLDOWN_SEC=$GuardCooldownSec; export CUDA_DEVICE_MAX_CONNECTIONS=1; export MALLOC_ARENA_MAX=2; cd /mnt/c/mamba && nice -n $Niceness scripts/run_local_training_blackwell.sh $configUnix $runDirUnix $resumeArg > $runDirUnix/launcher.log 2>&1"
"@
Set-Content -Path $launcher -Value $launcherContent -Encoding ASCII

Start-Process -FilePath "cmd.exe" -ArgumentList "/c start `"`" /b /belownormal `"$launcher`"" -WindowStyle Hidden | Out-Null

Write-Host ("Run id            {0}" -f $RunId)
Write-Host ("Run dir           {0}" -f $runDirWin)
Write-Host ("Config            {0}" -f $resolvedConfig.Path)
Write-Host ("GPU_MEM_GB        {0}" -f $GpuMemGb)
Write-Host ("TORCH_NUM_THREADS {0}" -f $TorchNumThreads)
Write-Host ("TORCH_INTEROP     {0}" -f $InteropThreads)
Write-Host ("nice              {0}" -f $Niceness)
Write-Host ("CPU cores         {0}" -f $(if ($CpuCoreList) { $CpuCoreList } else { "all available" }))
Write-Host ("Adaptive guard    {0}" -f $(if ($AdaptiveGuard) { "enabled" } else { "disabled" }))
if ($AdaptiveGuard) {
    Write-Host ("Guard thresholds  gpu<={0}% cpu<={1}% resume<={2}% temp<={3}C mem<={4}% cooldown={5}s" -f $GuardGpuUtilPct, $GuardCpuUtilPct, $GuardResumeUtilPct, $GuardTempC, $GuardMemoryPct, $GuardCooldownSec)
}
Write-Host ("GPU power limit   {0}" -f $(if ($GpuPowerLimitW -gt 0) { "$GpuPowerLimitW W" } else { "unchanged" }))
Write-Host ("Resume from       {0}" -f $(if ($ResumeFrom) { $ResumeFrom } else { "none" }))
Write-Host ("Slow mode         {0}" -f $(if ($SlowMode) { "enabled" } else { "disabled" }))
Write-Host "Local safe mode   enabled"
Write-Host ("Power plan        {0}" -f $(if ($UseGameTurbo) { "GameTurbo" } else { "unchanged" }))
Write-Host ("Dynamic tuning    {0}" -f $(if ($StopDynamicTuning) { "stopped" } else { "unchanged" }))
Write-Host "Local run launched."
