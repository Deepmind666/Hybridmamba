param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$RunId = "",
    [string]$ResumeFrom = "",
    [int]$GpuMemGb = 22,
    [int]$TorchNumThreads = 1,
    [int]$InteropThreads = 1,
    [int]$Niceness = 15,
    [string]$CpuCoreList = "0-7",
    [int]$GpuPowerLimitW = 300,
    [int]$GuardGpuUtilPct = 80,
    [int]$GuardCpuUtilPct = 80,
    [int]$GuardResumeUtilPct = 65,
    [int]$GuardTempC = 76,
    [int]$GuardMemoryPct = 88,
    [int]$GuardCooldownSec = 30,
    [switch]$AllowBlockedHost,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$guardScript = Join-Path $repoRoot "scripts\check_local_host_stability.ps1"
$launcher = Join-Path $repoRoot "scripts\start_local_training_blackwell.ps1"

if (-not (Test-Path $guardScript)) {
    throw "Missing host guard: $guardScript"
}
if (-not (Test-Path $launcher)) {
    throw "Missing launcher: $launcher"
}

$resolvedConfig = Resolve-Path $ConfigPath
$guard = & $guardScript -EmitJson | ConvertFrom-Json

if ($guard.status -eq "blocked" -and -not $AllowBlockedHost) {
    throw "Local host is blocked. Re-run with -AllowBlockedHost only if you explicitly accept the stability risk."
}

if (-not $RunId) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baseName = [IO.Path]::GetFileNameWithoutExtension($resolvedConfig.Path)
    $RunId = "{0}_adaptive_{1}" -f $baseName, $stamp
}

$launcherArgs = @{
    ConfigPath = $resolvedConfig.Path
    RunId = $RunId
    GpuMemGb = $GpuMemGb
    TorchNumThreads = $TorchNumThreads
    InteropThreads = $InteropThreads
    Niceness = $Niceness
    CpuCoreList = $CpuCoreList
    GpuPowerLimitW = $GpuPowerLimitW
    SlowMode = $true
    AdaptiveGuard = $true
    GuardGpuUtilPct = $GuardGpuUtilPct
    GuardCpuUtilPct = $GuardCpuUtilPct
    GuardResumeUtilPct = $GuardResumeUtilPct
    GuardTempC = $GuardTempC
    GuardMemoryPct = $GuardMemoryPct
    GuardCooldownSec = $GuardCooldownSec
}

if ($ResumeFrom) {
    $launcherArgs.ResumeFrom = (Resolve-Path $ResumeFrom).Path
}

Write-Host ("Guard status      {0}" -f $guard.status)
Write-Host ("Allow blocked     {0}" -f $AllowBlockedHost.IsPresent)
Write-Host ("Config            {0}" -f $resolvedConfig.Path)
Write-Host ("Run id            {0}" -f $RunId)
Write-Host ("Safety profile    gpu_mem={0}G threads={1}/{2} nice={3} cpu={4} power={5}W" -f $GpuMemGb, $TorchNumThreads, $InteropThreads, $Niceness, $CpuCoreList, $GpuPowerLimitW)
Write-Host ("Adaptive guard    gpu<={0}% cpu<={1}% resume<={2}% temp<={3}C mem<={4}% cooldown={5}s" -f $GuardGpuUtilPct, $GuardCpuUtilPct, $GuardResumeUtilPct, $GuardTempC, $GuardMemoryPct, $GuardCooldownSec)
Write-Host "Progress command:"
Write-Host ("  powershell -NoProfile -ExecutionPolicy Bypass -File C:\mamba\scripts\watch_progress_local_fat.ps1 -LocalRunId {0} -RefreshSeconds 30" -f $RunId)

if ($DryRun) {
    Write-Host "Dry run only; no training launched."
    return
}

& $launcher @launcherArgs
