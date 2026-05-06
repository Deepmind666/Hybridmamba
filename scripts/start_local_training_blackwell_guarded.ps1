param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [ValidateSet("smoke", "formal")][string]$Mode = "formal",
    [string]$RunId,
    [int]$GpuMemGb = 24,
    [int]$TorchNumThreads = 4,
    [int]$InteropThreads = 1,
    [string]$CpuCoreList = "",
    [string]$Distro = "Ubuntu-24.04",
    [switch]$AllowSmokeWhenBlocked
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$checkScript = Join-Path $repoRoot "scripts\check_local_host_stability.ps1"
$launcherScript = Join-Path $repoRoot "scripts\run_local_training_blackwell.sh"

if (-not (Test-Path $checkScript)) {
    throw "Missing host stability script: $checkScript"
}
if (-not (Test-Path $launcherScript)) {
    throw "Missing local launcher: $launcherScript"
}

$resolvedConfig = Resolve-Path $ConfigPath
if (-not $resolvedConfig.Path.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ConfigPath must live under $repoRoot"
}

$guard = & $checkScript -EmitJson | ConvertFrom-Json
if ($Mode -eq "formal" -and $guard.status -ne "eligible") {
    throw "Local formal training blocked by host guard. Status=$($guard.status). Use FatMachine for this run."
}
if ($Mode -eq "smoke" -and $guard.status -eq "blocked" -and -not $AllowSmokeWhenBlocked) {
    throw "Local smoke training blocked by host guard. Status=blocked."
}
if ($Mode -eq "smoke" -and $guard.status -eq "blocked" -and $AllowSmokeWhenBlocked) {
    Write-Warning "AllowSmokeWhenBlocked: launching short local smoke despite host guard blocked."
}

if (-not $RunId) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baseName = [IO.Path]::GetFileNameWithoutExtension($resolvedConfig.Path)
    $RunId = "{0}_{1}_{2}" -f $baseName, $Mode, $stamp
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

$launcherContent = @"
@echo off
mkdir "$runDirWin" 2>nul
wsl -d $Distro bash -lc "export MAMBA_HOST_GUARD_APPROVED=1; export MAMBA_LOCAL_SAFE_MODE=1; export MAMBA_TRAIN_ENTRY=scripts/run_train_mmdet3_manual.py; export GPU_MEM_GB=$GpuMemGb; export OMP_NUM_THREADS=$TorchNumThreads; export MKL_NUM_THREADS=$TorchNumThreads; export OPENBLAS_NUM_THREADS=$TorchNumThreads; export TORCH_NUM_THREADS=$TorchNumThreads; export TORCH_NUM_INTEROP_THREADS=$InteropThreads; export CPU_CORE_LIST=$CpuCoreList; export CUDA_DEVICE_MAX_CONNECTIONS=1; export MALLOC_ARENA_MAX=2; cd /mnt/c/mamba && nice -n 10 scripts/run_local_training_blackwell.sh $configUnix $runDirUnix > $runDirUnix/launcher.log 2>&1"
"@
Set-Content -Path $launcher -Value $launcherContent -Encoding ASCII

Start-Process -FilePath $launcher | Out-Null

Write-Host ("Guard status      {0}" -f $guard.status)
Write-Host ("Launch mode       {0}" -f $Mode)
Write-Host ("Run id            {0}" -f $RunId)
Write-Host ("Run dir           {0}" -f $runDirWin)
Write-Host ("Config            {0}" -f $resolvedConfig.Path)
Write-Host ("GPU_MEM_GB        {0}" -f $GpuMemGb)
Write-Host ("TORCH_NUM_THREADS {0}" -f $TorchNumThreads)
Write-Host ("TORCH_INTEROP     {0}" -f $InteropThreads)
Write-Host ("CPU cores         {0}" -f $(if ($CpuCoreList) { $CpuCoreList } else { "all available" }))
Write-Host "Local safe mode   enabled"
Write-Host "Host tweaks       disabled"
Write-Host "Local run launched through the guarded entry point."
