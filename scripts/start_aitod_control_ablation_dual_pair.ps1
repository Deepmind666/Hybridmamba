param(
    [string]$LocalRunId = "",
    [string]$FatRunId = "",
    [int]$LocalGpuMemGb = 10,
    [int]$FatGpuMemGb = 14,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$localConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_120e_aitodv2_stable.py"
$fatConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_120e_aitodv2_fusion05_stable.py"
$localLauncher = Join-Path $repoRoot "scripts\start_local_training_blackwell_adaptive.ps1"
$fatLauncher = Join-Path $repoRoot "scripts\start_fatmachine_run_blackwell.ps1"
$fatSync = Join-Path $repoRoot "scripts\sync_to_fatmachine_blackwell.ps1"

foreach ($path in @($localConfig, $fatConfig, $localLauncher, $fatLauncher, $fatSync)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $LocalRunId) {
    $LocalRunId = "local_aitodv2_hybridmamba_base_control_mem${LocalGpuMemGb}_$stamp"
}
if (-not $FatRunId) {
    $FatRunId = "fat_aitodv2_hybridmambadet_fusion05_mem${FatGpuMemGb}_$stamp"
}

Write-Host "Next AI-TOD-v2 dual pair"
Write-Host ("  Local config : {0}" -f $localConfig)
Write-Host ("  Local run id : {0}" -f $LocalRunId)
Write-Host ("  Fat config   : {0}" -f $fatConfig)
Write-Host ("  Fat run id   : {0}" -f $FatRunId)
Write-Host ("  ETA          : local first validation 2.5-4h; Fat first validation 1.5-2.5h after launch.")
Write-Host ("  Early stop   : likely 12-20h for a weak run with patience=12 and val_interval=1.")
Write-Host ("  Resource plan: local GPU_MEM_GB={0}, CPU cores 0-1, power 220W; Fat GPU_MEM_GB={1}, guard memory 90%." -f $LocalGpuMemGb, $FatGpuMemGb)

if ($DryRun) {
    Write-Host "Dry run only; no sync or training launched."
    return
}

Write-Host "Syncing current repo/config/data bundle to FatMachine..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $fatSync

Write-Host "Launching local HybridMamba_Base_B control..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $localLauncher `
    -ConfigPath $localConfig `
    -RunId $LocalRunId `
    -GpuMemGb $LocalGpuMemGb `
    -TorchNumThreads 1 `
    -InteropThreads 1 `
    -Niceness 18 `
    -CpuCoreList "0-1" `
    -GpuPowerLimitW 220 `
    -GuardGpuUtilPct 70 `
    -GuardCpuUtilPct 60 `
    -GuardResumeUtilPct 50 `
    -GuardTempC 68 `
    -GuardMemoryPct 68 `
    -GuardCooldownSec 120 `
    -AllowBlockedHost

Write-Host "Launching Fat HybridMambaDet fusion05 ablation..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $fatLauncher `
    -ConfigPath $fatConfig `
    -RunId $FatRunId `
    -GpuMemGb $FatGpuMemGb `
    -TorchNumThreads 2 `
    -InteropThreads 1 `
    -AdaptiveGuard `
    -GuardGpuUtilPct 80 `
    -GuardCpuUtilPct 75 `
    -GuardResumeUtilPct 65 `
    -GuardTempC 74 `
    -GuardMemoryPct 90 `
    -GuardCheckIntervalSec 2 `
    -GuardCooldownSec 90 `
    -Background

Write-Host "Launched next AI-TOD-v2 dual pair."
Write-Host ("  Watch local: powershell -NoProfile -ExecutionPolicy Bypass -File C:\mamba\scripts\watch_progress_local_fat.ps1 -LocalRunId {0} -FatRunId {1} -IncludeFat -RefreshSeconds 30" -f $LocalRunId, $FatRunId)
