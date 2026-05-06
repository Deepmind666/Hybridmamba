param(
    [string]$LocalRunId = "",
    [string]$FatRunId = "",
    [string]$LocalResumeFrom = "C:\mamba\artifacts\runs\local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923\last.pth",
    [string]$FatResumeFrom = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/runs/fat_aitodv2_hybridmambadet_fusion05_resume_mem14_20260502_0923/last.pth",
    [int]$LocalGpuMemGb = 10,
    [int]$FatGpuMemGb = 14,
    [switch]$SkipSync,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$localConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_72e_aitodv2_stable.py"
$fatConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_72e_aitodv2_fusion05_stable.py"
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
    $LocalRunId = "local_aitodv2_hybridmamba_base_72e_resume_mem${LocalGpuMemGb}_$stamp"
}
if (-not $FatRunId) {
    $FatRunId = "fat_aitodv2_hybridmambadet_fusion05_72e_resume_mem${FatGpuMemGb}_$stamp"
}

Write-Host "72e AI-TOD-v2 dual pair"
Write-Host ("  Local config : {0}" -f $localConfig)
Write-Host ("  Local run id : {0}" -f $LocalRunId)
Write-Host ("  Local resume : {0}" -f $LocalResumeFrom)
Write-Host ("  Fat config   : {0}" -f $fatConfig)
Write-Host ("  Fat run id   : {0}" -f $FatRunId)
Write-Host ("  Fat resume   : {0}" -f $FatResumeFrom)
Write-Host ("  Cap          : 72 epochs max, early stop enabled.")
Write-Host ("  ETA          : local about 2-4 days to cap from current checkpoint; Fat about 2-3 days, but early stop may cut that shorter.")

if ($DryRun) {
    Write-Host "Dry run only; no sync or training launched."
    return
}

if ($SkipSync) {
    Write-Host "Skipping full FatMachine sync; assuming 72e configs and launcher are already present."
} else {
    Write-Host "Syncing current repo/config/data bundle to FatMachine..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $fatSync
}

Write-Host "Launching local HybridMamba_Base_B 72e resume..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $localLauncher `
    -ConfigPath $localConfig `
    -RunId $LocalRunId `
    -ResumeFrom $LocalResumeFrom `
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

Write-Host "Launching Fat HybridMambaDet fusion05 72e resume..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $fatLauncher `
    -ConfigPath $fatConfig `
    -RunId $FatRunId `
    -ResumeFrom $FatResumeFrom `
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

Write-Host "Launched 72e AI-TOD-v2 dual pair."
