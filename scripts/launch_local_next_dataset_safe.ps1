param(
    [ValidateSet("tinyvim_aitodv2_ultrasafe", "tinyvim_aitodv2", "hybrid_aitodv2")]
    [string]$Preset = "tinyvim_aitodv2_ultrasafe",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = "C:\mamba"
$launcher = Join-Path $repoRoot "scripts\start_local_training_blackwell.ps1"

if (-not (Test-Path $launcher)) {
    throw "Missing launcher: $launcher"
}

switch ($Preset) {
    "tinyvim_aitodv2_ultrasafe" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_120e_aitodv2_local_ultrasafe.py"
    }
    "tinyvim_aitodv2" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_120e_aitodv2_local.py"
    }
    "hybrid_aitodv2" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_aitodv2.py"
    }
}

if (-not (Test-Path $configPath)) {
    throw "Config not found: $configPath"
}

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runTag = "{0}_{1}_safe" -f $Preset, $runStamp

$safeGpuMem = 14
$safeThreads = 1
$safeInterop = 1
$safeNiceness = 15
$safePowerW = 300

if ($Preset -ne "tinyvim_aitodv2_ultrasafe") {
    $safeGpuMem = 20
    $safeThreads = 2
    $safeInterop = 1
    $safeNiceness = 12
    $safePowerW = 380
}

$cmdArgs = @{
    ConfigPath = $configPath
    RunId = $runTag
    GpuMemGb = $safeGpuMem
    TorchNumThreads = $safeThreads
    InteropThreads = $safeInterop
    Niceness = $safeNiceness
    GpuPowerLimitW = $safePowerW
    SlowMode = $true
    AdaptiveGuard = $true
    GuardGpuUtilPct = 80
    GuardCpuUtilPct = 80
    GuardResumeUtilPct = 65
    GuardTempC = 76
    GuardMemoryPct = 88
    GuardCooldownSec = 30
}

Write-Host ("Preset            {0}" -f $Preset)
Write-Host ("Config            {0}" -f $configPath)
Write-Host ("Run id            {0}" -f $runTag)
Write-Host ("Safety profile    gpu_mem={0}G threads={1} interop={2} nice={3} power={4}W" -f $safeGpuMem, $safeThreads, $safeInterop, $safeNiceness, $safePowerW)
Write-Host ("Dry run           {0}" -f $DryRun.IsPresent)

if ($DryRun) {
    Write-Host ("Would run: {0} -ConfigPath {1} -RunId {2} -GpuMemGb {3} -TorchNumThreads {4} -InteropThreads {5} -Niceness {6} -GpuPowerLimitW {7} -SlowMode -AdaptiveGuard" -f $launcher, $configPath, $runTag, $safeGpuMem, $safeThreads, $safeInterop, $safeNiceness, $safePowerW)
    return
}

& $launcher @cmdArgs
