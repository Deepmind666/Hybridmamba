param(
    [string]$LocalRunId = "",
    [string]$FatRunId = "",
    [int]$LocalGpuMemGb = 8,
    [int]$FatGpuMemGb = 14,
    [ValidateSet("tinyvim", "hybridbase")]
    [string]$LocalModel = "tinyvim",
    [ValidateSet("hybriddet")]
    [string]$FatModel = "hybriddet",
    [switch]$SkipSync,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$localConfig = if ($LocalModel -eq "hybridbase") {
    Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_72e_aitodv2_tinyproto.py"
} else {
    Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_72e_aitodv2_tinyproto.py"
}
$fatConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_72e_aitodv2_fusion05_tinyproto.py"
$localLauncher = Join-Path $repoRoot "scripts\start_local_training_blackwell_adaptive.ps1"
$fatLauncher = Join-Path $repoRoot "scripts\start_fatmachine_run_blackwell.ps1"

foreach ($path in @($localConfig, $fatConfig, $localLauncher, $fatLauncher)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $LocalRunId) {
    $LocalRunId = "local_aitodv2_${LocalModel}_tinyproto_72e_mem${LocalGpuMemGb}_$stamp"
}
if (-not $FatRunId) {
    $FatRunId = "fat_aitodv2_hybridmambadet_fusion05_tinyproto_72e_mem${FatGpuMemGb}_$stamp"
}

Write-Host "AI-TOD-v2 tiny-object protocol v1 dual pair"
Write-Host ("  Local config : {0}" -f $localConfig)
Write-Host ("  Local run id : {0}" -f $LocalRunId)
Write-Host ("  Fat config   : {0}" -f $fatConfig)
Write-Host ("  Fat run id   : {0}" -f $FatRunId)
Write-Host "  Resume       : none; start from pretrained TinyViM backbone for a clean protocol comparison."
Write-Host "  Protocol     : smaller anchors, lower IoU assignment thresholds, nms_pre=3000, max_per_img=1000."
Write-Host "  ETA          : first validation usually 2-5h per machine under guard; cap can be several days, early stop may cut earlier."

if ($DryRun) {
    Write-Host "Dry run only; no sync or training launched."
    return
}

if ($SkipSync) {
    Write-Host "Skipping FatMachine file sync."
} else {
    Write-Host "Syncing tinyproto configs and docs to FatMachine..."
    ssh FatMachine "powershell -NoProfile -Command `"New-Item -ItemType Directory -Force C:\Users\sshuser\codex_runs\hybrid-mamba\code\tinyvim\detection\configs_v3 | Out-Null; New-Item -ItemType Directory -Force C:\Users\sshuser\codex_runs\hybrid-mamba\docs | Out-Null`""
    scp "$repoRoot\code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_72e_aitodv2_tinyproto.py" "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/code/tinyvim/detection/configs_v3/"
    scp "$repoRoot\code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_72e_aitodv2_tinyproto.py" "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/code/tinyvim/detection/configs_v3/"
    scp "$repoRoot\code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_72e_aitodv2_fusion05_tinyproto.py" "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/code/tinyvim/detection/configs_v3/"
    scp "$repoRoot\docs\aitodv2-tiny-object-protocol.md" "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/docs/"
}

Write-Host "Launching local tinyproto run..."
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

Write-Host "Launching Fat tinyproto run..."
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

Write-Host "Launched AI-TOD-v2 tiny-object protocol v1 dual pair."
