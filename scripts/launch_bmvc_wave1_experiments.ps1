<#
.SYNOPSIS
  BMVC wave-1: local training + Fat formal HybridMambaDet aligned to TinyViM 300e VisDrone-V2.

.DESCRIPTION
  - Local (default -LocalMode formal): stable full run `retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py` via
    `start_local_training_blackwell.ps1` (does not use the host-stability gate; you accept machine risk).
  - Local (-LocalMode smoke): 3-epoch smoke via `start_recommended_training_blackwell.ps1 -Mode smoke` (honours guard unless -ForceLocalSmokeWhenBlocked).
  - Fat: same formal config via sync + remote launcher.

  Prerequisites: converted VisDrone COCO annotations, `weights/tinyvim/tinyvim_b_300e.pth`, SSH host `FatMachine`.

.EXAMPLE
  .\scripts\launch_bmvc_wave1_experiments.ps1 -LocalOnly -DryRun
  .\scripts\launch_bmvc_wave1_experiments.ps1 -LocalOnly -LocalMode formal
  .\scripts\launch_bmvc_wave1_experiments.ps1 -LocalOnly -LocalMode smoke -ForceLocalSmokeWhenBlocked
  .\scripts\launch_bmvc_wave1_experiments.ps1 -FatOnly -SyncFirst
  .\scripts\launch_bmvc_wave1_experiments.ps1 -All
#>
param(
    [switch]$LocalOnly,
    [switch]$FatOnly,
    [switch]$All,
    [switch]$SyncFirst,
    [switch]$DryRun,
    [switch]$ForceLocalSmokeWhenBlocked,
    [ValidateSet("smoke", "formal")]
    [string]$LocalMode = "formal"
)

$ErrorActionPreference = "Stop"
$repoRoot = "C:\mamba"
$smokeConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_300e_visdrone_es_bs1_cpuassign_smoke.py"
$formalConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py"
$recommended = Join-Path $repoRoot "scripts\start_recommended_training_blackwell.ps1"
$localDirect = Join-Path $repoRoot "scripts\start_local_training_blackwell.ps1"
$fatPreset = Join-Path $repoRoot "scripts\launch_fat_experiment_preset.ps1"

if (-not (Test-Path $smokeConfig)) { throw "Missing smoke config: $smokeConfig" }
if (-not (Test-Path $formalConfig)) { throw "Missing formal config: $formalConfig" }
if (-not (Test-Path $recommended)) { throw "Missing: $recommended" }
if (-not (Test-Path $localDirect)) { throw "Missing: $localDirect" }
if (-not (Test-Path $fatPreset)) { throw "Missing: $fatPreset" }

if ($LocalOnly -and $FatOnly) {
    throw "Use at most one of -LocalOnly or -FatOnly."
}

if ($LocalOnly) {
    $doLocal = $true
    $doFat = $false
} elseif ($FatOnly) {
    $doLocal = $false
    $doFat = $true
} else {
    # Default or -All: run both tracks (local per -LocalMode, then Fat formal).
    $doLocal = $true
    $doFat = $true
}

if ($doLocal) {
    if ($LocalMode -eq "formal") {
        Write-Host "=== BMVC wave-1: LOCAL formal (HybridMambaDet-B stable 1x ES, 1024x640) ===" -ForegroundColor Cyan
        Write-Warning "Local formal uses start_local_training_blackwell.ps1 (no host-stability gate). Prefer running check_local_host_stability.ps1 first."
        if ($DryRun) {
            Write-Host ("Would launch: {0} -ConfigPath {1}" -f $localDirect, $formalConfig)
        } else {
            & $localDirect -ConfigPath $formalConfig
        }
    } else {
        Write-Host "=== BMVC wave-1: LOCAL smoke (3 epochs, VisDrone V2) ===" -ForegroundColor Cyan
        if ($DryRun) {
            & $recommended -ConfigPath $smokeConfig -Mode smoke -AllowLocalSmokeWhenBlocked:$ForceLocalSmokeWhenBlocked -DryRun
        } else {
            & $recommended -ConfigPath $smokeConfig -Mode smoke -AllowLocalSmokeWhenBlocked:$ForceLocalSmokeWhenBlocked
        }
    }
}

if ($doFat) {
    Write-Host "=== BMVC wave-1: FAT formal HybridMambaDet-B stable 1x ES (1024x640) ===" -ForegroundColor Cyan
    if ($DryRun) {
        & $fatPreset -Preset hybridmambadet_stable_1x_visdrone_es -SyncFirst:$SyncFirst -DryRun
    } else {
        & $fatPreset -Preset hybridmambadet_stable_1x_visdrone_es -SyncFirst:$SyncFirst
    }
}

Write-Host 'Done. After Fat run: sync train.log and eval_metrics.json back, then plot with scripts/plot_detection_run_progress.py --run-dir <run>.' -ForegroundColor DarkGray
