param(
    [ValidateSet(
        "tinyvim_formal_visdrone",
        "tinyvim_stable_1x_visdrone_es",
        "tinyvim_first8_visdrone_recovery",
        "tinyvim_first8_visdrone_recovery_stable",
        "hybridmambadet_formal_300e_visdrone_es_bs1",
        "hybridmambadet_stable_1x_visdrone_es",
        "hybridmamba_base_stable_1x_visdrone_es",
        "r50_sanity_visdrone",
        "hybridmambadet_1x_visdrone",
        "hybridmambadet_1x_visdrone_es",
        "hybridmambadet_1x_aitodv2",
        "hybridmambadet_1x_dotahbb")]
    [string]$Preset = "tinyvim_formal_visdrone",
    [switch]$SyncFirst,
    [switch]$Background,
    [switch]$AdaptiveGuard,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$recommendedLauncher = Join-Path $repoRoot "scripts\start_recommended_training_blackwell.ps1"

if (-not (Test-Path $recommendedLauncher)) {
    throw "Missing launcher script: $recommendedLauncher"
}

switch ($Preset) {
    "tinyvim_formal_visdrone" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign.py"
        $runTag = "fat_tinyvim300e"
    }
    "tinyvim_stable_1x_visdrone_es" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_1x_visdrone_es_stable.py"
        $runTag = "fat_tinyvim1x_stable"
    }
    "tinyvim_first8_visdrone_recovery" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign_first8.py"
        $runTag = "fat_tinyvim300e_first8"
    }
    "tinyvim_first8_visdrone_recovery_stable" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign_first8_stable.py"
        $runTag = "fat_tinyvim300e_first8_stable"
    }
    "hybridmambadet_formal_300e_visdrone_es_bs1" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_300e_visdrone_es_bs1_cpuassign.py"
        $runTag = "fat_hybridmambadet300e"
    }
    "hybridmambadet_stable_1x_visdrone_es" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py"
        $runTag = "fat_hybridmambadet1x_stable"
    }
    "hybridmamba_base_stable_1x_visdrone_es" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_1x_visdrone_es_stable.py"
        $runTag = "fat_hybridmamba_base1x_stable"
    }
    "r50_sanity_visdrone" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_r50_fpn_2x_visdrone_sanity.py"
        $runTag = "fat_r50sanity2x"
    }
    "hybridmambadet_1x_visdrone" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone.py"
        $runTag = "fat_hybridmambadet1x"
    }
    "hybridmambadet_1x_visdrone_es" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es.py"
        $runTag = "fat_hybridmambadet1x_es"
    }
    "hybridmambadet_1x_aitodv2" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_aitodv2.py"
        $runTag = "fat_hybridmambadet1x_aitodv2"
    }
    "hybridmambadet_1x_dotahbb" {
        $configPath = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_dotahbb.py"
        $runTag = "fat_hybridmambadet1x_dotahbb"
    }
    default {
        throw "Unsupported preset: $Preset"
    }
}

if (-not (Test-Path $configPath)) {
    throw "Config not found: $configPath"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runId = "{0}_{1}" -f $runTag, $stamp

Write-Host ("Preset            {0}" -f $Preset)
Write-Host ("Config            {0}" -f $configPath)
Write-Host ("Planned run id    {0}" -f $runId)
Write-Host ("Target            FatMachine (formal)")
Write-Host ("Sync first        {0}" -f $SyncFirst.IsPresent)
Write-Host ("Background        {0}" -f $Background.IsPresent)
Write-Host ("Adaptive guard    {0}" -f $AdaptiveGuard.IsPresent)
Write-Host ("Dry run           {0}" -f $DryRun.IsPresent)

if ($DryRun) {
    & $recommendedLauncher -ConfigPath $configPath -Mode formal -RunId $runId -SyncFirst:$SyncFirst -Background:$Background -AdaptiveGuard:$AdaptiveGuard -DryRun
    return
}

& $recommendedLauncher -ConfigPath $configPath -Mode formal -RunId $runId -SyncFirst:$SyncFirst -Background:$Background -AdaptiveGuard:$AdaptiveGuard
