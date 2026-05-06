param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [ValidateSet("smoke", "formal")][string]$Mode = "formal",
    [string]$RunId,
    [int]$GpuMemGb = 24,
    [switch]$SyncFirst,
    [switch]$AllowLocalFormalWhenEligible,
    [switch]$AllowLocalSmokeWhenBlocked,
    [switch]$Background,
    [switch]$AdaptiveGuard,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$guardScript = Join-Path $repoRoot "scripts\check_local_host_stability.ps1"
$localLauncher = Join-Path $repoRoot "scripts\start_local_training_blackwell_guarded.ps1"
$remoteLauncher = Join-Path $repoRoot "scripts\start_fatmachine_run_blackwell.ps1"
$syncScript = Join-Path $repoRoot "scripts\sync_to_fatmachine_blackwell.ps1"

$resolvedConfig = Resolve-Path $ConfigPath
$guard = & $guardScript -EmitJson | ConvertFrom-Json

if (-not $RunId) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baseName = [IO.Path]::GetFileNameWithoutExtension($resolvedConfig.Path)
    $RunId = "{0}_{1}_{2}" -f $baseName, $Mode, $stamp
}

if ($Mode -eq "smoke") {
    if ($guard.status -eq "blocked" -and -not $AllowLocalSmokeWhenBlocked) {
        throw "Recommended launcher refused local smoke because host status is blocked."
    }

    if ($DryRun) {
        Write-Host "Selected target    local"
        Write-Host ("Mode              {0}" -f $Mode)
        Write-Host ("Guard status      {0}" -f $guard.status)
        Write-Host ("Allow smoke blocked {0}" -f $AllowLocalSmokeWhenBlocked.IsPresent)
        Write-Host ("Run id            {0}" -f $RunId)
        return
    }

    & $localLauncher -ConfigPath $resolvedConfig.Path -Mode smoke -RunId $RunId -GpuMemGb $GpuMemGb -AllowSmokeWhenBlocked:$AllowLocalSmokeWhenBlocked
    return
}

$useLocalFormal = $AllowLocalFormalWhenEligible -and $guard.status -eq "eligible"

if ($DryRun) {
    Write-Host ("Selected target    {0}" -f $(if ($useLocalFormal) { "local" } else { "FatMachine" }))
    Write-Host ("Mode              {0}" -f $Mode)
    Write-Host ("Guard status      {0}" -f $guard.status)
    Write-Host ("Run id            {0}" -f $RunId)
    if (-not $useLocalFormal) {
        Write-Host ("Sync first        {0}" -f $SyncFirst.IsPresent)
        Write-Host ("Background        {0}" -f $Background.IsPresent)
        Write-Host ("Adaptive guard    {0}" -f $AdaptiveGuard.IsPresent)
    }
    return
}

if ($useLocalFormal) {
    & $localLauncher -ConfigPath $resolvedConfig.Path -Mode formal -RunId $RunId -GpuMemGb $GpuMemGb
    return
}

if ($SyncFirst) {
    & $syncScript
}

& $remoteLauncher -ConfigPath $resolvedConfig.Path -RunId $RunId -Background:$Background -AdaptiveGuard:$AdaptiveGuard
Write-Host "Formal run was routed to FatMachine."
