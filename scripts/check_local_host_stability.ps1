[CmdletBinding()]
param(
    [int]$FormalUnexpectedShutdownLookbackDays = 7,
    [int]$FormalWheaLookbackHours = 24,
    [int]$SmokeUnexpectedShutdownLookbackHours = 24,
    [int]$SmokeWheaLookbackMinutes = 60,
    [switch]$EmitJson
)

$ErrorActionPreference = "Stop"

function Get-SafeWinEvents {
    param([hashtable]$Filter)

    try {
        return @(Get-WinEvent -FilterHashtable $Filter -ErrorAction Stop)
    } catch {
        return @()
    }
}

function Get-SafeNvidiaGpuInfo {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }

    $fields = @(
        "name",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total"
    )

    $raw = & nvidia-smi "--query-gpu=$($fields -join ',')" "--format=csv,noheader,nounits" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        return $null
    }

    $csv = ($fields -join ",") + "`n" + (($raw | ForEach-Object { $_.Trim() }) -join "`n")
    return ($csv | ConvertFrom-Csv | Select-Object -First 1)
}

function Get-TrainingLikeProcesses {
    $patterns = @(
        "run_train_mmdet3",
        "train.py",
        "smoke_detection",
        "mmengine",
        "mmdet",
        "hybrid-mamba",
        "\\mamba\\"
    )

    return @(
        Get-CimInstance Win32_Process | Where-Object {
            $commandLine = $_.CommandLine
            $_.CommandLine -and (
                $_.Name -match "python|wsl|bash|sh" -or
                $_.CommandLine -match "python|wsl|bash|sh"
            ) -and (
                @($patterns | Where-Object { $commandLine -match $_ }).Count -gt 0
            )
        } | Select-Object Name, ProcessId, CreationDate, CommandLine
    )
}

function Add-Reason {
    param(
        [System.Collections.Generic.List[string]]$Target,
        [string]$Message
    )

    if (-not [string]::IsNullOrWhiteSpace($Message)) {
        $Target.Add($Message)
    }
}

$now = Get-Date
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$board = Get-CimInstance Win32_BaseBoard | Select-Object -First 1
$bios = Get-CimInstance Win32_BIOS | Select-Object -First 1
$gpu = Get-SafeNvidiaGpuInfo
$trainingProcesses = Get-TrainingLikeProcesses

$formalShutdowns = Get-SafeWinEvents @{
    LogName = "System"
    StartTime = $now.AddDays(-$FormalUnexpectedShutdownLookbackDays)
    Id = 41, 6008
}
$smokeShutdowns = Get-SafeWinEvents @{
    LogName = "System"
    StartTime = $now.AddHours(-$SmokeUnexpectedShutdownLookbackHours)
    Id = 41, 6008
}
$formalWhea = Get-SafeWinEvents @{
    LogName = "System"
    ProviderName = "Microsoft-Windows-WHEA-Logger"
    StartTime = $now.AddHours(-$FormalWheaLookbackHours)
}
$smokeWhea = Get-SafeWinEvents @{
    LogName = "System"
    ProviderName = "Microsoft-Windows-WHEA-Logger"
    StartTime = $now.AddMinutes(-$SmokeWheaLookbackMinutes)
}

$formalReasons = [System.Collections.Generic.List[string]]::new()
$smokeReasons = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()

if ($formalShutdowns.Count -gt 0) {
    Add-Reason $formalReasons "unexpected shutdown evidence (Kernel-Power/EventLog 6008) exists within the last $FormalUnexpectedShutdownLookbackDays days"
}
if ($smokeShutdowns.Count -gt 0) {
    Add-Reason $smokeReasons "unexpected shutdown evidence exists within the last $SmokeUnexpectedShutdownLookbackHours hours"
}
if ($formalWhea.Count -gt 0) {
    Add-Reason $formalReasons "WHEA hardware errors exist within the last $FormalWheaLookbackHours hours"
}
if ($smokeWhea.Count -gt 0) {
    Add-Reason $smokeReasons "WHEA hardware errors exist within the last $SmokeWheaLookbackMinutes minutes"
}
if ($trainingProcesses.Count -gt 0) {
    Add-Reason $formalReasons "another training-like python/wsl process is already active"
    Add-Reason $smokeReasons "another training-like python/wsl process is already active"
}

if ($gpu) {
    $gpuUtil = [double]$gpu."utilization.gpu"
    $gpuMemUsed = [double]$gpu."memory.used"
    $gpuTemp = [double]$gpu."temperature.gpu"
    $gpuPowerDraw = [double]$gpu."power.draw"
    $gpuPowerLimit = [double]$gpu."power.limit"

    if ($gpuMemUsed -gt 4096) {
        Add-Reason $formalReasons "GPU memory is not near-idle ($gpuMemUsed MiB in use)"
        Add-Reason $smokeReasons "GPU memory is not near-idle ($gpuMemUsed MiB in use)"
    }
    if ($gpuUtil -gt 10) {
        Add-Reason $formalReasons "GPU utilization is already elevated ($gpuUtil%)"
        Add-Reason $smokeReasons "GPU utilization is already elevated ($gpuUtil%)"
    }
    if ($gpuTemp -ge 65) {
        Add-Reason $warnings "GPU temperature is already $gpuTemp C before launch"
    }
    if ($gpuPowerLimit -gt 0 -and $gpuPowerDraw / $gpuPowerLimit -ge 0.7) {
        Add-Reason $warnings "GPU board power is already high before launch ($gpuPowerDraw / $gpuPowerLimit W)"
    }
}

$status = if ($formalReasons.Count -eq 0) {
    "eligible"
} elseif ($smokeReasons.Count -eq 0) {
    "smoke_only"
} else {
    "blocked"
}

$recommendedTarget = switch ($status) {
    "eligible" { "local_formal_allowed_but_remote_overnight_runs_still_preferred" }
    "smoke_only" { "local_smoke_only_use_fatmachine_for_formal_runs" }
    default { "fatmachine_only_until_host_health_is_clean" }
}

$nextSteps = switch ($status) {
    "eligible" {
        @(
            "Local smoke and local formal are allowed by the guard."
            "Keep one heavy job per machine and launch through start_local_training_blackwell_guarded.ps1."
            "Use FatMachine for overnight runs if you want the lowest interruption risk."
        )
    }
    "smoke_only" {
        @(
            "Only short smoke or preflight runs are allowed locally."
            "Use start_local_training_blackwell_guarded.ps1 -Mode smoke for any local validation."
            "Send every formal run to FatMachine until the host has a clean formal window."
        )
    }
    default {
        @(
            "Do not launch any local training on this host."
            "Fix BIOS, memory, voltage, or thermal stability first, then wait for a clean observation window."
            "Run formal work on FatMachine while the local host remains blocked."
        )
    }
}

$report = [pscustomobject]@{
    checkedAt = $now.ToString("yyyy-MM-dd HH:mm:ss")
    host = [pscustomobject]@{
        computerName = $env:COMPUTERNAME
        lastBoot = ($os.LastBootUpTime.ToLocalTime()).ToString("yyyy-MM-dd HH:mm:ss")
        cpu = $cpu.Name
        logicalProcessors = $cpu.NumberOfLogicalProcessors
        motherboard = "$($board.Manufacturer) $($board.Product)".Trim()
        biosVersion = $bios.SMBIOSBIOSVersion
    }
    gpu = if ($gpu) {
        [pscustomobject]@{
            name = $gpu.name
            driverVersion = $gpu.driver_version
            pstate = $gpu.pstate
            temperatureC = [double]$gpu."temperature.gpu"
            powerDrawW = [double]$gpu."power.draw"
            powerLimitW = [double]$gpu."power.limit"
            gpuUtilPercent = [double]$gpu."utilization.gpu"
            memUtilPercent = [double]$gpu."utilization.memory"
            memoryUsedMiB = [double]$gpu."memory.used"
            memoryTotalMiB = [double]$gpu."memory.total"
        }
    } else {
        $null
    }
    status = $status
    recommendedTarget = $recommendedTarget
    windowsSignals = [pscustomobject]@{
        formalUnexpectedShutdownCount = $formalShutdowns.Count
        smokeUnexpectedShutdownCount = $smokeShutdowns.Count
        formalWheaCount = $formalWhea.Count
        smokeWheaCount = $smokeWhea.Count
        lastUnexpectedShutdown = ($formalShutdowns | Sort-Object TimeCreated -Descending | Select-Object -First 1 -ExpandProperty TimeCreated -ErrorAction SilentlyContinue)
        lastWhea = ($formalWhea | Sort-Object TimeCreated -Descending | Select-Object -First 1 -ExpandProperty TimeCreated -ErrorAction SilentlyContinue)
    }
    activeTrainingProcesses = @($trainingProcesses)
    formalBlockers = @($formalReasons)
    smokeBlockers = @($smokeReasons)
    warnings = @($warnings)
    nextSteps = @($nextSteps)
}

if ($EmitJson) {
    $report | ConvertTo-Json -Depth 6
    exit 0
}

Write-Host ("=" * 72)
Write-Host "HybridMamba Local Host Stability"
Write-Host ("=" * 72)
Write-Host ("Host              {0}" -f $report.host.computerName)
Write-Host ("CPU               {0}" -f $report.host.cpu)
Write-Host ("Motherboard       {0}" -f $report.host.motherboard)
Write-Host ("BIOS              {0}" -f $report.host.biosVersion)
Write-Host ("Last boot         {0}" -f $report.host.lastBoot)
if ($report.gpu) {
    Write-Host ("GPU               {0}" -f $report.gpu.name)
    Write-Host ("GPU driver        {0}" -f $report.gpu.driverVersion)
    Write-Host ("GPU state         {0}, {1} C, {2} MiB / {3} MiB, util {4}%" -f $report.gpu.pstate, $report.gpu.temperatureC, $report.gpu.memoryUsedMiB, $report.gpu.memoryTotalMiB, $report.gpu.gpuUtilPercent)
}
Write-Host ("Status            {0}" -f $report.status)
Write-Host ("Target            {0}" -f $report.recommendedTarget)
Write-Host
Write-Host ("Formal signals    unexpected shutdowns={0}, WHEA={1}" -f $report.windowsSignals.formalUnexpectedShutdownCount, $report.windowsSignals.formalWheaCount)
Write-Host ("Smoke signals     unexpected shutdowns={0}, WHEA={1}" -f $report.windowsSignals.smokeUnexpectedShutdownCount, $report.windowsSignals.smokeWheaCount)
if ($report.windowsSignals.lastUnexpectedShutdown) {
    Write-Host ("Last shutdown     {0}" -f ([datetime]$report.windowsSignals.lastUnexpectedShutdown).ToString("yyyy-MM-dd HH:mm:ss"))
}
if ($report.windowsSignals.lastWhea) {
    Write-Host ("Last WHEA         {0}" -f ([datetime]$report.windowsSignals.lastWhea).ToString("yyyy-MM-dd HH:mm:ss"))
}
Write-Host

if ($report.activeTrainingProcesses.Count -gt 0) {
    Write-Host "Active training-like processes:"
    $report.activeTrainingProcesses | Format-Table -AutoSize | Out-String | Write-Host
}

if ($report.formalBlockers.Count -gt 0) {
    Write-Host "Formal blockers:"
    $report.formalBlockers | ForEach-Object { Write-Host ("- {0}" -f $_) }
    Write-Host
}

if ($report.smokeBlockers.Count -gt 0) {
    Write-Host "Smoke blockers:"
    $report.smokeBlockers | ForEach-Object { Write-Host ("- {0}" -f $_) }
    Write-Host
}

if ($report.warnings.Count -gt 0) {
    Write-Host "Warnings:"
    $report.warnings | ForEach-Object { Write-Host ("- {0}" -f $_) }
    Write-Host
}

Write-Host "Next steps:"
$report.nextSteps | ForEach-Object { Write-Host ("- {0}" -f $_) }
Write-Host ("=" * 72)
