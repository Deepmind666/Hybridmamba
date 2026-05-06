param(
    [int]$Hours = 24,
    [string]$LocalCurrentRunId = "local_hybridmambadet_stage01_resume_e27_mem16_20260430_1608",
    [string]$FatCurrentRunId = "fat_hybridmambadet_fusion10_stable_retry_20260429_2045",
    [string]$QueueName = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$queueRoot = Join-Path $repoRoot "artifacts\queues"
New-Item -ItemType Directory -Force $queueRoot | Out-Null

if (-not $QueueName) {
    $QueueName = "aitod_next_dual_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$queueDir = Join-Path $queueRoot $QueueName
New-Item -ItemType Directory -Force $queueDir | Out-Null
$deadline = (Get-Date).AddHours($Hours)

function Start-Controller {
    param(
        [ValidateSet("local", "fat")][string]$Side
    )

    $logPath = Join-Path $queueDir "$Side.log"
    $controller = Join-Path $queueDir "$Side-controller.ps1"

    $script = @"
`$ErrorActionPreference = "Stop"
`$repoRoot = "$repoRoot"
`$deadline = [datetime]"$($deadline.ToString("o"))"
`$logPath = "$logPath"
`$localCurrentRunId = "$LocalCurrentRunId"
`$fatCurrentRunId = "$FatCurrentRunId"

function Write-QueueLog {
    param([string]`$Message)
    `$line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), `$Message
    Add-Content -Path `$logPath -Value `$line
}

function Get-LocalTrainLines {
    try {
        `$procs = Get-CimInstance Win32_Process | Where-Object {
            (`$_.Name -eq "wsl.exe" -or `$_.Name -eq "cmd.exe") -and
            (`$_.CommandLine -match "run_train_mmdet3_manual.py" -or `$_.CommandLine -match [regex]::Escape(`$localCurrentRunId))
        }
        return @(`$procs | ForEach-Object { "{0} {1}" -f `$_.ProcessId, `$_.CommandLine })
    } catch {
        Write-QueueLog ("local process query failed: {0}" -f `$_.Exception.Message)
        return @("query_failed")
    }
}

function Get-FatTrainLines {
    try {
        `$remoteScript = @'
`$runId = "__FAT_CURRENT_RUN_ID__"
Get-CimInstance Win32_Process | Where-Object {
    (`$_.Name -eq "wsl.exe" -or `$_.Name -eq "cmd.exe") -and
    (`$_.CommandLine -match "run_train_mmdet3_manual.py" -or `$_.CommandLine -match [regex]::Escape(`$runId))
} | ForEach-Object { "{0} {1}" -f `$_.ProcessId, `$_.CommandLine }
'@
        `$remoteScript = `$remoteScript.Replace("__FAT_CURRENT_RUN_ID__", `$fatCurrentRunId)
        `$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(`$remoteScript))
        `$out = & ssh.exe FatMachine "powershell -NoProfile -EncodedCommand `$encoded" 2>`$null
        return @(`$out | Where-Object { `$_ -match "run_train_mmdet3_manual.py" -or `$_ -match [regex]::Escape(`$fatCurrentRunId) })
    } catch {
        Write-QueueLog ("fat process query failed: {0}" -f `$_.Exception.Message)
        return @("query_failed")
    }
}

function Wait-ForLocalIdle {
    Write-QueueLog ("waiting for local current run to finish: {0}" -f `$localCurrentRunId)
    while ((Get-Date) -lt `$deadline) {
        `$lines = Get-LocalTrainLines
        if (`$lines.Count -eq 0) {
            Write-QueueLog "local training is idle"
            return `$true
        }
        Write-QueueLog ("local still busy: {0}" -f (`$lines -join " | "))
        Start-Sleep -Seconds 300
    }
    Write-QueueLog "deadline reached while waiting for local idle"
    return `$false
}

function Wait-ForFatIdle {
    Write-QueueLog ("waiting for Fat current run to finish: {0}" -f `$fatCurrentRunId)
    while ((Get-Date) -lt `$deadline) {
        `$lines = Get-FatTrainLines
        if (`$lines.Count -eq 0) {
            Write-QueueLog "Fat training is idle"
            return `$true
        }
        Write-QueueLog ("Fat still busy: {0}" -f (`$lines -join " | "))
        Start-Sleep -Seconds 300
    }
    Write-QueueLog "deadline reached while waiting for Fat idle"
    return `$false
}

function Start-LocalAitodBaseline {
    `$config = Join-Path `$repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_120e_aitodv2_stable.py"
    `$runId = "local_aitodv2_tinyvim_stable_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    `$launcher = Join-Path `$repoRoot "scripts\start_local_training_blackwell_adaptive.ps1"
    Write-QueueLog ("launching local AI-TOD baseline {0}; estimated completion after launch: 6-10h if early-stop near epoch 20-30" -f `$runId)
    `$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `$launcher -ConfigPath `$config -RunId `$runId -GpuMemGb 16 -TorchNumThreads 1 -InteropThreads 1 -Niceness 15 -CpuCoreList "0-5" -GpuPowerLimitW 300 -GuardGpuUtilPct 80 -GuardCpuUtilPct 75 -GuardResumeUtilPct 65 -GuardTempC 74 -GuardMemoryPct 75 -GuardCooldownSec 90 -AllowBlockedHost 2>&1
    foreach (`$line in `$out) { Write-QueueLog ("launch: {0}" -f `$line) }
}

function Start-FatAitodFinal {
    `$config = Join-Path `$repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_120e_aitodv2_stable.py"
    `$runId = "fat_aitodv2_hybridmambadet_stable_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    `$sync = Join-Path `$repoRoot "scripts\sync_to_fatmachine_blackwell.ps1"
    `$launcher = Join-Path `$repoRoot "scripts\start_fatmachine_run_blackwell.ps1"
    Write-QueueLog "syncing AI-TOD/DOTA-capable repo bundle to FatMachine before launch"
    `$syncOut = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `$sync 2>&1
    foreach (`$line in `$syncOut) { Write-QueueLog ("sync: {0}" -f `$line) }
    Write-QueueLog ("launching Fat AI-TOD final {0}; estimated completion after launch: 10-16h if early-stop near epoch 20-30" -f `$runId)
    `$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `$launcher -ConfigPath `$config -RunId `$runId -GpuMemGb 16 -TorchNumThreads 2 -InteropThreads 1 -AdaptiveGuard -GuardGpuUtilPct 80 -GuardCpuUtilPct 75 -GuardResumeUtilPct 65 -GuardTempC 74 -GuardMemoryPct 75 -GuardCheckIntervalSec 2 -GuardCooldownSec 90 -Background 2>&1
    foreach (`$line in `$out) { Write-QueueLog ("launch: {0}" -f `$line) }
}

Write-QueueLog ("controller started; side=$Side deadline={0}" -f `$deadline.ToString("yyyy-MM-dd HH:mm:ss"))
if ("$Side" -eq "local") {
    if (Wait-ForLocalIdle) { Start-LocalAitodBaseline }
} else {
    if (Wait-ForFatIdle) { Start-FatAitodFinal }
}
Write-QueueLog "controller finished"
"@

    Set-Content -Path $controller -Value $script -Encoding ASCII
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $controller) -WindowStyle Hidden | Out-Null
    Write-Host ("Started {0} AI-TOD queue controller" -f $Side)
    Write-Host ("  log: {0}" -f $logPath)
}

Start-Controller -Side local
Start-Controller -Side fat

Write-Host ("Queue name: {0}" -f $QueueName)
Write-Host ("Queue dir: {0}" -f $queueDir)
Write-Host ("Deadline: {0}" -f $deadline.ToString("yyyy-MM-dd HH:mm:ss"))
Write-Host "Planned next runs:"
Write-Host "  Local: AI-TOD-v2 TinyViM baseline, estimated 6-10h after launch."
Write-Host "  Fat: AI-TOD-v2 HybridMambaDet final, estimated 10-16h after launch."
