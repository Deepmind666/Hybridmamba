param(
    [string]$LocalRunId = "local_aitodv2_tinyvim_stable_retry_mem14_20260501_002610",
    [string]$FatRunId = "fat_aitodv2_hybridmambadet_stable_mem92_20260501_002325",
    [int]$Hours = 8,
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Continue"

$repoRoot = "C:\mamba"
$runsRoot = Join-Path $repoRoot "artifacts\runs"
$queueRoot = Join-Path $repoRoot "artifacts\queues"
$watchDir = Join-Path $queueRoot ("aitod_dual_watch_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force $watchDir | Out-Null
$logPath = Join-Path $watchDir "watch.log"
$deadline = (Get-Date).AddHours($Hours)

function Write-WatchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $logPath -Value $line
}

function Get-LocalTrainActive {
    param([string]$RunId)
    try {
        $out = & wsl.exe -d Ubuntu-24.04 -- bash -lc "pgrep -af run_train_mmdet3_manual.py || true" 2>$null
        return @($out | Where-Object { $_ -match [regex]::Escape($RunId) }).Count -gt 0
    } catch {
        Write-WatchLog ("local process query failed: {0}" -f $_.Exception.Message)
        return $false
    }
}

function Get-LocalProgress {
    param([string]$RunId)
    $trainLog = Join-Path (Join-Path $runsRoot $RunId) "train.log"
    if (-not (Test-Path $trainLog)) {
        return "no train.log"
    }
    $line = Select-String -Path $trainLog -Pattern "Epoch\(train\)" | Select-Object -Last 1
    if ($line) {
        return $line.Line.Trim()
    }
    $guard = Select-String -Path $trainLog -Pattern "Adaptive guard status" | Select-Object -Last 1
    if ($guard) {
        return $guard.Line.Trim()
    }
    return "train.log exists, no train iter yet"
}

function Test-LocalHasValidation {
    param([string]$RunId)
    $trainLog = Join-Path (Join-Path $runsRoot $RunId) "train.log"
    if (-not (Test-Path $trainLog)) {
        return $false
    }
    return [bool](Select-String -Path $trainLog -Pattern "Epoch\(val\)" -Quiet)
}

function Get-LocalFailure {
    param([string]$RunId)
    $runDir = Join-Path $runsRoot $RunId
    $logs = @("launcher.log", "train.log") | ForEach-Object { Join-Path $runDir $_ } | Where-Object { Test-Path $_ }
    foreach ($path in $logs) {
        $hit = Select-String -Path $path -Pattern "Traceback|RuntimeError|CUDA error|out of memory|Segmentation|Killed" -CaseSensitive:$false | Select-Object -Last 1
        if ($hit) {
            return $hit.Line.Trim()
        }
    }
    return ""
}

function Start-LocalRetry {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $newRunId = "local_aitodv2_tinyvim_stable_retry_mem10_$stamp"
    $config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_120e_aitodv2_stable.py"
    $launcher = Join-Path $repoRoot "scripts\start_local_training_blackwell.ps1"
    Write-WatchLog ("launching local retry {0}; expected first validation: 2-4h after launch if it keeps current local speed" -f $newRunId)
    $out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher `
        -ConfigPath $config `
        -RunId $newRunId `
        -ResumeFrom (Join-Path (Join-Path $runsRoot "local_aitodv2_tinyvim_stable_retry_mem12_20260501_013258") "last.pth") `
        -GpuMemGb 10 -TorchNumThreads 1 -InteropThreads 1 `
        -AdaptiveGuard `
        -GuardGpuUtilPct 70 -GuardCpuUtilPct 60 -GuardResumeUtilPct 50 -GuardTempC 68 -GuardMemoryPct 68 `
        -GuardCheckIntervalSec 2 -GuardCooldownSec 120 `
        -Niceness 18 -CpuCoreList "0-1" -GpuPowerLimitW 220 2>&1
    foreach ($line in $out) {
        Write-WatchLog ("launch: {0}" -f $line)
    }
    return $newRunId
}

function Get-FatProgress {
    param([string]$RunId)
    $remoteLog = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/runs/$RunId/train.log"
    $job = $null
    try {
        $job = Start-Job -ArgumentList $remoteLog -ScriptBlock {
            param([string]$Path)
            & ssh.exe -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=1 FatMachine "wsl.exe -d Ubuntu-24.04 -- tail -n 80 $Path" 2>$null
        }
        if (-not (Wait-Job -Job $job -Timeout 25)) {
            Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
            return "Fat progress query timed out"
        }
        $out = Receive-Job -Job $job -ErrorAction SilentlyContinue
        $line = @($out | Where-Object { $_ -match "Epoch\(train\)" }) | Select-Object -Last 1
        if ($line) {
            return $line.Trim()
        }
        $guard = @($out | Where-Object { $_ -match "Adaptive guard status" }) | Select-Object -Last 1
        if ($guard) {
            return $guard.Trim()
        }
        return "no recent Fat train line"
    } catch {
        return ("Fat progress query failed: {0}" -f $_.Exception.Message)
    } finally {
        if ($job) {
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null
        }
    }
}

Write-WatchLog ("watcher started; local={0}; fat={1}; deadline={2}" -f $LocalRunId, $FatRunId, $deadline.ToString("yyyy-MM-dd HH:mm:ss"))

while ((Get-Date) -lt $deadline) {
    $localActive = Get-LocalTrainActive -RunId $LocalRunId
    $localProgress = Get-LocalProgress -RunId $LocalRunId
    $localFailure = Get-LocalFailure -RunId $LocalRunId
    $localHasVal = Test-LocalHasValidation -RunId $LocalRunId
    $fatProgress = Get-FatProgress -RunId $FatRunId

    Write-WatchLog ("local active={0}; val={1}; progress={2}" -f $localActive, $localHasVal, $localProgress)
    if ($localFailure) {
        Write-WatchLog ("local failure marker: {0}" -f $localFailure)
    }
    Write-WatchLog ("Fat progress={0}" -f $fatProgress)

    if ((-not $localActive) -and (-not $localHasVal)) {
        if ($localFailure) {
            $LocalRunId = Start-LocalRetry
        } else {
            Write-WatchLog "local inactive before first validation but no failure marker found; leaving unchanged for manual inspection"
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-WatchLog "watcher reached deadline and exited"
