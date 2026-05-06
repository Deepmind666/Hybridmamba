param(
    [string]$LocalRunId = "",
    [string]$FatRunId = "",
    [int]$RefreshSeconds = 5,
    [switch]$IncludeFat,
    [switch]$FatOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$localRunsRoot = Join-Path $repoRoot "artifacts\runs"

function Resolve-LatestRunId {
    param(
        [string]$RunsRoot,
        [string]$NameHint = ""
    )

    if (-not (Test-Path $RunsRoot)) {
        return $null
    }

    $dirs = Get-ChildItem -Path $RunsRoot -Directory | Sort-Object LastWriteTime -Descending
    if ($NameHint) {
        $matched = $dirs | Where-Object { $_.Name -like "*$NameHint*" }
        if ($matched) {
            return $matched[0].Name
        }
    }
    if ($dirs) {
        return $dirs[0].Name
    }
    return $null
}

function Get-LocalGpuLine {
    try {
        $raw = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>$null
        if (-not $raw) { return "GPU: no nvidia-smi output" }
        $parts = $raw.Split(",") | ForEach-Object { $_.Trim() }
        return ("GPU: mem {0}/{1} MiB | util {2}% | temp {3}C | power {4}/{5} W" -f $parts[0], $parts[1], $parts[2], $parts[3], $parts[4], $parts[5])
    } catch {
        return "GPU: nvidia-smi failed"
    }
}

function Get-RecentSignalLines {
    param(
        [string]$LogPath,
        [int]$TailCount = 300,
        [int]$KeepCount = 12
    )

    if (-not (Test-Path $LogPath)) {
        return @("Log not found: $LogPath")
    }

    try {
        $lines = Get-Content -Path $LogPath -Tail $TailCount -ErrorAction Stop
        $signals = $lines | Where-Object {
            $_ -match "Epoch\(train\)|Epoch\(val\)|loss:|eta:|Early stopping|RuntimeError|Traceback|CUDA out of memory|Killed|error|ERROR"
        }
        if (-not $signals) {
            return @("No key lines in last $TailCount lines.")
        }
        return @($signals | Select-Object -Last $KeepCount)
    } catch {
        return @("Failed to read log: $($_.Exception.Message)")
    }
}

if (-not $FatOnly -and -not $LocalRunId) {
    $LocalRunId = Resolve-LatestRunId -RunsRoot $localRunsRoot -NameHint "local_fatstyle"
    if (-not $LocalRunId) {
        $LocalRunId = Resolve-LatestRunId -RunsRoot $localRunsRoot
    }
}

$localRunDir = $null
$localLog = $null
$localEval = $null

if (-not $FatOnly -and $LocalRunId) {
    $localRunDir = Join-Path $localRunsRoot $LocalRunId
    $localTrainLog = Join-Path $localRunDir "train.log"
    $localLauncherLog = Join-Path $localRunDir "launcher.log"
    if (Test-Path $localTrainLog) {
        $localLog = $localTrainLog
    } else {
        $localLog = $localLauncherLog
    }
    $localEval = Join-Path $localRunDir "eval_metrics.json"
    if (-not (Test-Path $localRunDir)) {
        throw "Local run directory not found: $localRunDir"
    }
}

if ($IncludeFat -and -not $FatRunId) {
    try {
        $fatLatest = ssh FatMachine "cmd /c dir /ad /b /o-d C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs" 2>$null | Select-Object -First 1
        if ($fatLatest) {
            $FatRunId = ($fatLatest | Select-Object -First 1).ToString().Trim()
        } else {
            Write-Host "IncludeFat enabled but no Fat run found. Showing local only." -ForegroundColor Yellow
            $IncludeFat = $false
        }
    } catch {
        Write-Host "IncludeFat enabled but SSH failed. Showing local only." -ForegroundColor Yellow
        $IncludeFat = $false
    }
}

if ($FatRunId) {
    $IncludeFat = $true
}

Write-Host "Watcher started. Press Ctrl+C to exit." -ForegroundColor Cyan
Start-Sleep -Seconds 1

while ($true) {
    Clear-Host
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host "=== Local + Fat watcher ==="
    Write-Host "Time: $now"
    Write-Host ""

    if (-not $FatOnly) {
        Write-Host "[Local] RunId: $LocalRunId" -ForegroundColor Green
        Write-Host ("Dir: {0}" -f $localRunDir)
        Write-Host (Get-LocalGpuLine)

        if (Test-Path $localLog) {
            $logItem = Get-Item $localLog
            Write-Host ("Log: {0} | {1} KB | LastWrite {2}" -f $localLog, [math]::Round($logItem.Length / 1KB, 1), $logItem.LastWriteTime.ToString("HH:mm:ss"))
        } else {
            Write-Host "Log: train.log/launcher.log not created"
        }

        if (Test-Path $localEval) {
            try {
                $eval = Get-Content $localEval -Raw | ConvertFrom-Json
                Write-Host ("Eval: epoch={0} iter={1} latest_mAP={2} best_mAP={3}" -f $eval.epoch, $eval.iter, $eval.latest.'coco/bbox_mAP', $eval.best_value)
            } catch {
                Write-Host "Eval: parse failed"
            }
        } else {
            Write-Host "Eval: no eval_metrics.json yet"
        }

        Write-Host ""
        Write-Host "[Local] Key log lines:" -ForegroundColor Green
        $localSignals = Get-RecentSignalLines -LogPath $localLog
        foreach ($line in $localSignals) {
            Write-Host ("  " + $line)
        }
    } else {
        Write-Host "[Local] skipped (FatOnly mode)" -ForegroundColor DarkGray
    }

    if ($IncludeFat) {
        Write-Host ""
        Write-Host ("[Fat] RunId: {0}" -f $FatRunId) -ForegroundColor Yellow
        $fatTrain = "C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\$FatRunId\train.log"
        $fatEval = "C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\$FatRunId\eval_metrics.json"

        try {
            $fatEvalCmd = "if exist `"$fatEval`" (type `"$fatEval`") else (echo NO_EVAL)"
            $fatEvalOut = ssh FatMachine "cmd /c $fatEvalCmd" 2>$null
            if ($fatEvalOut -and ($fatEvalOut -notmatch "NO_EVAL")) {
                try {
                    $fatObj = ($fatEvalOut -join "`n") | ConvertFrom-Json
                    Write-Host ("Fat Eval: epoch={0} iter={1} latest_mAP={2} best_mAP={3}" -f $fatObj.epoch, $fatObj.iter, $fatObj.latest.'coco/bbox_mAP', $fatObj.best_value)
                } catch {
                    Write-Host "Fat Eval: exists but parse failed"
                }
            } else {
                Write-Host "Fat Eval: no eval_metrics.json yet"
            }
        } catch {
            Write-Host "Fat Eval: ssh fetch failed"
        }

        try {
            $fatTailCmd = "powershell -NoProfile -Command `"if (Test-Path '$fatTrain') { Get-Content '$fatTrain' -Tail 8 } else { 'NO_TRAIN_LOG' }`""
            $fatTail = ssh FatMachine $fatTailCmd 2>$null
            Write-Host "[Fat] Log tail:"
            if ($fatTail) {
                foreach ($line in $fatTail) {
                    Write-Host ("  " + $line)
                }
            } else {
                Write-Host "  no output (maybe ssh failed)"
            }
        } catch {
            Write-Host "[Fat] log fetch failed"
        }
    }

    Start-Sleep -Seconds $RefreshSeconds
}
