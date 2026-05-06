param(
    [int]$Hours = 12,
    [string]$CurrentRunId = "local_tinyvim1x_stable_resume_e14_mem18_20260429_000956",
    [string]$QueueName = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$runsRoot = Join-Path $repoRoot "artifacts\runs"
$queueRoot = Join-Path $repoRoot "artifacts\queues"
New-Item -ItemType Directory -Force $queueRoot | Out-Null

if (-not $QueueName) {
    $QueueName = "mamba12h_local_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$queueDir = Join-Path $queueRoot $QueueName
New-Item -ItemType Directory -Force $queueDir | Out-Null
$logPath = Join-Path $queueDir "queue.log"
$deadline = (Get-Date).AddHours($Hours)

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $logPath -Value $line
}

function Get-LocalTrainProcesses {
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $out = & wsl -d Ubuntu-24.04 -- bash -lc "ps -eo pid,ppid,etime,args | grep run_train_mmdet3_manual.py | grep -v grep || true" 2>$null
        return @($out | Where-Object { $_ -match "run_train_mmdet3_manual.py" })
    } catch {
        Write-QueueLog ("Failed to query WSL processes: {0}" -f $_.Exception.Message)
        return @()
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Test-LocalTrainingActive {
    return ((Get-LocalTrainProcesses).Count -gt 0)
}

function Wait-ForLocalIdle {
    param([string]$Reason)
    Write-QueueLog ("Waiting for local training to finish: {0}" -f $Reason)
    while ((Get-Date) -lt $deadline) {
        $procs = Get-LocalTrainProcesses
        if ($procs.Count -eq 0) {
            Write-QueueLog "Local training is idle."
            return $true
        }
        Write-QueueLog ("Local training still active ({0} process lines)." -f $procs.Count)
        Start-Sleep -Seconds 180
    }
    Write-QueueLog "Deadline reached while waiting for local training."
    return $false
}

function Get-RunFailure {
    param([string]$RunId)
    $runDir = Join-Path $runsRoot $RunId
    $logs = @(
        (Join-Path $runDir "launcher.log"),
        (Join-Path $runDir "train.log")
    ) | Where-Object { Test-Path $_ }
    foreach ($log in $logs) {
        $hit = Select-String -Path $log -Pattern "Traceback|RuntimeError|CUDA error|out of memory|Segmentation|Killed" -CaseSensitive:$false | Select-Object -Last 1
        if ($hit) {
            return $hit.Line
        }
    }
    return ""
}

function Write-RunSummary {
    param([string]$RunId)
    $runDir = Join-Path $runsRoot $RunId
    $evalPath = Join-Path $runDir "eval_metrics.json"
    if (Test-Path $evalPath) {
        try {
            $eval = Get-Content $evalPath -Raw | ConvertFrom-Json
            Write-QueueLog ("Run {0}: eval epoch={1}, latest AP={2}, best AP={3}" -f $RunId, $eval.epoch, $eval.latest.'coco/bbox_mAP', $eval.best_value)
        } catch {
            Write-QueueLog ("Run {0}: eval_metrics.json exists but parse failed." -f $RunId)
        }
    } else {
        Write-QueueLog ("Run {0}: no eval_metrics.json yet." -f $RunId)
    }
    $failure = Get-RunFailure -RunId $RunId
    if ($failure) {
        Write-QueueLog ("Run {0}: failure marker: {1}" -f $RunId, $failure)
    }
}

function Start-LocalExperiment {
    param(
        [string]$Tag,
        [string]$ConfigPath,
        [int]$GpuMemGb,
        [string]$ResumeFrom = "",
        [int]$Attempt = 1
    )
    if ((Get-Date) -ge $deadline) {
        Write-QueueLog ("Skip {0}: deadline already reached." -f $Tag)
        return $null
    }

    $runId = "{0}_{1}_a{2}" -f $Tag, (Get-Date -Format "yyyyMMdd_HHmmss"), $Attempt
    $launcher = Join-Path $repoRoot "scripts\start_local_training_blackwell_adaptive.ps1"
    $args = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-ConfigPath", $ConfigPath,
        "-RunId", $runId,
        "-GpuMemGb", $GpuMemGb,
        "-GuardGpuUtilPct", "75",
        "-GuardCpuUtilPct", "75",
        "-GuardResumeUtilPct", "60",
        "-GuardTempC", "74",
        "-GuardMemoryPct", "82",
        "-GuardCooldownSec", "45",
        "-AllowBlockedHost"
    )
    if ($ResumeFrom) {
        $args += @("-ResumeFrom", $ResumeFrom)
    }

    Write-QueueLog ("Launching local run {0}" -f $runId)
    $out = & powershell @args 2>&1
    foreach ($line in $out) {
        Write-QueueLog ("launch: {0}" -f $line)
    }
    Start-Sleep -Seconds 120
    [void](Wait-ForLocalIdle -Reason $runId)
    Write-RunSummary -RunId $runId

    $failure = Get-RunFailure -RunId $runId
    $lastPath = Join-Path (Join-Path $runsRoot $runId) "last.pth"
    if ($failure -and $Attempt -lt 2 -and (Test-Path $lastPath) -and (Get-Date) -lt $deadline) {
        Write-QueueLog ("Retrying {0} from last checkpoint with lower memory cap." -f $runId)
        return Start-LocalExperiment -Tag ("{0}_retry" -f $Tag) -ConfigPath $ConfigPath -GpuMemGb ([Math]::Max(14, $GpuMemGb - 2)) -ResumeFrom $lastPath -Attempt ($Attempt + 1)
    }
    return $runId
}

Write-QueueLog ("Local 12h queue started. Deadline={0}" -f $deadline.ToString("yyyy-MM-dd HH:mm:ss"))

if ($CurrentRunId) {
    [void](Wait-ForLocalIdle -Reason ("current run {0}" -f $CurrentRunId))
    Write-RunSummary -RunId $CurrentRunId
    $failure = Get-RunFailure -RunId $CurrentRunId
    $lastPath = Join-Path (Join-Path $runsRoot $CurrentRunId) "last.pth"
    if ($failure -and (Test-Path $lastPath) -and (Get-Date) -lt $deadline) {
        $tinyConfig = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_1x_visdrone_es_stable.py"
        [void](Start-LocalExperiment -Tag "local_tinyvim1x_stable_resume_retry" -ConfigPath $tinyConfig -GpuMemGb 16 -ResumeFrom $lastPath)
    }
}

$jobs = @(
    @{
        Tag = "local_hybridmambadet_stage01_stable"
        Config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stage01_stable.py"
        GpuMemGb = 20
    },
    @{
        Tag = "local_hybridmambadet_fusion05_stable"
        Config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion05_stable.py"
        GpuMemGb = 20
    },
    @{
        Tag = "local_hybridmamba_base1x_stable"
        Config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_1x_visdrone_es_stable.py"
        GpuMemGb = 20
    }
)

foreach ($job in $jobs) {
    if ((Get-Date) -ge $deadline) {
        Write-QueueLog "Deadline reached before launching remaining local jobs."
        break
    }
    [void](Start-LocalExperiment -Tag $job.Tag -ConfigPath $job.Config -GpuMemGb $job.GpuMemGb)
}

Write-QueueLog "Local 12h queue finished or reached deadline."
