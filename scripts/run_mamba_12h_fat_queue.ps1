param(
    [int]$Hours = 12,
    [string]$QueueName = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$queueRoot = Join-Path $repoRoot "artifacts\queues"
New-Item -ItemType Directory -Force $queueRoot | Out-Null

if (-not $QueueName) {
    $QueueName = "mamba12h_fat_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$queueDir = Join-Path $queueRoot $QueueName
New-Item -ItemType Directory -Force $queueDir | Out-Null
$logPath = Join-Path $queueDir "queue.log"
$deadline = (Get-Date).AddHours($Hours)
$remoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba"
$remoteRootUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba"

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $logPath -Value $line
}

function Sync-FatQueueFiles {
    Write-QueueLog "Syncing queue configs and training script to FatMachine."
    $files = @(
        "scripts\run_train_mmdet3_manual.py",
        "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stage01_stable.py",
        "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion05_stable.py",
        "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion10_stable.py",
        "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_1x_visdrone_es_stable.py"
    )
    foreach ($rel in $files) {
        $src = Join-Path $repoRoot $rel
        $dstDir = (Split-Path ($rel -replace "\\", "/") -Parent)
        scp $src ("FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/{0}/" -f $dstDir) | Out-Null
    }
}

function Install-FatCudaProbe {
    $probe = @'
#!/usr/bin/env bash
set -uo pipefail
export PYTORCH_NVML_BASED_CUDA_CHECK=1
/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/tools/micromamba run -p /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/.mamba-env-cu128 python -X faulthandler - <<'PY'
import torch
print('torch', torch.__version__, flush=True)
print('is_available', torch.cuda.is_available(), flush=True)
print('device_count', torch.cuda.device_count(), flush=True)
print('tensor', torch.randn(1, device='cuda'), flush=True)
PY
'@
    $probePath = Join-Path $queueDir "fat_cuda_tensor_probe.sh"
    [System.IO.File]::WriteAllText($probePath, ($probe -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)
    scp $probePath "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/fat_cuda_tensor_probe_queue.sh" | Out-Null
}

function Test-FatCudaReady {
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $out = ssh FatMachine "wsl -d Ubuntu-24.04 -u ns3user -- bash $remoteRootUnix/fat_cuda_tensor_probe_queue.sh" 2>&1
        $exitCode = $LASTEXITCODE
        foreach ($line in $out) {
            Write-QueueLog ("cuda-probe: {0}" -f $line)
        }
        return ($exitCode -eq 0)
    } catch {
        Write-QueueLog ("Fat CUDA probe threw: {0}" -f $_.Exception.Message)
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Get-FatTrainProcessLines {
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $out = ssh FatMachine "wsl -d Ubuntu-24.04 -u ns3user -- bash -lc `"ps -eo pid,ppid,etime,args | grep run_train_mmdet3_manual.py | grep -v grep || true`"" 2>$null
        return @($out | Where-Object { $_ -match "run_train_mmdet3_manual.py" })
    } catch {
        Write-QueueLog ("Failed to query Fat WSL processes: {0}" -f $_.Exception.Message)
        return @()
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Wait-ForFatRunEnd {
    param([string]$RunId)
    Write-QueueLog ("Waiting for Fat run to finish: {0}" -f $RunId)
    while ((Get-Date) -lt $deadline) {
        $procs = Get-FatTrainProcessLines
        $matched = @($procs | Where-Object { $_ -match [regex]::Escape($RunId) })
        if ($matched.Count -eq 0) {
            Write-QueueLog ("Fat run process no longer active: {0}" -f $RunId)
            return $true
        }
        Write-QueueLog ("Fat run still active: {0}" -f $RunId)
        Start-Sleep -Seconds 300
    }
    Write-QueueLog "Deadline reached while waiting for Fat run."
    return $false
}

function Write-FatRunSummary {
    param([string]$RunId)
    $remoteRun = "C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\$RunId"
    $script = @"
`$run = '$remoteRun'
if (Test-Path (Join-Path `$run 'eval_metrics.json')) {
  Write-Host '--- eval ---'
  Get-Content (Join-Path `$run 'eval_metrics.json') -Tail 80
} else {
  Write-Host '--- no eval ---'
}
if (Test-Path (Join-Path `$run 'launcher.log')) {
  Write-Host '--- launcher tail ---'
  Get-Content (Join-Path `$run 'launcher.log') -Tail 40
}
if (Test-Path (Join-Path `$run 'train.log')) {
  Write-Host '--- train tail ---'
  Get-Content (Join-Path `$run 'train.log') -Tail 40
}
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $out = ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded" 2>&1
    foreach ($line in $out) {
        Write-QueueLog ("fat-summary: {0}" -f $line)
    }
}

function Start-FatExperiment {
    param(
        [string]$Tag,
        [string]$ConfigPath,
        [int]$GpuMemGb
    )
    if ((Get-Date) -ge $deadline) {
        Write-QueueLog ("Skip {0}: deadline already reached." -f $Tag)
        return $null
    }

    while ((Get-Date) -lt $deadline) {
        if (Test-FatCudaReady) {
            break
        }
        Write-QueueLog "Fat CUDA is not ready; retrying after 10 minutes."
        Start-Sleep -Seconds 600
    }
    if ((Get-Date) -ge $deadline) {
        return $null
    }

    $runId = "{0}_{1}" -f $Tag, (Get-Date -Format "yyyyMMdd_HHmmss")
    $launcher = Join-Path $repoRoot "scripts\start_fatmachine_run_blackwell.ps1"
    $args = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-ConfigPath", $ConfigPath,
        "-RunId", $runId,
        "-GpuMemGb", $GpuMemGb,
        "-TorchNumThreads", "2",
        "-InteropThreads", "1",
        "-AdaptiveGuard",
        "-Background"
    )

    Write-QueueLog ("Launching Fat run {0}" -f $runId)
    $out = & powershell @args 2>&1
    foreach ($line in $out) {
        Write-QueueLog ("launch: {0}" -f $line)
    }

    Start-Sleep -Seconds 180
    $remoteRunDir = "C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\$runId"
    $manifestCheck = ssh FatMachine "cmd /c if exist `"$remoteRunDir\RUN_MANIFEST.json`" (echo MANIFEST_OK) else (echo NO_MANIFEST)" 2>$null
    foreach ($line in $manifestCheck) {
        Write-QueueLog ("manifest-check: {0}" -f $line)
    }

    [void](Wait-ForFatRunEnd -RunId $runId)
    Write-FatRunSummary -RunId $runId
    return $runId
}

Write-QueueLog ("Fat 12h queue started. Deadline={0}" -f $deadline.ToString("yyyy-MM-dd HH:mm:ss"))
Sync-FatQueueFiles
Install-FatCudaProbe

$jobs = @(
    @{
        Tag = "fat_hybridmamba_base1x_stable_queue"
        Config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmamba_base_b_fpn_1x_visdrone_es_stable.py"
        GpuMemGb = 22
    },
    @{
        Tag = "fat_hybridmambadet_stage01_stable_queue"
        Config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stage01_stable.py"
        GpuMemGb = 22
    },
    @{
        Tag = "fat_hybridmambadet_fusion10_stable_queue"
        Config = Join-Path $repoRoot "code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion10_stable.py"
        GpuMemGb = 22
    }
)

foreach ($job in $jobs) {
    if ((Get-Date) -ge $deadline) {
        Write-QueueLog "Deadline reached before launching remaining Fat jobs."
        break
    }
    [void](Start-FatExperiment -Tag $job.Tag -ConfigPath $job.Config -GpuMemGb $job.GpuMemGb)
}

Write-QueueLog "Fat 12h queue finished or reached deadline."
