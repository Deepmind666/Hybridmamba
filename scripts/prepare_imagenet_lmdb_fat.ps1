param(
    [string]$SourceRootWin = "C:\Users\sshuser\data\imagenet",
    [string]$TargetRootWsl = "/home/ns3user/data/imagenet_lmdb",
    [string]$RemoteRootWin = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$RunId = "",
    [int]$LmdbWorkers = 8,
    [int]$WriteFrequency = 5000,
    [ValidateSet("SshKeepAlive", "ScheduledTask")]
    [string]$LaunchMode = "SshKeepAlive",
    [switch]$ForceRebuild,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if (-not $RunId) {
    $RunId = "fat_imagenet_lmdb_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

$remoteRootWsl = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba"
$remoteEnv = "$remoteRootWsl/.mamba-env-cu128"
$remoteMicromamba = "$remoteRootWsl/artifacts/tools/micromamba"
$sourceRootWsl = "/mnt/c/Users/sshuser/data/imagenet"
$remoteRunRootWin = Join-Path $RemoteRootWin "artifacts\data_prep\$RunId"
$localRunRoot = Join-Path "C:\mamba\artifacts\data_prep" $RunId
$remoteLauncherWin = "C:\Users\sshuser\AppData\Local\Temp\prepare_lmdb_$RunId.sh"
$localLauncher = Join-Path $env:TEMP ("prepare_lmdb_$RunId.sh")
$remoteLauncherWsl = "/mnt/c/Users/sshuser/AppData/Local/Temp/prepare_lmdb_$RunId.sh"
$remoteTaskLauncherWin = "C:\Users\sshuser\AppData\Local\Temp\launch_lmdb_$RunId.ps1"
$localTaskLauncher = Join-Path $env:TEMP ("launch_lmdb_$RunId.ps1")

$force = if ($ForceRebuild) { "1" } else { "0" }
$launcher = @"
#!/usr/bin/env bash
set -euo pipefail
src="$sourceRootWsl"
dst="$TargetRootWsl"
run_root="/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/data_prep/$RunId"
mkdir -p "`$run_root" "`$dst"
exec >> "`$run_root/prepare.log" 2>&1
echo "[`$(date '+%F %T')] sync ImageNet folders to ext4"
rsync -a --info=progress2 --partial "`$src/train/" "`$dst/train/"
rsync -a --info=progress2 --partial "`$src/val/" "`$dst/val/"
export LMDB_ROOT="`$dst"
export LMDB_WORKERS="$LmdbWorkers"
export LMDB_WRITE_FREQUENCY="$WriteFrequency"
export LMDB_FORCE_REBUILD="$force"
cd "$remoteRootWsl/external/mobilemamba"
"$remoteMicromamba" run -p "$remoteEnv" python - <<'PY'
import os
import os.path as osp
import pickle

import lmdb
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

root = os.environ["LMDB_ROOT"]
workers = int(os.environ.get("LMDB_WORKERS", "8"))
write_frequency = int(os.environ.get("LMDB_WRITE_FREQUENCY", "5000"))
force = os.environ.get("LMDB_FORCE_REBUILD", "0") == "1"

def raw_reader(path):
    with open(path, "rb") as f:
        return f.read()

def build(name):
    img_dir = osp.join(root, name)
    lmdb_path = osp.join(root, f"{name}.lmdb")
    if osp.exists(osp.join(lmdb_path, "data.mdb")) and not force:
        print(f"{name}.lmdb exists, skip")
        return
    dataset = ImageFolder(root=img_dir, loader=raw_reader)
    loader = DataLoader(dataset, batch_size=1, num_workers=workers, collate_fn=lambda x: x)
    db = lmdb.open(lmdb_path, subdir=True, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)
    txn = db.begin(write=True)
    count = 0
    for idx, data in enumerate(loader):
        image, label = data[0]
        txn.put(str(idx).encode("ascii"), pickle.dumps((image, label)))
        count = idx + 1
        if count % write_frequency == 0:
            print(f"{name} {count}/{len(loader)}")
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    keys = [str(k).encode("ascii") for k in range(count)]
    txn = db.begin(write=True)
    txn.put(b"__keys__", pickle.dumps(keys))
    txn.put(b"__len__", pickle.dumps(len(keys)))
    txn.commit()
    db.sync()
    db.close()
    print(f"{name}.lmdb ready: {count} samples")

build("train")
build("val")
PY
echo "[`$(date '+%F %T')] ready: `$dst"
"@
[System.IO.File]::WriteAllText($localLauncher, ($launcher -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)
$taskLauncher = @"
`$ErrorActionPreference = 'Stop'
Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', 'Ubuntu-24.04', '-u', 'ns3user', '--exec', 'bash', '$remoteLauncherWsl') -WindowStyle Hidden -Wait
"@
[System.IO.File]::WriteAllText($localTaskLauncher, ($taskLauncher -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)
New-Item -ItemType Directory -Force $localRunRoot | Out-Null

Write-Host "Fat ImageNet LMDB preparation"
Write-Host ("Source        {0}" -f $SourceRootWin)
Write-Host ("Target WSL    {0}" -f $TargetRootWsl)
Write-Host ("Workers       {0}" -f $LmdbWorkers)
Write-Host ("Launch mode   {0}" -f $LaunchMode)
Write-Host "ETA           rsync + LMDB conversion commonly takes 3-8 hours for full ImageNet; resumable for folder sync"
Write-Host "Resource note CPU/disk busy; do not overlap with latency-sensitive work if avoidable"
if ($DryRun) {
    Write-Host ("Local staged  {0}" -f $localLauncher)
    Write-Host ("Remote script {0}" -f $remoteLauncherWin)
    return
}

$mkdirRemote = @"
New-Item -ItemType Directory -Force '$remoteRunRootWin' | Out-Null
"@
$mkdirEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($mkdirRemote))
ssh FatMachine "powershell -NoProfile -EncodedCommand $mkdirEncoded" | Out-Null
scp $localLauncher ("FatMachine:" + $remoteLauncherWin.Replace("\", "/")) | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy LMDB launcher to Fat"
}
scp $localTaskLauncher ("FatMachine:" + $remoteTaskLauncherWin.Replace("\", "/")) | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy scheduled-task launcher to Fat"
}
if ($LaunchMode -eq "SshKeepAlive") {
    $keepAlive = Join-Path $localRunRoot "run_fat_keepalive.ps1"
    $keepAliveOut = Join-Path $localRunRoot "ssh_keepalive.out.log"
    $keepAliveErr = Join-Path $localRunRoot "ssh_keepalive.err.log"
    $keepAliveScript = @"
`$ErrorActionPreference = 'Stop'
& ssh FatMachine "wsl -d Ubuntu-24.04 -u ns3user --exec bash $remoteLauncherWsl"
exit `$LASTEXITCODE
"@
    [System.IO.File]::WriteAllText($keepAlive, ($keepAliveScript -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)
    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $keepAlive) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $keepAliveOut `
        -RedirectStandardError $keepAliveErr `
        -PassThru
    Start-Sleep -Seconds 8
    if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
        throw "Fat SSH keepalive process exited immediately; see $keepAliveErr"
    }
    Write-Host "Started Fat LMDB preparation via local SSH keepalive."
    Write-Host ("Local keepalive pid {0}" -f $proc.Id)
    Write-Host ("Local logs          {0}" -f $localRunRoot)
    Write-Host ("Remote log          {0}" -f (Join-Path $remoteRunRootWin "prepare.log"))
    return
}
$remoteLaunch = @"
`$taskName = 'HybridMambaPrep_$RunId'
`$script = '$remoteTaskLauncherWin'
`$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + `$script + '"')
`$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
Register-ScheduledTask -TaskName `$taskName -Action `$action -Trigger `$trigger -Force | Out-Null
Start-ScheduledTask -TaskName `$taskName
Start-Sleep -Seconds 8
`$task = Get-ScheduledTask -TaskName `$taskName
`$info = Get-ScheduledTaskInfo -TaskName `$taskName
if (`$task.State -ne 'Running') {
    throw ("Fat scheduled task did not stay running: state={0}, lastTaskResult={1}" -f `$task.State, `$info.LastTaskResult)
}
Write-Output ("Started scheduled task {0}: state={1}" -f `$taskName, `$task.State)
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteLaunch))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded" | Out-Null
Write-Host "Started Fat LMDB preparation via Windows Scheduled Task."
Write-Host ("Remote log    {0}" -f (Join-Path $remoteRunRootWin "prepare.log"))
