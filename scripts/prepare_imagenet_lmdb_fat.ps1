param(
    [string]$SourceRootWin = "C:\Users\sshuser\data\imagenet",
    [string]$TargetRootWsl = "/home/ns3user/data/imagenet_lmdb",
    [string]$RemoteRootWin = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$RunId = "",
    [int]$LmdbWorkers = 8,
    [int]$WriteFrequency = 5000,
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
$remoteLauncherWin = "C:\Users\sshuser\AppData\Local\Temp\prepare_lmdb_$RunId.sh"
$localLauncher = Join-Path $env:TEMP ("prepare_lmdb_$RunId.sh")
$remoteLauncherWsl = "/mnt/c/Users/sshuser/AppData/Local/Temp/prepare_lmdb_$RunId.sh"

$force = if ($ForceRebuild) { "1" } else { "0" }
$launcher = @"
#!/usr/bin/env bash
set -euo pipefail
src="$sourceRootWsl"
dst="$TargetRootWsl"
run_root="/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/data_prep/$RunId"
mkdir -p "`$run_root" "`$dst"
exec > >(tee -a "`$run_root/prepare.log") 2>&1
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

Write-Host "Fat ImageNet LMDB preparation"
Write-Host ("Source        {0}" -f $SourceRootWin)
Write-Host ("Target WSL    {0}" -f $TargetRootWsl)
Write-Host ("Workers       {0}" -f $LmdbWorkers)
Write-Host "ETA           rsync + LMDB conversion commonly takes 3-8 hours for full ImageNet; resumable for folder sync"
Write-Host "Resource note CPU/disk busy; do not overlap with latency-sensitive work if avoidable"
if ($DryRun) {
    Write-Host ("Local staged  {0}" -f $localLauncher)
    Write-Host ("Remote script {0}" -f $remoteLauncherWin)
    return
}

ssh FatMachine "powershell -NoProfile -Command `"New-Item -ItemType Directory -Force '$remoteRunRootWin' | Out-Null`"" | Out-Null
scp $localLauncher ("FatMachine:" + $remoteLauncherWin.Replace("\", "/")) | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy LMDB launcher to Fat"
}
$remoteLaunch = @"
`$session = 'prep_$RunId'
& wsl.exe -d Ubuntu-24.04 -u ns3user -- tmux new-session -d -s `$session "bash '$remoteLauncherWsl'"
if (`$LASTEXITCODE -ne 0) { throw "Failed to create tmux session: `$session" }
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteLaunch))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded" | Out-Null
Write-Host "Started Fat LMDB preparation in tmux."
Write-Host ("Remote log    {0}" -f (Join-Path $remoteRunRootWin "prepare.log"))
