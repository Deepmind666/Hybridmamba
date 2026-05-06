param(
    [string]$LocalRoot = "C:\mamba\data\imagenet",
    [string]$FatTrain = "C:\Users\sshuser\data\imagenet\train.tar",
    [string]$LogRoot = "C:\mamba\artifacts\data_downloads\imagenet1k_official",
    [string]$PrepareScript = "C:\mamba\scripts\prepare_imagenet1k_from_official_tars.py"
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $LocalRoot | Out-Null
New-Item -ItemType Directory -Force $LogRoot | Out-Null

$expected = [int64]147897477120
$log = Join-Path $LogRoot "sync_train_from_fat_prepare_local.log"
$localTrain = Join-Path $LocalRoot "train.tar"
$incoming = Join-Path $LocalRoot "train.tar.incoming"

function Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message
    Add-Content -Path $log -Value $line
    Write-Output $line
}

function Test-Complete {
    param([string]$Path)
    return ((Test-Path $Path) -and ((Get-Item $Path).Length -eq $expected))
}

Log "Sync-from-Fat workflow started."
if (Test-Complete $localTrain) {
    Log ("Local train.tar already complete: {0}" -f $localTrain)
} else {
    if (Test-Path $localTrain) {
        $partialLen = (Get-Item $localTrain).Length
        Log ("Removing incomplete local train.tar before LAN sync: bytes={0}" -f $partialLen)
        Remove-Item -LiteralPath $localTrain -Force
    }
    if (Test-Path $incoming) {
        Log ("Removing stale incoming file: bytes={0}" -f (Get-Item $incoming).Length)
        Remove-Item -LiteralPath $incoming -Force
    }
    Log ("Copying Fat train.tar to local incoming file: {0}" -f $incoming)
    scp "FatMachine:C:/Users/sshuser/data/imagenet/train.tar" $incoming *> (Join-Path $LogRoot "scp_train_from_fat.log")
    if ($LASTEXITCODE -ne 0) {
        Log ("ERROR: scp train.tar from Fat failed with exit={0}" -f $LASTEXITCODE)
        exit $LASTEXITCODE
    }
    if (-not (Test-Complete $incoming)) {
        $len = if (Test-Path $incoming) { (Get-Item $incoming).Length } else { 0 }
        Log ("ERROR: incoming size mismatch: actual={0} expected={1}" -f $len, $expected)
        exit 3
    }
    Move-Item -LiteralPath $incoming -Destination $localTrain -Force
    Log "Local train.tar is complete."
}

$trainDir = Join-Path $LocalRoot "train"
$complete = Join-Path $trainDir ".complete"
if (Test-Path $complete) {
    Log "Local train ImageFolder already complete."
} else {
    Log "Starting local train ImageFolder materialization."
    python $PrepareScript --root $LocalRoot --split train *> (Join-Path $LogRoot "prepare_train_local_from_fat.log")
    if ($LASTEXITCODE -ne 0) {
        Log ("ERROR: local train materialization failed with exit={0}" -f $LASTEXITCODE)
        exit $LASTEXITCODE
    }
    Log "Local train ImageFolder materialization finished."
}

Log "Sync-from-Fat workflow finished."
