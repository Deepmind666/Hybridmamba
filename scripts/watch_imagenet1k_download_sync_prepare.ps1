param(
    [string]$LocalRoot = "C:\mamba\data\imagenet",
    [string]$FatRoot = "C:\Users\sshuser\data\imagenet",
    [string]$LogRoot = "C:\mamba\artifacts\data_downloads\imagenet1k_official",
    [string]$PrepareScript = "C:\mamba\scripts\prepare_imagenet1k_from_official_tars.py",
    [string]$FatPrepareScript = "C:\Users\sshuser\codex_runs\hybrid-mamba\scripts\prepare_imagenet1k_from_official_tars.py",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
$log = Join-Path $LogRoot "watch_sync_prepare.log"

$expected = @{
    "ILSVRC2012_devkit_t12.tar.gz" = [int64]2568145
    "val.tar" = [int64]6744924160
    "train.tar" = [int64]147897477120
}

function Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message
    Add-Content -Path $log -Value $line
    Write-Output $line
}

function Test-FileSize {
    param([string]$Path, [int64]$Bytes)
    return ((Test-Path $Path) -and ((Get-Item $Path).Length -eq $Bytes))
}

function Start-LocalPrepare {
    param([string]$Split)
    $out = Join-Path $LogRoot ("prepare_{0}_local_stdout.log" -f $Split)
    $err = Join-Path $LogRoot ("prepare_{0}_local_stderr.log" -f $Split)
    $existing = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "prepare_imagenet1k_from_official_tars.py" -and $_.CommandLine -match "--split $Split"
    })
    if ($existing.Count -gt 0) {
        Log ("Local prepare {0} already active: {1}" -f $Split, (($existing | Select-Object -First 3 -ExpandProperty ProcessId) -join ","))
        return
    }
    $p = Start-Process -FilePath python.exe -ArgumentList @($PrepareScript, "--root", $LocalRoot, "--split", $Split) -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Log ("Started local prepare {0}: pid={1}" -f $Split, $p.Id)
}

function Start-FatPrepare {
    param([string]$Split)
    $remote = @"
`$logRoot='C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\data_downloads\imagenet1k_official'
`$out=Join-Path `$logRoot 'prepare_${Split}_fat_stdout.log'
`$err=Join-Path `$logRoot 'prepare_${Split}_fat_stderr.log'
`$existing=@(Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -and `$_.CommandLine -match 'prepare_imagenet1k_from_official_tars.py' -and `$_.CommandLine -match '--split ${Split}' })
if(`$existing.Count -gt 0){ "Fat prepare ${Split} already active: " + ((`$existing | Select-Object -First 3 -ExpandProperty ProcessId) -join ','); exit 0 }
`$p=Start-Process -FilePath python.exe -ArgumentList @('$FatPrepareScript','--root','$FatRoot','--split','${Split}') -WindowStyle Hidden -RedirectStandardOutput `$out -RedirectStandardError `$err -PassThru
"Started Fat prepare ${Split}: pid=" + `$p.Id
"@
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remote))
    $result = ssh FatMachine "powershell -NoProfile -EncodedCommand $enc"
    Log (($result -join " ") -replace "\s+", " ")
}

Log "Watcher started."
Start-LocalPrepare "val"

while ($true) {
    $train = Join-Path $LocalRoot "train.tar"
    $val = Join-Path $LocalRoot "val.tar"
    $devkit = Join-Path $LocalRoot "ILSVRC2012_devkit_t12.tar.gz"
    $trainSize = if (Test-Path $train) { (Get-Item $train).Length } else { 0 }
    $valOk = Test-FileSize $val $expected["val.tar"]
    $devkitOk = Test-FileSize $devkit $expected["ILSVRC2012_devkit_t12.tar.gz"]
    $trainOk = Test-FileSize $train $expected["train.tar"]
    Log ("poll: devkit={0} val={1} train={2}/{3}" -f $devkitOk, $valOk, $trainSize, $expected["train.tar"])
    if ($devkitOk -and $valOk -and $trainOk) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Log "Local official tar set complete."
Start-LocalPrepare "train"

Log "Syncing train.tar to Fat."
scp (Join-Path $LocalRoot "train.tar") "FatMachine:C:/Users/sshuser/data/imagenet/train.tar" *> (Join-Path $LogRoot "scp_train_to_fat.log")
if ($LASTEXITCODE -ne 0) {
    Log ("ERROR: scp train.tar to Fat failed with exit={0}" -f $LASTEXITCODE)
    exit $LASTEXITCODE
}
Log "train.tar synced to Fat."
Start-FatPrepare "train"
Log "Watcher finished launch steps."
