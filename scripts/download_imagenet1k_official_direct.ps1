param(
    [string]$DatasetRoot = "C:\mamba\data\imagenet",
    [string]$LogRoot = "C:\mamba\artifacts\data_downloads\imagenet1k_official",
    [switch]$SkipTrain,
    [switch]$SkipVal,
    [switch]$SkipDevkit
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

New-Item -ItemType Directory -Force $DatasetRoot | Out-Null
New-Item -ItemType Directory -Force $LogRoot | Out-Null

$manifest = @(
    [pscustomobject]@{
        Name = "ILSVRC2012_devkit_t12.tar.gz"
        Url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
        Bytes = [int64]2568145
        Skip = [bool]$SkipDevkit
    },
    [pscustomobject]@{
        Name = "val.tar"
        Url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
        Bytes = [int64]6744924160
        Skip = [bool]$SkipVal
    },
    [pscustomobject]@{
        Name = "train.tar"
        Url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
        Bytes = [int64]147897477120
        Skip = [bool]$SkipTrain
    }
)

$mainLog = Join-Path $LogRoot "download.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message
    Add-Content -Path $mainLog -Value $line
    Write-Output $line
}

function Test-Complete {
    param([string]$Path, [int64]$Bytes)
    if (-not (Test-Path $Path)) {
        return $false
    }
    return ((Get-Item $Path).Length -eq $Bytes)
}

Write-Log ("Official ImageNet-1K download started. root={0}" -f $DatasetRoot)
foreach ($item in $manifest) {
    if ($item.Skip) {
        Write-Log ("SKIP requested: {0}" -f $item.Name)
        continue
    }

    $outPath = Join-Path $DatasetRoot $item.Name
    $curlLog = Join-Path $LogRoot ("curl_{0}.log" -f $item.Name.Replace(".", "_"))

    if (Test-Complete $outPath $item.Bytes) {
        Write-Log ("Already complete: {0} bytes={1}" -f $outPath, $item.Bytes)
        continue
    }

    if (Test-Path $outPath) {
        Write-Log ("Resuming {0}: current={1} expected={2}" -f $outPath, (Get-Item $outPath).Length, $item.Bytes)
    } else {
        Write-Log ("Downloading {0}: expected={1}" -f $outPath, $item.Bytes)
    }

    $curlCmd = @(
        "curl.exe",
        "-L",
        "-C -",
        "--retry 999",
        "--retry-delay 10",
        "--retry-all-errors",
        "--connect-timeout 30",
        "--speed-time 120",
        "--speed-limit 1024",
        "--progress-bar",
        "-o `"$outPath`"",
        "`"$($item.Url)`"",
        ">> `"$curlLog`" 2>&1"
    ) -join " "
    & cmd.exe /d /c $curlCmd

    if ($LASTEXITCODE -ne 0) {
        throw ("curl failed for {0}; exit={1}; see {2}" -f $item.Name, $LASTEXITCODE, $curlLog)
    }

    $actual = if (Test-Path $outPath) { (Get-Item $outPath).Length } else { 0 }
    if ($actual -ne $item.Bytes) {
        throw ("Downloaded size mismatch for {0}: actual={1} expected={2}" -f $outPath, $actual, $item.Bytes)
    }
    Write-Log ("Complete: {0} bytes={1}" -f $outPath, $actual)
}

Write-Log "Official ImageNet-1K download finished."
