param(
    [ValidateSet("Auto", "HuggingFace", "Kaggle", "TeacherOnly")]
    [string]$Source = "Auto",
    [string]$DatasetRoot = "C:\mamba\data\imagenet",
    [string]$DownloadRoot = "C:\mamba\data\downloads\imagenet1k",
    [string]$TeacherPath = "C:\mamba\weights\tinyvim\regnety_160-a5fe301d.pth",
    [string]$TeacherUrl = "https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth",
    [string]$HfRepo = "ILSVRC/imagenet-1k",
    [string]$HfToken = "",
    [string]$KaggleCompetition = "imagenet-object-localization-challenge",
    [switch]$SkipTeacher,
    [switch]$MaterializeFromHf,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message)
}

function Get-KaggleExe {
    $cmd = Get-Command kaggle -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $scriptRoot = Join-Path $env:APPDATA "Python"
    if (Test-Path $scriptRoot) {
        $userScript = Get-ChildItem -Path $scriptRoot -Filter kaggle.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\Scripts\\kaggle\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($userScript) { return $userScript.FullName }
    }
    return ""
}

function Get-HfExe {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $scriptRoot = Join-Path $env:APPDATA "Python"
    if (Test-Path $scriptRoot) {
        $userScript = Get-ChildItem -Path $scriptRoot -Filter "$Name.exe" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\Scripts\\$Name\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($userScript) { return $userScript.FullName }
    }
    return ""
}

function Test-HfAuth {
    if ($HfToken) { return $true }
    $env:PYTHONIOENCODING = "utf-8"
    $hf = Get-HfExe "hf"
    if ($hf) {
        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $out = (& $hf auth whoami 2>&1 | Out-String).Trim()
        $ErrorActionPreference = $oldEap
        return (($LASTEXITCODE -eq 0) -and $out -and ($out -notmatch "Not logged in"))
    }
    $old = Get-HfExe "huggingface-cli"
    if ($old) {
        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $out = (& $old whoami 2>&1 | Out-String).Trim()
        $ErrorActionPreference = $oldEap
        return (($LASTEXITCODE -eq 0) -and $out -and ($out -notmatch "Not logged in"))
    }
    return $false
}

function Test-KaggleAuth {
    return (Test-Path (Join-Path $env:USERPROFILE ".kaggle\kaggle.json"))
}

function Test-ImageNetRoot {
    param([string]$Root)
    $train = Join-Path $Root "train"
    $val = Join-Path $Root "val"
    $trainTar = Join-Path $Root "train.tar"
    $valTar = Join-Path $Root "val.tar"
    $folderReady = (Test-Path $train) -and (Test-Path $val)
    $tarReady = (Test-Path $trainTar) -and (Test-Path $valTar)
    return ($folderReady -or $tarReady)
}

function New-HardLinkOrCopy {
    param([string]$SourcePath, [string]$DestPath)
    if (Test-Path $DestPath) { return }
    try {
        New-Item -ItemType HardLink -Path $DestPath -Target $SourcePath | Out-Null
    } catch {
        Copy-Item -LiteralPath $SourcePath -Destination $DestPath
    }
}

New-Item -ItemType Directory -Force (Split-Path $TeacherPath -Parent) | Out-Null
New-Item -ItemType Directory -Force $DatasetRoot | Out-Null
New-Item -ItemType Directory -Force $DownloadRoot | Out-Null

if (-not $SkipTeacher) {
    if (Test-Path $TeacherPath) {
        $teacherSize = (Get-Item $TeacherPath).Length
        Write-Step ("TinyViM teacher OK: {0} bytes" -f $teacherSize)
    } elseif ($DryRun) {
        Write-Step ("Dry run: would download TinyViM teacher from {0}" -f $TeacherUrl)
    } else {
        Write-Step ("Downloading TinyViM teacher to {0}" -f $TeacherPath)
        Invoke-WebRequest -Uri $TeacherUrl -OutFile $TeacherPath
        Write-Step ("TinyViM teacher downloaded: {0} bytes" -f (Get-Item $TeacherPath).Length)
    }
}

if ($Source -eq "TeacherOnly") {
    return
}

if (Test-ImageNetRoot $DatasetRoot) {
    Write-Step ("ImageNet root already ready: {0}" -f $DatasetRoot)
    return
}

$hfReady = Test-HfAuth
$kaggleReady = Test-KaggleAuth
$kaggleExe = Get-KaggleExe

if ($Source -eq "Auto") {
    if ($hfReady) {
        $Source = "HuggingFace"
    } elseif ($kaggleReady -and $kaggleExe) {
        $Source = "Kaggle"
    } else {
        Write-Step "No authenticated ImageNet-1K source is configured."
        Write-Host ("Hugging Face auth: {0}" -f $(if ($hfReady) { "OK" } else { "MISSING" }))
        Write-Host ("Kaggle auth:       {0}" -f $(if ($kaggleReady) { "OK" } else { "MISSING C:\Users\admin\.kaggle\kaggle.json" }))
        Write-Host ("Kaggle CLI:        {0}" -f $(if ($kaggleExe) { $kaggleExe } else { "MISSING" }))
        if ($DryRun) { return }
        throw "ImageNet-1K is gated. Configure HF or Kaggle credentials, then rerun this script."
    }
}

if ($Source -eq "HuggingFace") {
    if (-not $hfReady) {
        throw "Hugging Face is not authenticated. Run `hf auth login` after accepting access to $HfRepo, or pass -HfToken."
    }
    if (-not $MaterializeFromHf) {
        Write-Step "Hugging Face access is ready. Add -MaterializeFromHf to start the long ImageFolder materialization job."
        return
    }
    $cacheDir = Join-Path (Split-Path $DownloadRoot -Parent) "hf_cache"
    $args = @(
        "C:\mamba\scripts\materialize_imagenet1k_hf.py",
        "--repo-id", $HfRepo,
        "--output-root", $DatasetRoot,
        "--cache-dir", $cacheDir
    )
    if ($HfToken) {
        $args += @("--token", $HfToken)
    }
    Write-Step ("Materializing Hugging Face ImageNet-1K into {0}" -f $DatasetRoot)
    & python @args
    return
}

if ($Source -eq "Kaggle") {
    if (-not $kaggleReady) {
        throw "Missing Kaggle credential file: C:\Users\admin\.kaggle\kaggle.json"
    }
    if (-not $kaggleExe) {
        throw "Missing kaggle CLI. Run: python -m pip install --user kaggle"
    }
    $kaggleRoot = Join-Path $DownloadRoot "kaggle"
    New-Item -ItemType Directory -Force $kaggleRoot | Out-Null
    if ($DryRun) {
        Write-Step ("Dry run: would download Kaggle competition {0} to {1}" -f $KaggleCompetition, $kaggleRoot)
        return
    }
    Write-Step ("Downloading Kaggle competition {0} to {1}" -f $KaggleCompetition, $kaggleRoot)
    & $kaggleExe competitions download -c $KaggleCompetition -p $kaggleRoot
    $trainTar = Get-ChildItem -Path $kaggleRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "ILSVRC2012.*train.*\.tar$|train\.tar$" } |
        Select-Object -First 1
    $valTar = Get-ChildItem -Path $kaggleRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "ILSVRC2012.*val.*\.tar$|val\.tar$" } |
        Select-Object -First 1
    if ($trainTar -and $valTar) {
        New-HardLinkOrCopy $trainTar.FullName (Join-Path $DatasetRoot "train.tar")
        New-HardLinkOrCopy $valTar.FullName (Join-Path $DatasetRoot "val.tar")
        Write-Step ("Linked train.tar/val.tar under {0}" -f $DatasetRoot)
    } else {
        Write-Step "Kaggle download completed, but train/val tar files were not found automatically. Inspect the Kaggle download folder."
    }
}
