param(
    [string]$LocalDatasetRoot = "C:\mamba\data\imagenet",
    [string]$TinyViMTeacherPath = "C:\mamba\weights\tinyvim\regnety_160-a5fe301d.pth",
    [string]$FatDatasetRoot = "C:\Users\sshuser\data\imagenet",
    [switch]$SkipFat
)

$ErrorActionPreference = "Stop"

function Write-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    $status = if ($Ok) { "OK" } else { "MISSING" }
    if ($Detail) {
        Write-Host ("{0,-28} {1,-8} {2}" -f $Name, $status, $Detail)
    } else {
        Write-Host ("{0,-28} {1}" -f $Name, $status)
    }
}

function Test-ImageNetRoot {
    param([string]$Root)
    $exists = Test-Path $Root
    if (-not $exists) {
        return @{ Exists = $false; Train = $false; Val = $false; Tar = $false; TrainClasses = 0; ValClasses = 0 }
    }
    $trainPath = Join-Path $Root "train"
    $valPath = Join-Path $Root "val"
    $train = Test-Path $trainPath
    $val = Test-Path $valPath
    $tar = (Test-Path (Join-Path $Root "train.tar")) -and (Test-Path (Join-Path $Root "val.tar"))
    $trainClasses = if ($train) { @(Get-ChildItem -Path $trainPath -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
    $valClasses = if ($val) { @(Get-ChildItem -Path $valPath -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
    return @{ Exists = $true; Train = $train; Val = $val; Tar = $tar; TrainClasses = $trainClasses; ValClasses = $valClasses }
}

function Get-UserScriptExe {
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

$local = Test-ImageNetRoot $LocalDatasetRoot
Write-Host "Local ImageNet-1K readiness"
Write-Check "dataset root" $local.Exists $LocalDatasetRoot
Write-Check "train/val folders" ($local.Train -and $local.Val) ("train_classes={0}, val_classes={1}" -f $local.TrainClasses, $local.ValClasses)
Write-Check "train.tar/val.tar" $local.Tar "TinyViM fallback format"
Write-Check "TinyViM teacher" (Test-Path $TinyViMTeacherPath) $TinyViMTeacherPath
$kaggleExe = Get-UserScriptExe "kaggle"
Write-Check "Kaggle CLI" ([bool]$kaggleExe) $(if ($kaggleExe) { $kaggleExe } else { "install with python -m pip install --user kaggle" })
Write-Check "Kaggle credentials" (Test-Path (Join-Path $env:USERPROFILE ".kaggle\kaggle.json")) (Join-Path $env:USERPROFILE ".kaggle\kaggle.json")
$env:PYTHONIOENCODING = "utf-8"
$hfAuthed = $false
$hf = Get-UserScriptExe "hf"
if ($hf) {
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $hfOut = (& $hf auth whoami 2>&1 | Out-String).Trim()
    $ErrorActionPreference = $oldEap
    $hfAuthed = (($LASTEXITCODE -eq 0) -and $hfOut -and ($hfOut -notmatch "Not logged in"))
} else {
    $oldHf = Get-UserScriptExe "huggingface-cli"
    if ($oldHf) {
        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $hfOut = (& $oldHf whoami 2>&1 | Out-String).Trim()
        $ErrorActionPreference = $oldEap
        $hfAuthed = (($LASTEXITCODE -eq 0) -and $hfOut -and ($hfOut -notmatch "Not logged in"))
    }
}
Write-Check "Hugging Face auth" $hfAuthed "required for gated ILSVRC/imagenet-1k"
Write-Host ""

if (-not $SkipFat) {
    Write-Host "Fat ImageNet-1K readiness"
    $remoteScript = @"
`$ProgressPreference = 'SilentlyContinue'
`$checks = @(
  @('MobileMamba code', 'C:\Users\sshuser\codex_runs\hybrid-mamba\external\mobilemamba\run.py'),
  @('micromamba', 'C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\tools\micromamba'),
  @('cu128 env', 'C:\Users\sshuser\codex_runs\hybrid-mamba\.mamba-env-cu128'),
  @('dataset root', '$FatDatasetRoot')
)
foreach (`$item in `$checks) {
  `$name = `$item[0]
  `$path = `$item[1]
  if (Test-Path `$path) {
    Write-Output ("{0,-28} OK       {1}" -f `$name, `$path)
  } else {
    Write-Output ("{0,-28} MISSING  {1}" -f `$name, `$path)
  }
}
`$train = Join-Path '$FatDatasetRoot' 'train'
`$val = Join-Path '$FatDatasetRoot' 'val'
if ((Test-Path `$train) -and (Test-Path `$val)) {
  `$trainClasses = @(Get-ChildItem -Path `$train -Directory -ErrorAction SilentlyContinue).Count
  `$valClasses = @(Get-ChildItem -Path `$val -Directory -ErrorAction SilentlyContinue).Count
  Write-Output ("{0,-28} OK       train_classes={1}, val_classes={2}" -f 'train/val folders', `$trainClasses, `$valClasses)
} else {
  Write-Output ("{0,-28} MISSING  train/val folders under dataset root" -f 'train/val folders')
}
function Get-UserScriptExe {
  param([string]`$Name)
  `$cmd = Get-Command `$Name -ErrorAction SilentlyContinue
  if (`$cmd) { return `$cmd.Source }
  `$scriptRoot = Join-Path `$env:APPDATA 'Python'
  if (Test-Path `$scriptRoot) {
    `$userScript = Get-ChildItem -Path `$scriptRoot -Filter "`$Name.exe" -Recurse -ErrorAction SilentlyContinue |
      Where-Object { `$_.FullName -match "\\Scripts\\`$Name\.exe`$" } |
      Sort-Object FullName -Descending |
      Select-Object -First 1
    if (`$userScript) { return `$userScript.FullName }
  }
  return ''
}
`$kaggle = Get-UserScriptExe 'kaggle'
if (`$kaggle) {
  Write-Output ("{0,-28} OK       {1}" -f 'Kaggle CLI', `$kaggle)
} else {
  Write-Output ("{0,-28} MISSING  install with python -m pip install --user kaggle" -f 'Kaggle CLI')
}
`$kaggleJson = Join-Path `$env:USERPROFILE '.kaggle\kaggle.json'
if (Test-Path `$kaggleJson) {
  Write-Output ("{0,-28} OK       {1}" -f 'Kaggle credentials', `$kaggleJson)
} else {
  Write-Output ("{0,-28} MISSING  {1}" -f 'Kaggle credentials', `$kaggleJson)
}
`$hfAuthed = `$false
`$hf = Get-UserScriptExe 'hf'
if (`$hf) {
  `$oldEap = `$ErrorActionPreference
  `$ErrorActionPreference = 'Continue'
  `$hfOut = (& `$hf auth whoami 2>&1 | Out-String).Trim()
  `$ErrorActionPreference = `$oldEap
  `$hfAuthed = ((`$LASTEXITCODE -eq 0) -and `$hfOut -and (`$hfOut -notmatch 'Not logged in'))
}
if (`$hfAuthed) {
  Write-Output ("{0,-28} OK       authenticated" -f 'Hugging Face auth')
} else {
  Write-Output ("{0,-28} MISSING  required for gated ILSVRC/imagenet-1k" -f 'Hugging Face auth')
}
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
    ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded"
}
