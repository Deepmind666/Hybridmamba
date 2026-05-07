param(
    [string]$DatasetRootWsl = "/home/lkr/data/imagenet",
    [string]$WeightsRoot = "C:\mamba\weights\tinyvim",
    [string]$OutputRoot = "C:\mamba\artifacts\runs",
    [string]$RunId = "",
    [string]$Model = "TinyViM_B",
    [string]$TeacherModel = "regnety_160",
    [string]$TeacherPath = "",
    [string]$ResumeCheckpoint = "",
    [int]$BatchSize = 64,
    [int]$Epochs = 100,
    [double]$BaseLr = 0.004,
    [int]$InputSize = 224,
    [int]$NumWorkers = 4,
    [int]$TorchNumThreads = 2,
    [int]$InteropThreads = 1,
    [int]$Niceness = 8,
    [switch]$AllowNoTeacher,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = "C:\mamba"
$wslRepoRoot = "/mnt/c/mamba"

function Convert-WindowsPathStringToWsl {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert Windows path string to WSL form: $Path"
}

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = Resolve-Path $Path
    return Convert-WindowsPathStringToWsl $resolved.Path
}

if (-not $RunId) {
    $RunId = "tinyvim_b_imagenet1k_ext4_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

if (-not $TeacherPath) {
    $TeacherPath = Join-Path $WeightsRoot "regnety_160-a5fe301d.pth"
}
$teacherExists = Test-Path $TeacherPath
if (-not $teacherExists -and -not $AllowNoTeacher -and -not $DryRun) {
    throw "Missing TinyViM teacher checkpoint: $TeacherPath. Use -AllowNoTeacher only if intentionally disabling distillation."
}

$resumeWsl = ""
if ($ResumeCheckpoint) {
    if (-not (Test-Path $ResumeCheckpoint)) {
        throw "Missing resume checkpoint: $ResumeCheckpoint"
    }
    $resumeWsl = Convert-ToWslPath $ResumeCheckpoint
}

$ready = wsl -d Ubuntu-24.04 -- bash -lc "test -d '$DatasetRootWsl/train' && test -d '$DatasetRootWsl/val' && echo ready || echo missing"
if (($ready -notmatch "ready") -and -not $DryRun) {
    throw "ImageNet ext4 data is not ready: $DatasetRootWsl. Run scripts\prepare_imagenet_ext4_local.ps1 first."
}

$runDirWin = Join-Path $OutputRoot $RunId
if ((Test-Path $runDirWin) -and -not $DryRun) {
    throw "Run directory already exists: $runDirWin"
}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $runDirWin | Out-Null
}
$runDirWsl = Convert-WindowsPathStringToWsl $runDirWin
$teacherWsl = if ($teacherExists) { Convert-ToWslPath $TeacherPath } else { "" }
$distillLine = if ($teacherExists) {
    "--distillation-type hard --teacher-model $TeacherModel --teacher-path '$teacherWsl'"
} else {
    "--distillation-type none"
}
$resumeLine = if ($ResumeCheckpoint) { "--resume '$resumeWsl'" } else { "" }
$effectiveLr = [Math]::Round($BaseLr * $BatchSize / 512.0, 6)

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$launcherWin = Join-Path $env:TEMP ("tinyvim_ext4_$stamp.sh")
$launcherWsl = Convert-WindowsPathStringToWsl $launcherWin
$launcher = @"
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=$TorchNumThreads
export MKL_NUM_THREADS=$TorchNumThreads
export OPENBLAS_NUM_THREADS=$TorchNumThreads
export TORCH_NUM_THREADS=$TorchNumThreads
export TORCH_NUM_INTEROP_THREADS=$InteropThreads
cd "$wslRepoRoot"
mkdir -p "$runDirWsl"
exec nice -n $Niceness "`$HOME/.local/bin/micromamba" run -p "$wslRepoRoot/.mamba-env-cu128" python code/tinyvim/main.py \
  --model "$Model" \
  --batch-size $BatchSize \
  --epochs $Epochs \
  --input-size $InputSize \
  --model-ema \
  --opt adamw \
  --weight-decay 0.025 \
  --lr $BaseLr \
  --warmup-epochs 5 \
  --aa rand-m9-mstd0.5-inc1 \
  --smoothing 0.1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --data-set IMNET \
  --data-path "$DatasetRootWsl" \
  --output_dir "$runDirWsl" \
  --num_workers $NumWorkers \
  --pin-mem \
  --dist-eval \
  $distillLine \
  $resumeLine \
  > "$runDirWsl/launcher.log" 2>&1
"@
[System.IO.File]::WriteAllText($launcherWin, ($launcher -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)

Write-Host "TinyViM-B ext4 accelerated training"
Write-Host ("Run id        {0}" -f $RunId)
Write-Host ("Run dir       {0}" -f $runDirWin)
Write-Host ("Dataset WSL   {0}" -f $DatasetRootWsl)
Write-Host ("Batch size    {0}" -f $BatchSize)
Write-Host ("Workers       {0}" -f $NumWorkers)
Write-Host ("Effective lr  {0}" -f $effectiveLr)
Write-Host ("Resume        {0}" -f $(if ($ResumeCheckpoint) { $ResumeCheckpoint } else { "<none>" }))
Write-Host "ETA           after ext4 data is ready, expect around 1.5x-3x faster than /mnt/c if I/O was the bottleneck"
if ($DryRun) {
    Write-Host ("Launcher      {0}" -f $launcherWin)
    return
}

Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", $launcherWsl) -WindowStyle Hidden | Out-Null
Write-Host "Launched in background."
