param(
    [string]$Checkpoint = "C:\mamba\artifacts\runs\local_imagenet1k_tinyvim_b_100e_20260506_2019\TinyViM_B\2026_05_06_20_20_33\checkpoint_0.pth",
    [string]$DatasetRoot = "C:\mamba\data\imagenet",
    [string]$TeacherPath = "C:\mamba\weights\tinyvim\regnety_160-a5fe301d.pth",
    [int]$BatchSize = 64,
    [int]$Epochs = 100,
    [double]$Lr = 0.004,
    [int]$InputSize = 224,
    [int]$NumWorkers = 8,
    [int]$TorchNumThreads = 4,
    [int]$InteropThreads = 1,
    [int]$Niceness = 5,
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

if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint: $Checkpoint"
}
if (-not (Test-Path $DatasetRoot)) {
    throw "Missing dataset root: $DatasetRoot"
}
if (-not (Test-Path $TeacherPath)) {
    throw "Missing teacher checkpoint: $TeacherPath"
}

$checkpointWsl = Convert-ToWslPath $Checkpoint
$datasetWsl = Convert-ToWslPath $DatasetRoot
$teacherWsl = Convert-ToWslPath $TeacherPath
$runDirWin = Split-Path -Parent $Checkpoint
$runDirWsl = Convert-ToWslPath $runDirWin
$resumeStamp = Get-Date -Format "yyyyMMdd_HHmmss"

$launcherSh = Join-Path $env:TEMP ("tinyvim_resume_{0}.sh" -f $resumeStamp)
$launcherCmd = Join-Path $env:TEMP ("tinyvim_resume_{0}.cmd" -f $resumeStamp)
$launcherShWsl = Convert-WindowsPathStringToWsl $launcherSh

$launcherShTemplate = @'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=__TORCH_THREADS__
export MKL_NUM_THREADS=__TORCH_THREADS__
export OPENBLAS_NUM_THREADS=__TORCH_THREADS__
export TORCH_NUM_THREADS=__TORCH_THREADS__
export TORCH_NUM_INTEROP_THREADS=__INTEROP_THREADS__
cd "__WSL_REPO_ROOT__"
exec nice -n __NICENESS__ "$HOME/.local/bin/micromamba" run -p "__WSL_REPO_ROOT__/.mamba-env-cu128" python code/tinyvim/main.py \
  --model TinyViM_B \
  --batch-size __BATCH_SIZE__ \
  --epochs __EPOCHS__ \
  --input-size __INPUT_SIZE__ \
  --model-ema \
  --opt adamw \
  --weight-decay 0.025 \
  --lr __LR__ \
  --warmup-epochs 5 \
  --aa rand-m9-mstd0.5-inc1 \
  --smoothing 0.1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --data-set IMNET \
  --data-path "__DATASET_WSL__" \
  --output_dir "__RUN_DIR__" \
  --num_workers __NUM_WORKERS__ \
  --pin-mem \
  --dist-eval \
  --distillation-type hard \
  --teacher-model regnety_160 \
  --teacher-path "__TEACHER_WSL__" \
  --resume "__CHECKPOINT_WSL__" \
  > "__RUN_DIR__/resume___STAMP__.log" 2>&1
'@

$launcherShContent = $launcherShTemplate
foreach ($replacement in @(
    @{ Old = '__TORCH_THREADS__'; New = "$TorchNumThreads" },
    @{ Old = '__INTEROP_THREADS__'; New = "$InteropThreads" },
    @{ Old = '__WSL_REPO_ROOT__'; New = $wslRepoRoot },
    @{ Old = '__NICENESS__'; New = "$Niceness" },
    @{ Old = '__BATCH_SIZE__'; New = "$BatchSize" },
    @{ Old = '__EPOCHS__'; New = "$Epochs" },
    @{ Old = '__INPUT_SIZE__'; New = "$InputSize" },
    @{ Old = '__LR__'; New = "$Lr" },
    @{ Old = '__DATASET_WSL__'; New = $datasetWsl },
    @{ Old = '__RUN_DIR__'; New = $runDirWsl },
    @{ Old = '__NUM_WORKERS__'; New = "$NumWorkers" },
    @{ Old = '__TEACHER_WSL__'; New = $teacherWsl },
    @{ Old = '__CHECKPOINT_WSL__'; New = $checkpointWsl },
    @{ Old = '__STAMP__'; New = $resumeStamp }
)) {
    $launcherShContent = $launcherShContent.Replace($replacement.Old, [string]$replacement.New)
}

$launcherCmdContent = @"
@echo off
wsl -d Ubuntu-24.04 -- bash "$launcherShWsl"
"@

[System.IO.File]::WriteAllText($launcherSh, ($launcherShContent -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)
[System.IO.File]::WriteAllText($launcherCmd, ($launcherCmdContent -replace "`r`n", "`n"), [System.Text.Encoding]::ASCII)

if ($DryRun) {
    Write-Host ("Checkpoint        {0}" -f $Checkpoint)
    Write-Host ("Run dir           {0}" -f $runDirWin)
    Write-Host ("Dataset root      {0}" -f $DatasetRoot)
    Write-Host ("Teacher path      {0}" -f $TeacherPath)
    Write-Host ("Batch size        {0}" -f $BatchSize)
    Write-Host ("Epochs            {0}" -f $Epochs)
    Write-Host ("Base LR           {0}" -f $Lr)
    Write-Host ("Num workers       {0}" -f $NumWorkers)
    Write-Host ("Resume log        {0}" -f (Join-Path $runDirWin ("resume_{0}.log" -f $resumeStamp)))
    Write-Host ("Launcher script   {0}" -f $launcherCmd)
    return
}

$wslArgs = @("-d", "Ubuntu-24.04", "--", "bash", $launcherShWsl)
Start-Process -FilePath "wsl.exe" -ArgumentList $wslArgs -WindowStyle Hidden | Out-Null

Write-Host ("Checkpoint        {0}" -f $Checkpoint)
Write-Host ("Run dir           {0}" -f $runDirWin)
Write-Host ("Dataset root      {0}" -f $DatasetRoot)
Write-Host ("Teacher path      {0}" -f $TeacherPath)
Write-Host ("Batch size        {0}" -f $BatchSize)
Write-Host ("Epochs            {0}" -f $Epochs)
    Write-Host ("Base LR           {0}" -f $Lr)
Write-Host ("Num workers       {0}" -f $NumWorkers)
Write-Host ("Resume log        {0}" -f (Join-Path $runDirWin ("resume_{0}.log" -f $resumeStamp)))
Write-Host ("Launcher script   {0}" -f $launcherCmd)
Write-Host "Local TinyViM ImageNet-1K resume launched."
