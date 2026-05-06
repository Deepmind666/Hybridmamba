param(
    [string]$DatasetRoot = "C:\mamba\data\imagenet",
    [string]$WeightsRoot = "C:\mamba\weights\tinyvim",
    [string]$OutputRoot = "C:\mamba\artifacts\runs",
    [string]$RunId = "",
    [string]$Model = "TinyViM_B",
    [string]$TeacherModel = "regnety_160",
    [string]$TeacherPath = "",
    [int]$BatchSize = 64,
    [int]$Epochs = 100,
    [double]$BaseLr = 0.004,
    [int]$InputSize = 224,
    [int]$NumWorkers = 8,
    [int]$TorchNumThreads = 4,
    [int]$InteropThreads = 1,
    [int]$Niceness = 5,
    [switch]$AllowNoTeacher,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$wslRepoRoot = "/mnt/c/mamba"

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = Resolve-Path $Path
    $full = $resolved.Path
    if ($full -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert path to WSL form: $Path"
}

function Convert-WindowsPathStringToWsl {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert Windows path string to WSL form: $Path"
}

function Get-ImageNetLayout {
    param([string]$Root)
    $train = Join-Path $Root "train"
    $val = Join-Path $Root "val"
    $trainTar = Join-Path $Root "train.tar"
    $valTar = Join-Path $Root "val.tar"
    $hasFolders = (Test-Path $train) -and (Test-Path $val)
    $hasTars = (Test-Path $trainTar) -and (Test-Path $valTar)
    $trainClasses = if (Test-Path $train) { @(Get-ChildItem -Path $train -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
    $valClasses = if (Test-Path $val) { @(Get-ChildItem -Path $val -Directory -ErrorAction SilentlyContinue).Count } else { 0 }
    return [pscustomobject]@{
        RootExists = Test-Path $Root
        HasFolders = $hasFolders
        HasTars = $hasTars
        Ready = ($hasFolders -or $hasTars)
        Detail = ("folders={0} train_classes={1} val_classes={2}; tars={3}" -f $hasFolders, $trainClasses, $valClasses, $hasTars)
    }
}

if (-not $RunId) {
    $RunId = "tinyvim_imagenet1k_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}
$RunId = [regex]::Replace($RunId, '[^A-Za-z0-9._-]', '_')

if (-not $TeacherPath) {
    $TeacherPath = Join-Path $WeightsRoot "regnety_160-a5fe301d.pth"
}

$datasetLayout = Get-ImageNetLayout $DatasetRoot
if (-not $datasetLayout.Ready -and -not $DryRun) {
    throw "Missing usable ImageNet-1K data under $DatasetRoot. Need train/val folders or train.tar/val.tar. Current: $($datasetLayout.Detail)"
}

$teacherExists = Test-Path $TeacherPath
if (-not $teacherExists -and -not $AllowNoTeacher -and -not $DryRun) {
    throw "Missing TinyViM teacher checkpoint: $TeacherPath. Place the official RegNetY-160 weight here or rerun with -AllowNoTeacher."
}

$runDirWin = Join-Path $OutputRoot $RunId
if ((Test-Path $runDirWin) -and -not $DryRun) {
    throw "Run directory already exists: $runDirWin"
}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force $runDirWin | Out-Null
}

$runDirWsl = Convert-WindowsPathStringToWsl $runDirWin
$datasetWsl = if ($datasetLayout.RootExists) { Convert-ToWslPath $DatasetRoot } else { Convert-WindowsPathStringToWsl $DatasetRoot }
$teacherWsl = if ($teacherExists) { Convert-ToWslPath $TeacherPath } else { "" }

$launcherSh = Join-Path $env:TEMP ("tinyvim_imagenet1k_{0}.sh" -f $RunId)
$launcherCmd = Join-Path $env:TEMP ("tinyvim_imagenet1k_{0}.cmd" -f $RunId)
$launcherShWsl = Convert-WindowsPathStringToWsl $launcherSh

$distillArgs = if ($teacherExists) {
    @(
        "--distillation-type", "hard",
        "--teacher-model", $TeacherModel,
        "--teacher-path", $teacherWsl
    )
} else {
    @("--distillation-type", "none")
}
$distillLine = $distillArgs -join " "

$effectiveLr = [Math]::Round($BaseLr * $BatchSize / 512.0, 6)

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
mkdir -p "__RUN_DIR__"
if [[ -x "$HOME/.local/bin/micromamba" && -d "__WSL_REPO_ROOT__/.mamba-env-cu128" ]]; then
  "$HOME/.local/bin/micromamba" run -p "__WSL_REPO_ROOT__/.mamba-env-cu128" python code/tinyvim/main.py \
    --model "__MODEL__" \
    --batch-size __BATCH_SIZE__ \
    --epochs __EPOCHS__ \
    --input-size __INPUT_SIZE__ \
    --model-ema \
    --opt adamw \
    --weight-decay 0.025 \
    --lr __BASE_LR__ \
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
    __DISTILL_ARGS__ \
    > "__RUN_DIR__/launcher.log" 2>&1
  exit $?
elif [[ -x "$HOME/.local/bin/micromamba" && -d "__WSL_REPO_ROOT__/.mamba-env" ]]; then
  set +u
  eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
  micromamba activate "__WSL_REPO_ROOT__/.mamba-env"
  set -u
elif [[ -f "__WSL_REPO_ROOT__/.venv-wsl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "__WSL_REPO_ROOT__/.venv-wsl/bin/activate"
fi
cd "__WSL_REPO_ROOT__"
exec nice -n __NICENESS__ python code/tinyvim/main.py \
  --model "__MODEL__" \
  --batch-size __BATCH_SIZE__ \
  --epochs __EPOCHS__ \
  --input-size __INPUT_SIZE__ \
  --model-ema \
  --opt adamw \
  --weight-decay 0.025 \
  --lr __BASE_LR__ \
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
  __DISTILL_ARGS__ \
  > "__RUN_DIR__/launcher.log" 2>&1
'@

$launcherShContent = $launcherShTemplate
foreach ($replacement in @(
    @{ Old = '__TORCH_THREADS__'; New = "$TorchNumThreads" },
    @{ Old = '__INTEROP_THREADS__'; New = "$InteropThreads" },
    @{ Old = '__RUN_DIR__'; New = $runDirWsl },
    @{ Old = '__WSL_REPO_ROOT__'; New = $wslRepoRoot },
    @{ Old = '__NICENESS__'; New = "$Niceness" },
    @{ Old = '__MODEL__'; New = $Model },
    @{ Old = '__BATCH_SIZE__'; New = "$BatchSize" },
    @{ Old = '__EPOCHS__'; New = "$Epochs" },
    @{ Old = '__INPUT_SIZE__'; New = "$InputSize" },
    @{ Old = '__BASE_LR__'; New = "$BaseLr" },
    @{ Old = '__DATASET_WSL__'; New = $datasetWsl },
    @{ Old = '__NUM_WORKERS__'; New = "$NumWorkers" },
    @{ Old = '__DISTILL_ARGS__'; New = $distillLine }
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
    Write-Host ("Run id            {0}" -f $RunId)
    Write-Host ("Run dir           {0}" -f $runDirWin)
    Write-Host ("Dataset root      {0} ({1})" -f $DatasetRoot, $(if ($datasetLayout.RootExists) { "exists" } else { "missing" }))
    Write-Host ("Dataset layout    {0}" -f $datasetLayout.Detail)
    Write-Host ("Teacher path      {0} ({1})" -f $TeacherPath, $(if ($teacherExists) { "exists" } else { "missing" }))
    Write-Host ("Batch size        {0}" -f $BatchSize)
    Write-Host ("Epochs            {0}" -f $Epochs)
    Write-Host ("Base lr           {0}" -f $BaseLr)
    Write-Host ("Effective lr      {0}" -f $effectiveLr)
    Write-Host ("Distillation      {0}" -f $(if ($teacherExists) { "hard" } else { "none" }))
    Write-Host ("Launcher script   {0}" -f $launcherCmd)
    return
}

$wslArgs = @("-d", "Ubuntu-24.04", "--", "bash", $launcherShWsl)
Start-Process -FilePath "wsl.exe" -ArgumentList $wslArgs -WindowStyle Hidden | Out-Null

Write-Host ("Run id            {0}" -f $RunId)
Write-Host ("Run dir           {0}" -f $runDirWin)
Write-Host ("Dataset root      {0}" -f $DatasetRoot)
Write-Host ("Teacher path      {0}" -f $(if ($teacherExists) { $TeacherPath } else { "<none>" }))
Write-Host ("Batch size        {0}" -f $BatchSize)
Write-Host ("Epochs            {0}" -f $Epochs)
Write-Host ("Base lr           {0}" -f $BaseLr)
Write-Host ("Effective lr      {0}" -f $effectiveLr)
Write-Host ("Distillation      {0}" -f $(if ($teacherExists) { "hard" } else { "none" }))
Write-Host ("Launcher script   {0}" -f $launcherCmd)
Write-Host "Local TinyViM ImageNet-1K run launched."
