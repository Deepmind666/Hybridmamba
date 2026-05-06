$ErrorActionPreference = 'Stop'

$repoRoot = 'C:\mamba'
$outRoot = Join-Path $repoRoot 'artifacts\analysis\model_efficiency'
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$cmd = @"
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
cd /mnt/c/mamba
~/.local/bin/micromamba run -p /mnt/c/mamba/.mamba-env-cu128 python scripts/export_model_efficiency.py --output-root /mnt/c/mamba/artifacts/analysis/model_efficiency --warmup-iters 10 --benchmark-iters 40
"@

wsl -d Ubuntu-24.04 bash -lc $cmd
