#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.mamba-env-cu128"
MICROMAMBA_BIN="${HOME}/.local/bin/micromamba"
MICROMAMBA_TARBALL="/tmp/micromamba-linux-64.tar.bz2"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.micromamba}"

mkdir -p "${HOME}/.local/bin" "${MAMBA_ROOT_PREFIX}"

download_micromamba() {
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 3 -o "${MICROMAMBA_TARBALL}" https://micro.mamba.pm/api/micromamba/linux-64/latest && return 0
  fi
  python3 - <<'PY'
import urllib.request
url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
target = "/tmp/micromamba-linux-64.tar.bz2"
with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as handle:
    handle.write(response.read())
print(target)
PY
}

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  download_micromamba
  tar -xjf "${MICROMAMBA_TARBALL}" -C /tmp bin/micromamba
  mv /tmp/bin/micromamba "${MICROMAMBA_BIN}"
fi

eval "$("${MICROMAMBA_BIN}" shell hook --shell bash)"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  "${MICROMAMBA_BIN}" create -y -p "${ENV_PREFIX}" python=3.10 pip
fi

micromamba activate "${ENV_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

python -m pip install --upgrade pip wheel "setuptools<81" "numpy<2"
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install timm==1.0.20 einops pillow pycocotools tabulate gdown openmim
python -m pip install mmengine==0.10.7
MMCV_WITH_OPS=1 python -m pip install --no-cache-dir --force-reinstall --no-build-isolation mmcv==2.2.0
python -m pip install mmdet==3.3.0
python "${ROOT_DIR}/scripts/patch_mmdet_mmcv22.py" --site-packages "${ENV_PREFIX}/lib/python3.10/site-packages"

cat <<'EOF'

Blackwell-compatible environment bootstrap finished.

Next manual steps:
1. Activate the environment:
   eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
   micromamba activate /mnt/c/mamba/.mamba-env-cu128
2. Build or install selective_scan with:
   bash scripts/install_selective_scan_blackwell.sh
3. Run runtime validation with:
   python scripts/check_runtime_stack.py

EOF
