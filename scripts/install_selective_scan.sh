#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.mamba-env"
MICROMAMBA_BIN="${HOME}/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.micromamba}"
VMAMBA_DIR="${ROOT_DIR}/external/vmamba"
SETUP_PY="${VMAMBA_DIR}/kernels/selective_scan/setup.py"

if [[ -x "${MICROMAMBA_BIN}" && -d "${ENV_PREFIX}" ]]; then
  mkdir -p "${MAMBA_ROOT_PREFIX}"
  eval "$("${MICROMAMBA_BIN}" shell hook --shell bash)"
  micromamba activate "${ENV_PREFIX}"
fi

TORCH_CUDA_VERSION="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"

SYSTEM_CUDA_VERSION=""
CUDA_HOME_PATH="${CUDA_HOME:-/usr/local/cuda}"
if [[ -f "${CUDA_HOME_PATH}/version.json" ]]; then
  SYSTEM_CUDA_VERSION="$(CUDA_HOME_PATH="${CUDA_HOME_PATH}" python - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ['CUDA_HOME_PATH']) / 'version.json'
data = json.loads(path.read_text())
print(data.get('cuda', {}).get('version', ''))
PY
)"
elif [[ -f "${CUDA_HOME_PATH}/version.txt" ]]; then
  SYSTEM_CUDA_VERSION="$(sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' "${CUDA_HOME_PATH}/version.txt" | head -n 1)"
fi

if command -v nvcc >/dev/null 2>&1; then
  SYSTEM_CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
fi

if [[ -n "${SYSTEM_CUDA_VERSION}" && -n "${TORCH_CUDA_VERSION}" && "${SYSTEM_CUDA_VERSION}" != "${TORCH_CUDA_VERSION}" ]]; then
  echo "CUDA toolkit / PyTorch mismatch: system CUDA=${SYSTEM_CUDA_VERSION}, torch CUDA=${TORCH_CUDA_VERSION}" >&2
  echo "Current TinyViM reference stack cannot build selective_scan on this host without resolving the CUDA version mismatch." >&2
  exit 2
fi

if [[ ! -d "${VMAMBA_DIR}" ]]; then
  git clone --depth 1 https://github.com/MzeroMiko/VMamba.git "${VMAMBA_DIR}"
fi

if [[ ! -f "${SETUP_PY}" ]]; then
  echo "selective_scan setup.py not found at ${SETUP_PY}" >&2
  exit 1
fi

SETUP_PY_PATH="${SETUP_PY}" python - <<'PY'
from pathlib import Path
import os

setup_path = Path(os.environ["SETUP_PY_PATH"])
text = setup_path.read_text(encoding='utf-8')
needle = 'arch=compute_90,code=sm_90'
insert = '\n    cc_flag.append("-gencode")\n    cc_flag.append("arch=compute_120,code=sm_120")'
if 'arch=compute_120,code=sm_120' not in text and needle in text:
    text = text.replace(needle, needle + insert, 1)
    setup_path.write_text(text, encoding='utf-8')
PY

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
cd "${VMAMBA_DIR}/kernels/selective_scan"
python -m pip install --upgrade --no-cache-dir --no-build-isolation .
