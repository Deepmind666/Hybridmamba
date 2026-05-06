#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.mamba-env"
MICROMAMBA_BIN="${HOME}/.local/bin/micromamba"
MICROMAMBA_TARBALL="/tmp/micromamba-linux-64.tar.bz2"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.micromamba}"

mkdir -p "${HOME}/.local/bin"
mkdir -p "${MAMBA_ROOT_PREFIX}"

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

python -m pip install --upgrade pip wheel "setuptools<81" "numpy<2"
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install timm==0.5.4 einops pillow pycocotools tabulate gdown
python -m pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
python -m pip install mmdet==2.28.2 mmsegmentation==0.30.0

cat <<'EOF'

Environment bootstrap finished.

Next manual steps:
1. Activate the environment:
   eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
   micromamba activate /mnt/c/mamba/.mamba-env
2. Build or install selective_scan_cuda with:
   bash scripts/install_selective_scan.sh
3. Place TinyViM checkpoints under:
   weights/tinyvim/

EOF
