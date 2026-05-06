#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config> <work_dir> [extra train.py args...]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$1"
WORK_DIR="$2"
shift 2

if [[ -x "${HOME}/.local/bin/micromamba" && -d "${ROOT_DIR}/.mamba-env" ]]; then
  eval "$("${HOME}/.local/bin/micromamba" shell hook --shell bash)"
  micromamba activate "${ROOT_DIR}/.mamba-env"
elif [[ -f "${ROOT_DIR}/.venv-wsl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv-wsl/bin/activate"
fi

mkdir -p "${WORK_DIR}"
cd "${ROOT_DIR}/code/tinyvim/detection"

python train.py "${CONFIG_PATH}" --work-dir "${WORK_DIR}" --deterministic "$@"
