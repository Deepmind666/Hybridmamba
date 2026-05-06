#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.mamba-env-cu128"
MICROMAMBA_BIN="${HOME}/.local/bin/micromamba"

run_python() {
  if [[ -x "${MICROMAMBA_BIN}" && -d "${ENV_PREFIX}" ]]; then
    "${MICROMAMBA_BIN}" run -p "${ENV_PREFIX}" python "$@"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
    return
  fi
  python "$@"
}

if [[ $# -eq 0 ]]; then
  run_python "${ROOT_DIR}/scripts/show_run_status.py"
else
  run_python "${ROOT_DIR}/scripts/show_run_status.py" --run-dir "$1"
fi
