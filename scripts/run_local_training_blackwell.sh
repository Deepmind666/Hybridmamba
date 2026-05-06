#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config> <work_dir>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$1"
WORK_DIR="$2"
shift 2
EXTRA_ARGS=("$@")
GPU_MEM_GB="${GPU_MEM_GB:-28}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"
TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-2}"
CPU_CORE_LIST="${CPU_CORE_LIST:-}"
IONICE_CLASS="${IONICE_CLASS:-2}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-}"
MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRAIN_ENTRY="${MAMBA_TRAIN_ENTRY:-scripts/run_train_mmdet3.py}"
ADAPTIVE_GUARD="${MAMBA_ADAPTIVE_GUARD:-0}"
GUARD_GPU_UTIL_PCT="${MAMBA_GUARD_GPU_UTIL_PCT:-80}"
GUARD_CPU_UTIL_PCT="${MAMBA_GUARD_CPU_UTIL_PCT:-80}"
GUARD_RESUME_UTIL_PCT="${MAMBA_GUARD_RESUME_UTIL_PCT:-70}"
GUARD_TEMP_C="${MAMBA_GUARD_TEMP_C:-78}"
GUARD_MEMORY_PCT="${MAMBA_GUARD_MEMORY_PCT:-92}"
GUARD_CHECK_INTERVAL_SEC="${MAMBA_GUARD_CHECK_INTERVAL_SEC:-2}"
GUARD_COOLDOWN_SEC="${MAMBA_GUARD_COOLDOWN_SEC:-20}"

cmd=()

if command -v ionice >/dev/null 2>&1; then
  cmd+=(ionice -c "${IONICE_CLASS}" -n "${IONICE_LEVEL}")
fi

if [[ -n "${CPU_CORE_LIST}" ]] && command -v taskset >/dev/null 2>&1; then
  cmd+=(taskset -c "${CPU_CORE_LIST}")
fi

cmd+=(~/.local/bin/micromamba run -p "${ROOT_DIR}/.mamba-env-cu128")

if [[ -n "${CUDA_DEVICE_MAX_CONNECTIONS}" ]]; then
  cmd+=(env "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}")
fi

cmd+=(env "MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}")
cmd+=(env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}")
cmd+=(python "${ROOT_DIR}/${TRAIN_ENTRY}")
cmd+=("${CONFIG_PATH}")
cmd+=(--work-dir "${WORK_DIR}")
cmd+=(--gpu-mem-gb "${GPU_MEM_GB}")
cmd+=(--torch-num-threads "${TORCH_NUM_THREADS}")
cmd+=(--torch-num-interop-threads "${TORCH_NUM_INTEROP_THREADS}")
if [[ "${ADAPTIVE_GUARD}" == "1" || "${ADAPTIVE_GUARD,,}" == "true" ]]; then
  cmd+=(--adaptive-guard)
  cmd+=(--guard-gpu-util-pct "${GUARD_GPU_UTIL_PCT}")
  cmd+=(--guard-cpu-util-pct "${GUARD_CPU_UTIL_PCT}")
  cmd+=(--guard-resume-util-pct "${GUARD_RESUME_UTIL_PCT}")
  cmd+=(--guard-temp-c "${GUARD_TEMP_C}")
  cmd+=(--guard-memory-pct "${GUARD_MEMORY_PCT}")
  cmd+=(--guard-check-interval-sec "${GUARD_CHECK_INTERVAL_SEC}")
  cmd+=(--guard-cooldown-sec "${GUARD_COOLDOWN_SEC}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

"${cmd[@]}"
