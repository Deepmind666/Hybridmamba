# Environment

## Primary Local Environment

- Host OS: Windows
- Host hardware:
  - `Intel Core Ultra 9 285K`
  - `ASUS ROG MAXIMUS Z890 HERO`
  - BIOS `2006`
- Runtime: `WSL2 Ubuntu-24.04`
- GPU: RTX 5090
- Recommended environment bootstrap: `micromamba + Python 3.10`
- Recommended Python target: 3.10 for `TinyViM + MMDetection 2.28`
- Current state:
  - `torch 2.0.1+cu118`, `mmcv-full 1.7.2`, `mmdet 2.28.2`, `mmsegmentation 0.30.0`, `timm 0.5.4`, `einops` install successfully in WSL
  - `torch 2.0.1+cu118` cannot execute on RTX 5090 (`sm_120`)
  - see [runtime-blockers.md](/C:/mamba/docs/runtime-blockers.md)
  - local launch policy is documented in [local-machine-rules.md](/C:/mamba/docs/local-machine-rules.md)

## Blackwell-Compatible Parallel Environment

- Runtime env path: `C:\mamba\.mamba-env-cu128`
- Current validated stack:
  - `torch 2.7.1+cu128`
  - `mmengine 0.10.7`
  - `mmcv 2.2.0`
  - `mmdet 3.3.0` with a local compatibility patch for `mmcv 2.2.0`
  - `selective_scan_cuda_oflex`
- Canonical bootstrap:
  - [scripts/setup_wsl_env_blackwell.sh](/C:/mamba/scripts/setup_wsl_env_blackwell.sh)
- Canonical selective-scan install:
  - [scripts/install_selective_scan_blackwell.sh](/C:/mamba/scripts/install_selective_scan_blackwell.sh)

## Supplementary Remote Environment

- Host alias: `FatMachine`
- Runtime: `WSL2 Ubuntu-24.04`
- GPU: RTX 5090 D v2

## Policy

- Use Linux paths and commands for model build, training, and custom CUDA op compilation.
- Use PowerShell only for host orchestration, file staging, and remote invocation.
- Keep local and remote environments version-aligned.
- Use [scripts/setup_wsl_env.sh](/C:/mamba/scripts/setup_wsl_env.sh) as the canonical bootstrap entry.
- Run [scripts/check_local_host_stability.ps1](/C:/mamba/scripts/check_local_host_stability.ps1) before any local training.
- Use [scripts/start_local_training_blackwell_guarded.ps1](/C:/mamba/scripts/start_local_training_blackwell_guarded.ps1) as the only approved local Blackwell launcher.
- Use [scripts/start_recommended_training_blackwell.ps1](/C:/mamba/scripts/start_recommended_training_blackwell.ps1) as the preferred day-to-day launcher.
