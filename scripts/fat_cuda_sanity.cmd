@echo off
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/sshuser/codex_runs/hybrid-mamba && ~/.local/bin/micromamba run -p /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/.mamba-env-cu128 python scripts/cuda_sanity_check.py"
