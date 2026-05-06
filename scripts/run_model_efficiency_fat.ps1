$ErrorActionPreference = 'Stop'

$remoteScript = @'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
& wsl -d Ubuntu-24.04 -u ns3user -- bash -lc "cd /mnt/c/Users/sshuser/codex_runs/hybrid-mamba && /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/tools/micromamba run -p /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/.mamba-env-cu128 python scripts/export_model_efficiency.py --output-root /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/artifacts/analysis/model_efficiency --warmup-iters 10 --benchmark-iters 40"
'@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encoded"
