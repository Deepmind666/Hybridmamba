param(
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba"
)

$launcher = Join-Path $env:TEMP "hybrid_mamba_setup_blackwell.cmd"
$launcherContent = @"
@echo off
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/sshuser/codex_runs/hybrid-mamba && ./scripts/setup_wsl_env_blackwell.sh > /mnt/c/Users/sshuser/codex_runs/hybrid-mamba/setup_blackwell.log 2>&1"
"@
Set-Content -Path $launcher -Value $launcherContent -Encoding ASCII

scp $launcher "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/setup_blackwell.cmd"
ssh FatMachine "C:\Users\sshuser\codex_runs\hybrid-mamba\setup_blackwell.cmd"
