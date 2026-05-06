param(
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$ArchiveName = "hybrid_mamba_blackwell_sync.tar.gz"
)

$localRoot = "C:\mamba"
$archiveWin = Join-Path $localRoot "artifacts\tmp_validation\$ArchiveName"
$archiveWsl = "/mnt/c/mamba/artifacts/tmp_validation/$ArchiveName"
$remoteArchive = "/C:/Users/sshuser/codex_runs/hybrid-mamba/$ArchiveName"
$remoteRootUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba"

New-Item -ItemType Directory -Force (Split-Path $archiveWin -Parent) | Out-Null

$tarCommand = @"
cd /mnt/c/mamba
rm -f $archiveWsl
tar -czf $archiveWsl \
  README.md \
  .gitignore \
  .claude \
  .codex \
  code \
  docs \
  scripts \
  weights/tinyvim/tinyvim_b_300e.pth \
  data/visdrone/train \
  data/visdrone/val \
  data/converted/visdrone \
  data/converted/aitodv2 \
  data/converted/dota_hbb
"@

wsl -d Ubuntu-24.04 bash -lc $tarCommand

ssh FatMachine "cmd /c if not exist `"$RemoteRoot`" mkdir `"$RemoteRoot`"" | Out-Null
scp $archiveWin "FatMachine:$remoteArchive"

$extractCmdLocal = Join-Path $env:TEMP "hybrid_mamba_extract_blackwell.cmd"
$extractCmdRemote = Join-Path $RemoteRoot "extract_$ArchiveName.cmd"
$extractContent = @"
@echo off
wsl -d Ubuntu-24.04 bash -lc "mkdir -p $remoteRootUnix && tar -xzf $remoteRootUnix/$ArchiveName -C $remoteRootUnix"
"@
Set-Content -Path $extractCmdLocal -Value $extractContent -Encoding ASCII

scp $extractCmdLocal "FatMachine:/C:/Users/sshuser/codex_runs/hybrid-mamba/extract_$ArchiveName.cmd"
ssh FatMachine "$extractCmdRemote"
