param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$RemoteRoot = "C:\Users\sshuser\codex_runs\hybrid-mamba",
    [string]$ExtraArgs = ""
)

$localRoot = "C:\mamba"
$remoteWorkDir = Join-Path $RemoteRoot "artifacts\runs\$RunId"
$remoteDetectionDir = Join-Path $RemoteRoot "code\tinyvim\detection"
$remoteRepoUnix = "/mnt/c/Users/sshuser/codex_runs/hybrid-mamba"
$remoteWorkUnix = "$remoteRepoUnix/artifacts/runs/$RunId"
$relativeConfig = Resolve-Path $ConfigPath | ForEach-Object { $_.Path.Replace($localRoot, "").TrimStart("\") }
$remoteConfig = "$remoteRepoUnix/" + ($relativeConfig.Replace("\", "/"))
$remoteExtraArgs = $ExtraArgs.Replace("\", "/")

$remoteScript = @"
if (-not (Test-Path '$remoteWorkDir')) { New-Item -ItemType Directory -Force '$remoteWorkDir' | Out-Null }
if (-not (Test-Path '$remoteDetectionDir')) { throw 'Remote detection directory missing. Sync repo first.' }
wsl -d Ubuntu-24.04 bash -lc 'if [ -x $remoteRepoUnix/.mamba-env/bin/python ]; then export PATH=$remoteRepoUnix/.mamba-env/bin:$PATH; fi; cd $remoteRepoUnix/code/tinyvim/detection && python train.py $remoteConfig --work-dir $remoteWorkUnix --deterministic $remoteExtraArgs'
"@

$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
ssh FatMachine "powershell -NoProfile -EncodedCommand $encodedCommand"
