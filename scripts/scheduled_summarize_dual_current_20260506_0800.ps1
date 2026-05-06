param(
    [string]$OutDir = "C:\mamba\artifacts\analysis\dual_current_visdrone_20260506_0800",
    [string]$LocalRun = "C:\mamba\artifacts\runs\local_visdrone2019_tinyvim_b_100e_stable_mem22_20260504_2233",
    [string]$FatLogRemote = "C:\Users\sshuser\codex_runs\hybrid-mamba\runs\fat_visdrone2019_mobilemamba_b1_100e_official_fix_20260505_1245\launcher.log"
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$logRoot = "C:\mamba\artifacts\scheduled"
New-Item -ItemType Directory -Force $logRoot | Out-Null
New-Item -ItemType Directory -Force $OutDir | Out-Null
$log = Join-Path $logRoot "summarize_dual_current_20260506_0800.log"

function Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message
    Add-Content -Path $log -Value $line
    Write-Output $line
}

Log "Scheduled dual current summary started."
& powershell -ExecutionPolicy Bypass -Command "& 'C:\Users\admin\.codex\skills\experiment-resource-report\scripts\report_resources.ps1' -Machines @('local','FatMachine') -ExperimentPattern 'tinyvim|mobilemamba|mamba|main.py|run.py|imagenet'" *> (Join-Path $OutDir "resource_snapshot.txt")

$localLog = Join-Path $LocalRun "train.log"
$fatLogLocal = Join-Path $OutDir "fat_mobilemamba_b1_current_launcher.log"
if (-not (Test-Path $localLog)) {
    Log ("ERROR: local log missing: {0}" -f $localLog)
    exit 1
}

Log "Fetching Fat current run log."
$remoteArg = "FatMachine:`"$FatLogRemote`""
scp $remoteArg $fatLogLocal *> (Join-Path $OutDir "scp_fat_log.txt")
if (-not (Test-Path $fatLogLocal)) {
    Log ("ERROR: failed to fetch Fat log: {0}" -f $FatLogRemote)
    exit 1
}

Log "Rendering summary tables and publication figures."
python C:\mamba\scripts\summarize_dual_current_visdrone.py --local-log $localLog --fat-log $fatLogLocal --out-dir $OutDir *> (Join-Path $OutDir "render.log")
if ($LASTEXITCODE -ne 0) {
    Log "ERROR: render script failed. See render.log."
    exit $LASTEXITCODE
}
Log ("Summary complete: {0}" -f $OutDir)
