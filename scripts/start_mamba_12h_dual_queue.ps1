param(
    [int]$Hours = 12,
    [string]$CurrentLocalRunId = "local_tinyvim1x_stable_resume_e14_mem18_20260429_000956"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\mamba"
$queueRoot = Join-Path $repoRoot "artifacts\queues"
New-Item -ItemType Directory -Force $queueRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$localQueue = "mamba12h_local_$stamp"
$fatQueue = "mamba12h_fat_$stamp"

$localScript = Join-Path $repoRoot "scripts\run_mamba_12h_local_queue.ps1"
$fatScript = Join-Path $repoRoot "scripts\run_mamba_12h_fat_queue.ps1"

Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $localScript,
    "-Hours", $Hours,
    "-CurrentRunId", $CurrentLocalRunId,
    "-QueueName", $localQueue
) -WindowStyle Hidden

Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $fatScript,
    "-Hours", $Hours,
    "-QueueName", $fatQueue
) -WindowStyle Hidden

Write-Host ("Started local queue: {0}" -f $localQueue)
Write-Host ("Local queue log: {0}" -f (Join-Path $queueRoot "$localQueue\queue.log"))
Write-Host ("Started Fat queue: {0}" -f $fatQueue)
Write-Host ("Fat queue log: {0}" -f (Join-Path $queueRoot "$fatQueue\queue.log"))
Write-Host ("Deadline hours: {0}" -f $Hours)
