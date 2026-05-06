param(
    [string]$ChunkDir = "C:\mamba\artifacts\tmp\tinyvim_chunks",
    [string]$RemoteDir = "/C:/Users/sshuser/codex_runs/hybrid-mamba/weights/tinyvim/parts",
    [int]$StartFromIndex = 0,
    [int]$MaxRetriesPerChunk = 6
)

$ErrorActionPreference = "Stop"
$parts = Get-ChildItem -Path $ChunkDir -Filter "part_*.bin" | Sort-Object Name
if (-not $parts) {
    throw "No chunk files found in $ChunkDir"
}

foreach ($p in $parts) {
    $idx = [int]($p.BaseName.Split("_")[-1])
    if ($idx -lt $StartFromIndex) {
        continue
    }
    $uploaded = $false
    for ($attempt = 1; $attempt -le $MaxRetriesPerChunk; $attempt++) {
        Write-Host ("Uploading {0} (attempt {1}/{2})" -f $p.Name, $attempt, $MaxRetriesPerChunk)
        & scp $p.FullName ("FatMachine:{0}/{1}" -f $RemoteDir, $p.Name)
        if ($LASTEXITCODE -eq 0) {
            $uploaded = $true
            break
        }
        Start-Sleep -Seconds 2
    }
    if (-not $uploaded) {
        throw ("SCP failed at {0} after {1} retries" -f $p.Name, $MaxRetriesPerChunk)
    }
}

Write-Host "All chunks uploaded."
