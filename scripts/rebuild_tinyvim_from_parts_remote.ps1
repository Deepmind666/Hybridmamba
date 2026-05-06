$partsDir = "C:\Users\sshuser\codex_runs\hybrid-mamba\weights\tinyvim\parts2"
$outPath = "C:\Users\sshuser\codex_runs\hybrid-mamba\weights\tinyvim\tinyvim_b_300e.pth"

$parts = Get-ChildItem -Path $partsDir -Filter "part_*.bin" | Sort-Object Name
if (-not $parts) {
    throw "No parts found in $partsDir"
}

if (Test-Path $outPath) {
    Remove-Item -Path $outPath -Force
}

$stream = [System.IO.File]::Open($outPath, [System.IO.FileMode]::CreateNew)
try {
    foreach ($p in $parts) {
        $bytes = [System.IO.File]::ReadAllBytes($p.FullName)
        $stream.Write($bytes, 0, $bytes.Length)
    }
} finally {
    $stream.Close()
}

Write-Host ("parts={0}" -f $parts.Count)
Write-Host ("bytes={0}" -f (Get-Item $outPath).Length)
