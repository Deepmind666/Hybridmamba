param(
    [switch]$UseGameTurbo,
    [switch]$StopDynamicTuning
)

$ErrorActionPreference = "Continue"

$gameTurboGuid = "97292658-7d19-4fc5-a0d7-d2c3bde28179"
$serviceNames = @(
    "ArmouryCrateService",
    "ROG Live Service",
    "asComSvc",
    "dptftcs",
    "ipfsvc"
)
$processNames = @(
    "ArmouryCrate",
    "ArmouryCrate.UserSessionHelper",
    "ArmourySocketServer",
    "ArmourySwAgent",
    "AiLoginServer",
    "ASUS DriverHub",
    "asus_framework"
)

if ($UseGameTurbo) {
    powercfg /SETACTIVE $gameTurboGuid | Out-Null
}

if ($StopDynamicTuning) {
    foreach ($serviceName in $serviceNames) {
        $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        if ($service -and $service.Status -eq "Running") {
            try {
                Stop-Service -Name $serviceName -Force -ErrorAction Stop
                Write-Host ("Stopped service    {0}" -f $serviceName)
            } catch {
                Write-Host ("Failed service     {0}" -f $serviceName)
            }
        }
    }

    $procs = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -in $processNames
    }
    foreach ($proc in $procs) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            Write-Host ("Stopped process    {0} ({1})" -f $proc.ProcessName, $proc.Id)
        } catch {
            Write-Host ("Failed process     {0} ({1})" -f $proc.ProcessName, $proc.Id)
        }
    }
}

if (-not $UseGameTurbo -and -not $StopDynamicTuning) {
    Write-Host "Power plan         unchanged"
    Write-Host "Dynamic tuning     unchanged"
    return
}

Write-Host ("Power plan         {0}" -f (powercfg /GETACTIVESCHEME))
