Get-WinEvent -FilterHashtable @{
    LogName = 'System'
    ID = 41, 6008, 1074, 1076, 6005, 6006
    StartTime = (Get-Date).AddHours(-12)
} | Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, Message | Sort-Object TimeCreated -Descending | Select-Object -First 20
