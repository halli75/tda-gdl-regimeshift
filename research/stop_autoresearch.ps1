$pidFile = Join-Path (Join-Path $PSScriptRoot "autoresearch") "autoresearch.pid"

if (-not (Test-Path $pidFile)) {
    Write-Output "No PID file found."
    exit 0
}

$pidValue = Get-Content $pidFile | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($pidValue)) {
    Write-Output "PID file is empty."
    exit 0
}

$process = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Write-Output "Process $pidValue is not running."
    exit 0
}

Stop-Process -Id $process.Id -Force
Write-Output "Stopped autoresearch process $($process.Id)"
