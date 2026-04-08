param(
    [string]$PidFile = "",
    [string]$StateDir = "",
    [string]$Title = "Autoresearch Finished"
)

if ([string]::IsNullOrWhiteSpace($PidFile)) {
    $PidFile = Join-Path (Join-Path $PSScriptRoot "autoresearch") "autoresearch.pid"
}
if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $PSScriptRoot "autoresearch"
}

if (-not (Test-Path $PidFile)) {
    exit 1
}

$pidValue = Get-Content $PidFile | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($pidValue)) {
    exit 1
}

$targetPid = [int]$pidValue
while ($true) {
    $process = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        break
    }
    Start-Sleep -Seconds 10
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$summaryPath = Join-Path $StateDir "latest_summary.md"
$flagPath = Join-Path $StateDir "finished.flag"
$message = "Autoresearch completed at $timestamp.`n`nState dir: $StateDir"
if (Test-Path $summaryPath) {
    $message = $message + "`n`nLatest summary:`n" + (Get-Content $summaryPath -Raw)
}

Set-Content -Path $flagPath -Value $message

Add-Type -AssemblyName PresentationFramework
[System.Windows.MessageBox]::Show($message, $Title) | Out-Null
