param(
    [string]$Config = "configs/autoresearch_hybrid.yaml",
    [string]$TargetModel = "gcn_fusion",
    [int]$MaxIterations = 100,
    [double]$SleepSeconds = 1
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $PSScriptRoot "autoresearch"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$stdout = Join-Path $stateDir "autoresearch_stdout.log"
$stderr = Join-Path $stateDir "autoresearch_stderr.log"
$pidFile = Join-Path $stateDir "autoresearch.pid"
$command = "Set-Location '$projectRoot'; python research\run_autoresearch.py --config $Config --target-model $TargetModel --continuous --sleep-seconds $SleepSeconds --max-iterations $MaxIterations"

$process = Start-Process -FilePath powershell.exe `
    -ArgumentList "-NoProfile", "-Command", $command `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Set-Content -Path $pidFile -Value $process.Id
Write-Output "Started autoresearch with PID $($process.Id)"
