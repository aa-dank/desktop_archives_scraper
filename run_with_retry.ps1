# Runs the scraper worker and automatically restarts on transient DB connection
# failures (exit code 3). Any other exit code stops the loop.
#
# Usage:
#   .\run_with_retry.ps1
#   .\run_with_retry.ps1 -MaxRestarts 10 -RetryDelaySec 60

param(
    [int]$MaxRestarts = 5,
    [int]$RetryDelaySec = 30
)

$RestartCount = 0
$WorkerArgs = @(
    "-m", "desktop_archives_scraper.cli",
    "--max-runtime-seconds", "28800",
    "--embed",
    "--randomize",
    "--failure_retry_treshold", "2",
    "--log-level", "INFO",
    "--json-logs"
)

while ($true) {
    python @WorkerArgs
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        Write-Host "Worker completed successfully."
        break
    }

    if ($ExitCode -eq 3) {
        $RestartCount++
        if ($RestartCount -gt $MaxRestarts) {
            Write-Host "Exceeded max restarts ($MaxRestarts). Stopping."
            break
        }
        Write-Host "Worker exited with transient code 3 (DB connection lost). Restart $RestartCount of $MaxRestarts in ${RetryDelaySec}s..."
        Start-Sleep -Seconds $RetryDelaySec
    } else {
        Write-Host "Worker exited with code $ExitCode. Stopping."
        break
    }
}
