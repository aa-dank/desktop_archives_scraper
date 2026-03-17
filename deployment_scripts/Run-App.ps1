# ===========================================================================
# Run-App.ps1
#
# PURPOSE:
#   Starts the Desktop Archives Scraper using the local venv and code.
#   Called by Orchestrator.ps1 -- can also be run standalone if the local
#   app and venv are already in place.
#
# CONFIGURATION:
#   Two env files are loaded in order:
#     1. <LocalRoot>\app\.env   -- machine-specific secrets (DB creds, paths).
#                                  Never commit this file; it stays local.
#     2. run-params.env         -- runtime tuning (batch sizes, log level, etc).
#                                  Lives in the same folder as this script on
#                                  the share so ops can edit one file to affect
#                                  all desktops at once.
#   Values in run-params.env override same-named values from .env.
# ===========================================================================

param(
    # Root path where the local app code and venv live on this machine.
    # Must match the LocalRoot used in Orchestrator.ps1 / Update-Code.ps1.
    [string]$LocalRoot = "$env:LOCALAPPDATA\desktop_archives_scraper"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppDir    = Join-Path $LocalRoot "app"
$VenvPython = Join-Path $LocalRoot ".venv\Scripts\python.exe"

# run-params.env lives beside THIS script on the share.
# $PSScriptRoot resolves to the share folder where Run-App.ps1 is located.
$RunParamsPath = Join-Path $PSScriptRoot "run-params.env"

# Machine-specific secrets stay local and are never synced.
$DotEnvPath = Join-Path $AppDir ".env"

# ---------------------------------------------------------------------------
# Validate required local paths before doing anything
# ---------------------------------------------------------------------------

if (-not (Test-Path $AppDir)) {
    Write-Host "[run] ERROR: App directory not found: $AppDir" -ForegroundColor Red
    Write-Host "       -> Run Orchestrator.ps1 first to set up the local code." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[run] ERROR: Virtual environment not found: $VenvPython" -ForegroundColor Red
    Write-Host "       -> Run Orchestrator.ps1 (or Update-Code.ps1) to create the venv." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $DotEnvPath)) {
    Write-Host "[run] ERROR: Machine config file not found: $DotEnvPath" -ForegroundColor Red
    Write-Host "       -> Copy .env.example from the app folder to .env and fill in:" -ForegroundColor Red
    Write-Host "            DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD, FILE_SERVER_MOUNT" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Env file loader
# ---------------------------------------------------------------------------

function Import-EnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        Write-Host "[run] Env file not found, skipping: $Path" -ForegroundColor Yellow
        return
    }

    # Read KEY=VALUE pairs and apply them to the current process environment.
    # Env vars are the standard way the scraper CLI reads its configuration.
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()

        # Skip blank lines and comment lines (lines starting with #).
        if (-not $line -or $line.StartsWith("#")) { return }

        # Split only on the FIRST '=' so values containing '=' are handled correctly.
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            Write-Host "[run] Skipping malformed env line: $line" -ForegroundColor Yellow
            return
        }

        $name  = $parts[0].Trim()
        $value = $parts[1].Trim()

        if (-not $name) {
            Write-Host "[run] Skipping env line with empty key: $line" -ForegroundColor Yellow
            return
        }

        # Allow values to be wrapped in matching quotes in .env-style files.
        # Example supported forms:
        #   DB_PORT='5432'
        #   DB_HOST="10.132.65.32"
        if (
            ($value.Length -ge 2) -and
            (($value.StartsWith("'") -and $value.EndsWith("'")) -or
             ($value.StartsWith('"') -and $value.EndsWith('"')))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

# Load local secrets first (DB credentials, file server path, etc.).
Write-Host "[run] Loading machine config: $DotEnvPath"
Import-EnvFile -Path $DotEnvPath

# Load (and potentially override with) shared runtime parameters.
# Editing run-params.env on the share immediately affects all desktops
# from the next run -- no script changes needed.
Write-Host "[run] Loading run parameters: $RunParamsPath"
Import-EnvFile -Path $RunParamsPath

# ---------------------------------------------------------------------------
# Validate critical env vars are present before launching
# ---------------------------------------------------------------------------

$requiredVars = @("DB_HOST", "DB_PORT", "DB_NAME", "DB_USERNAME", "DB_PASSWORD", "FILE_SERVER_MOUNT")
$missingVars  = @($requiredVars | Where-Object { -not [System.Environment]::GetEnvironmentVariable($_, "Process") })

if ($missingVars.Count -gt 0) {
    Write-Host "[run] ERROR: Missing required environment variables:" -ForegroundColor Red
    $missingVars | ForEach-Object { Write-Host "         - $_" -ForegroundColor Red }
    Write-Host "       -> Add these to: $DotEnvPath" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

# Ask the operator how many hours the scraper should run this session.
# If they leave this blank, no runtime cap is applied.
$runtimeHoursInput = Read-Host "[run] Enter max runtime in hours (blank = no limit)"

if ($runtimeHoursInput -and $runtimeHoursInput.Trim()) {
    $hoursValue = 0.0
    $parsed = [double]::TryParse(
        $runtimeHoursInput.Trim(),
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$hoursValue
    )

    if (-not $parsed -or $hoursValue -le 0) {
        Write-Host "[run] ERROR: Runtime hours must be a positive number (example: 2 or 0.5)." -ForegroundColor Red
        exit 1
    }

    $maxRuntimeSeconds = [math]::Round($hoursValue * 3600, 3)
    [System.Environment]::SetEnvironmentVariable("MAX_RUNTIME_SECONDS", $maxRuntimeSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture), "Process")
    Write-Host "[run] MAX_RUNTIME_SECONDS set to $maxRuntimeSeconds seconds"
} else {
    # Clear any inherited value so blank input truly means no cap for this run.
    [System.Environment]::SetEnvironmentVariable("MAX_RUNTIME_SECONDS", $null, "Process")
    Write-Host "[run] No max runtime set; scraper will run continuously"
}

Set-Location $AppDir
Write-Host "[run] Starting desktop_archives_scraper ..."

# Invoke the CLI as a module so it picks up env vars set above.
# The CLI entry point in cli.py reads config from environment at startup.
& $VenvPython -m desktop_archives_scraper.cli
