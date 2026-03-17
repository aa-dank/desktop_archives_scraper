# ===========================================================================
# Orchestrator.ps1
#
# USAGE:
#   & "N:\PPDO\BS\Records Department\Tools\desktop_archives_scraper\development\deployment_scripts\Orchestrator.ps1"
#
# PURPOSE:
#   Entry point for running the Desktop Archives Scraper on any desktop.
#   This script lives on the shared file server alongside Run-App.ps1 and
#   run-params.env. Local machines only need a writable ProgramData folder
#   and Python installed -- everything else is handled here.
#
# WHAT IT DOES:
#   1. Pre-flight: validates that required files and tools are present.
#   2. Update-Code: pulls latest code from GitHub (no Git required),
#      creates/updates the local venv, and installs dependencies.
#   3. Run-App: loads run-params.env from this share folder + local .env,
#      then starts the scraper.
#
# UPDATING PARAMETERS:
#   Edit run-params.env (in the same folder as this script) to change how
#   the scraper behaves across all desktops. No script editing required.
#
# UPDATING THE APPLICATION:
#   Just run this script again. It will re-download and sync the latest
#   code from GitHub before running.
#
# NOTE ON LONG RUNS:
#   It is safe to update run-params.env or any script on this share while
#   the scraper is running on a desktop. PowerShell loads each script into
#   memory when it is first invoked; the running Python process is fully
#   independent and is not affected by file changes on the share.
# ===========================================================================

param(
    # Local path where app code, venv, and machine-specific .env are kept.
    # Uses the current user's AppData\Local folder so no admin rights are needed.
    # Each user who runs this script gets their own independent install.
    [string]$LocalRoot = "$env:LOCALAPPDATA\desktop_archives_scraper",

    # GitHub codeload URL for the branch to deploy (no Git required).
    # Change the branch name at the end to pin a specific branch.
    [string]$RepoZipUrl = "https://codeload.github.com/aa-dank/desktop_archives_scraper/zip/refs/heads/master",

    # Python version passed to the py.exe launcher when creating the venv.
    [string]$PythonVersion = "3.13"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# $PSScriptRoot resolves to the folder containing this script, wherever on
# the share it lives. All sibling scripts and config are found relative to it.
$ScriptRoot = $PSScriptRoot

$UpdateScript = Join-Path $ScriptRoot "Update-Code.ps1"
$RunScript    = Join-Path $ScriptRoot "Run-App.ps1"
$RunParams    = Join-Path $ScriptRoot "run-params.env"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function Write-Banner {
    param([string]$Message, [string]$Color = "Cyan")
    $line = "=" * 62
    Write-Host ""
    Write-Host $line -ForegroundColor $Color
    Write-Host "  $Message" -ForegroundColor $Color
    Write-Host $line -ForegroundColor $Color
    Write-Host ""
}

function Write-OK   { param([string]$Msg) Write-Host "  [OK]   $Msg" -ForegroundColor Green  }
function Write-Warn { param([string]$Msg) Write-Host "  [WARN] $Msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$Msg) Write-Host "  [FAIL] $Msg" -ForegroundColor Red    }
function Write-Info { param([string]$Msg) Write-Host "  [....] $Msg" -ForegroundColor Gray   }

function Write-Hint {
    # Prints an indented hint line in red to explain how to fix a failure.
    param([string]$Msg)
    Write-Host "         -> $Msg" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

Write-Banner "Desktop Archives Scraper"
Write-Info "Script root : $ScriptRoot"
Write-Info "Local root  : $LocalRoot"
Write-Info "Repo URL    : $RepoZipUrl"
Write-Host ""

Write-Banner "Pre-flight checks" "Yellow"
$ok = $true

# --- Sibling scripts ---
if (Test-Path $UpdateScript) {
    Write-OK "Update-Code.ps1 found"
} else {
    Write-Fail "Update-Code.ps1 not found: $UpdateScript"
    Write-Hint "The full deployment_scripts folder must be present on the share."
    $ok = $false
}

if (Test-Path $RunScript) {
    Write-OK "Run-App.ps1 found"
} else {
    Write-Fail "Run-App.ps1 not found: $RunScript"
    Write-Hint "The full deployment_scripts folder must be present on the share."
    $ok = $false
}

# --- Run parameters file ---
if (Test-Path $RunParams) {
    Write-OK "run-params.env found"
} else {
    Write-Fail "run-params.env not found: $RunParams"
    Write-Hint "Copy run-params.env.example (in the same folder) to run-params.env and configure it."
    $ok = $false
}

# --- Python ---
$pyExe = $null
try {
    # Prefer py.exe (Windows Python launcher) -- handles multiple installed versions cleanly.
    $null = Get-Command py -ErrorAction Stop
    $pyExe = "py"
    Write-OK "Python launcher (py.exe) found"
} catch {
    try {
        $null = Get-Command python -ErrorAction Stop
        $pyExe = "python"
        Write-OK "python found on PATH"
    } catch {
        Write-Fail "Python not found on this machine."
        Write-Hint "Install Python $PythonVersion via Chocolatey: choco install python"
        Write-Hint "Or download from: https://www.python.org/downloads/"
        $ok = $false
    }
}

# --- Local writable directory ---
try {
    New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
    $probe = Join-Path $LocalRoot ".write_probe"
    [IO.File]::WriteAllText($probe, "ok")
    Remove-Item $probe -Force
    Write-OK "Local app directory is writable: $LocalRoot"
} catch {
    Write-Fail "Cannot write to: $LocalRoot"
    Write-Hint "Try passing a path you own: -LocalRoot \"$env:LOCALAPPDATA\desktop_archives_scraper\""
    $ok = $false
}

# --- Internet / GitHub reachability ---
try {
    $probe = Invoke-WebRequest -Uri "https://github.com" -Method Head -TimeoutSec 10 -UseBasicParsing
    Write-OK "GitHub is reachable"
} catch {
    Write-Warn "GitHub reachability check failed -- code update may fail."
    Write-Hint "Check that this machine has internet access and no proxy is blocking github.com."
    # This is a warning, not a hard failure; the cached local copy may still be usable.
}

if (-not $ok) {
    Write-Banner "Pre-flight failed -- fix the issues above and try again." "Red"
    exit 1
}

Write-Banner "All checks passed -- starting" "Green"

# ---------------------------------------------------------------------------
# Step 1: Pull latest code from GitHub and update the local venv
# ---------------------------------------------------------------------------

Write-Banner "Step 1 of 2: Updating local code and dependencies" "Yellow"
try {
    & $UpdateScript -LocalRoot $LocalRoot -RepoZipUrl $RepoZipUrl -PythonVersion $PythonVersion
} catch {
    Write-Fail "Code update failed."
    Write-Hint "Error: $_"
    Write-Hint "Check internet connection and that $RepoZipUrl is accessible."
    Write-Hint "If the local code already exists, you can re-run without the update step."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 2: Run the application
# ---------------------------------------------------------------------------

Write-Banner "Step 2 of 2: Starting the scraper" "Yellow"
try {
    & $RunScript -LocalRoot $LocalRoot
} catch {
    Write-Fail "Application failed to start."
    Write-Hint "Error: $_"
    Write-Hint "Check the .env file at: $LocalRoot\app\.env"
    Write-Hint "Required keys: DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD, FILE_SERVER_MOUNT"
    Write-Hint "Check run-params.env in: $ScriptRoot"
    exit 1
}

Write-Banner "Finished" "Green"
