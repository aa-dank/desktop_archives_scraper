# ===========================================================================
# Update-Code.ps1
#
# PURPOSE:
#   Downloads the latest application code from GitHub (no Git needed),
#   mirrors it into the local machine's app folder, creates the shared venv
#   if it does not exist, and ensures all Python dependencies are installed.
#
#   Safe to re-run at any time (idempotent). Existing local state is preserved:
#     - .venv is never deleted or recreated if it already exists.
#     - .env (machine secrets) is never overwritten.
#     - No user data or output folders are touched.
#
# CALLED BY:
#   Orchestrator.ps1 (Step 1). Can also be run standalone to force a refresh.
# ===========================================================================

param(
    # Root location for per-user application state.
    # AppData\Local is always writable by the logged-in user with no admin rights.
    [string]$LocalRoot = "$env:LOCALAPPDATA\desktop_archives_scraper",

    # GitHub codeload zip endpoint. No Git or PAT token required for public repos.
    [string]$RepoZipUrl = "https://codeload.github.com/aa-dank/desktop_archives_scraper/zip/refs/heads/master",

    # Python version to use when creating the venv via py.exe launcher.
    [string]$PythonVersion = "3.13"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Core folder layout.
$AppDir = Join-Path $LocalRoot "app"
$VenvDir = Join-Path $LocalRoot ".venv"

# Optional seed venv location. If present/readable, this is copied to speed up
# first-time setup on each desktop user profile.
$SeedVenvDir = "C:\ProgramData\desktop_archives_scraper\.venv"

# Temporary working paths for zip download and extraction.
$ZipPath = Join-Path $env:TEMP "desktop_archives_scraper-master.zip"
$ExtractRoot = Join-Path $env:TEMP "desktop_archives_scraper_extract"

# Ensure target folders exist before any copy/install operation.
New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null

Write-Host "[update] Downloading repository snapshot from GitHub"
try {
    Invoke-WebRequest -Uri $RepoZipUrl -OutFile $ZipPath -UseBasicParsing
} catch {
    Write-Host "[update] ERROR: Could not download from: $RepoZipUrl" -ForegroundColor Red
    Write-Host "         -> Check internet connectivity on this machine." -ForegroundColor Red
    Write-Host "         -> Error detail: $_" -ForegroundColor Red
    exit 1
}

# Clean extraction folder from prior runs to avoid stale files mixing in.
if (Test-Path $ExtractRoot) {
    Remove-Item $ExtractRoot -Recurse -Force
}

Write-Host "[update] Expanding repository archive"
try {
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractRoot -Force
} catch {
    Write-Host "[update] ERROR: Failed to extract zip archive from $ZipPath" -ForegroundColor Red
    Write-Host "         -> The downloaded file may be corrupt. Delete it and retry:" -ForegroundColor Red
    Write-Host "            Remove-Item '$ZipPath' -Force" -ForegroundColor Red
    exit 1
}

# codeload zips usually extract to a single top-level folder like repo-branch.
$ExtractedRepo = Get-ChildItem -Path $ExtractRoot -Directory | Select-Object -First 1
if (-not $ExtractedRepo) {
    throw "Failed to locate extracted repository folder in: $ExtractRoot"
}

Write-Host "[update] Mirroring code into local app directory"
# Idempotent sync strategy:
# - /MIR keeps destination aligned to source (add/update/delete)
# - preserve local runtime state (.venv/.env/log/output/data) via exclusions
# - excludes pycache/test cache to reduce churn
$RoboArgs = @(
    $ExtractedRepo.FullName,
    $AppDir,
    "/MIR",
    "/XD", ".venv", ".git", "__pycache__", ".pytest_cache", "logs", "output", "data",
    "/XF", ".env"
)

robocopy @RoboArgs | Out-Null
$RoboExitCode = $LASTEXITCODE

# robocopy exit codes 0-7 are considered success states.
if ($RoboExitCode -gt 7) {
    throw "robocopy failed with exit code $RoboExitCode"
}

# Create venv only once; re-running this script safely reuses the existing one.
if (-not (Test-Path $VenvDir)) {
    $seedCopied = $false

    if (Test-Path $SeedVenvDir) {
        Write-Host "[update] Seeding venv from: $SeedVenvDir"
        try {
            # Copy the seed venv as a fast-start baseline, then pip install below
            # to normalize scripts/metadata for this user path.
            robocopy $SeedVenvDir $VenvDir /E | Out-Null
            $seedCopyExitCode = $LASTEXITCODE
            if ($seedCopyExitCode -le 7) {
                $seedVenvPython = Join-Path $VenvDir "Scripts\python.exe"
                if (Test-Path $seedVenvPython) {
                    $seedCopied = $true
                    Write-Host "[update] Seed venv copy completed"
                }
            }
        } catch {
            Write-Host "[update] WARNING: Failed to copy seed venv. Falling back to fresh venv creation." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[update] Seed venv not found at $SeedVenvDir; creating a new venv"
    }

    if (-not $seedCopied) {
        if (Test-Path $VenvDir) {
            Remove-Item $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        Write-Host "[update] Creating local virtual environment at: $VenvDir"
        try {
            # py.exe launcher handles multiple installed Python versions cleanly.
            $pyCmd = Get-Command py -ErrorAction SilentlyContinue
            if ($pyCmd) {
                & py "-$PythonVersion" -m venv $VenvDir
            } else {
                # Fallback: use whatever python is on PATH.
                & python -m venv $VenvDir
            }
        } catch {
            Write-Host "[update] ERROR: Failed to create virtual environment." -ForegroundColor Red
            Write-Host "         -> Ensure Python $PythonVersion is installed: choco install python" -ForegroundColor Red
            Write-Host "         -> Error detail: $_" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Write-Host "[update] Reusing existing virtual environment at: $VenvDir"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python was not found at: $VenvPython"
}

Write-Host "[update] Upgrading pip"
& $VenvPython -m pip install --upgrade pip

$RequirementsPath = Join-Path $AppDir "requirements.txt"

if (Test-Path $RequirementsPath) {
    Write-Host "[update] Installing Python dependencies from requirements.txt"
    Write-Host "         (This may take several minutes on first run due to torch/transformers)"
    try {
        & $VenvPython -m pip install -r $RequirementsPath
    } catch {
        Write-Host "[update] ERROR: pip install failed." -ForegroundColor Red
        Write-Host "         -> Error detail: $_" -ForegroundColor Red
        Write-Host "         -> If a package fails to build, ensure Visual C++ Build Tools are installed." -ForegroundColor Red
        exit 1
    }
} else {
    # Fallback: install the package directly from its folder (uses pyproject.toml).
    Write-Host "[update] requirements.txt not found; installing package from source folder"
    & $VenvPython -m pip install $AppDir
}

# Seed .env only on first-time setup. Never overwrite an existing .env because
# it holds machine-specific secrets (DB credentials, file server path).
$DotEnv = Join-Path $AppDir ".env"
$DotEnvExample = Join-Path $AppDir ".env.example"
if (-not (Test-Path $DotEnv) -and (Test-Path $DotEnvExample)) {
    Copy-Item $DotEnvExample $DotEnv
    Write-Host "[update] Created .env from .env.example"
    Write-Host "         -> IMPORTANT: Edit $DotEnv and fill in:" -ForegroundColor Yellow
    Write-Host "            DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD, FILE_SERVER_MOUNT" -ForegroundColor Yellow
} elseif (-not (Test-Path $DotEnv)) {
    Write-Host "[update] WARNING: No .env file found and no .env.example to copy from." -ForegroundColor Yellow
    Write-Host "         -> Manually create: $DotEnv" -ForegroundColor Yellow
    Write-Host "            with keys: DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD, FILE_SERVER_MOUNT" -ForegroundColor Yellow
}

Write-Host "[update] Completed successfully"
