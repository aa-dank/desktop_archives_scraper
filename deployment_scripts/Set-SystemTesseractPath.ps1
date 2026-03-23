# ===========================================================================
# Set-SystemTesseractPath.ps1
#
# PURPOSE:
#   Configure Tesseract for all users on this Windows machine.
#
# WHAT IT DOES:
#   1. Requires an elevated (Administrator) PowerShell session.
#   2. Validates that tesseract.exe exists in the provided install folder.
#   3. Adds the install folder to the machine-level PATH if not already present.
#   4. Optionally sets machine-level TESSERACT_CMD to full exe path.
#   5. Verifies direct invocation of tesseract.exe.
#
# USAGE (run as Administrator):
#   powershell -ExecutionPolicy Bypass -File .\Set-SystemTesseractPath.ps1
#   powershell -ExecutionPolicy Bypass -File .\Set-SystemTesseractPath.ps1 -TesseractDir "D:\Tools\Tesseract-OCR"
#   powershell -ExecutionPolicy Bypass -File .\Set-SystemTesseractPath.ps1 -SetMachineTesseractCmd:$false
# ===========================================================================

param(
    [string]$TesseractDir = "C:\Program Files\Tesseract-OCR",
    [switch]$SetMachineTesseractCmd = $true
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-OK {
    param([string]$Message)
    Write-Host "[OK]   $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

# Require elevation for machine-level env writes.
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Fail "Run this script in an elevated PowerShell session (Administrator)."
    exit 1
}

$tesseractExe = Join-Path $TesseractDir "tesseract.exe"
if (-not (Test-Path -LiteralPath $tesseractExe)) {
    Write-Fail "tesseract.exe not found at: $tesseractExe"
    Write-Warn "If installed elsewhere, rerun with -TesseractDir <actual folder>."
    exit 1
}
Write-OK "Found: $tesseractExe"

$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$machineEntries = @()

if (-not [string]::IsNullOrWhiteSpace($machinePath)) {
    $machineEntries = @(
        $machinePath -split ";" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -ne "" }
    )
}

$targetNorm = $TesseractDir.Trim().TrimEnd("\\")
$alreadyPresent = $false

foreach ($entry in $machineEntries) {
    $entryNorm = $entry.TrimEnd("\\")
    if ([string]::Equals($entryNorm, $targetNorm, [System.StringComparison]::OrdinalIgnoreCase)) {
        $alreadyPresent = $true
        break
    }
}

if ($alreadyPresent) {
    Write-OK "System PATH already contains: $TesseractDir"
} else {
    $newEntries = @($machineEntries + $TesseractDir)
    $newMachinePath = ($newEntries -join ";")
    [Environment]::SetEnvironmentVariable("Path", $newMachinePath, "Machine")
    Write-OK "Added to System PATH: $TesseractDir"
}

if ($SetMachineTesseractCmd) {
    [Environment]::SetEnvironmentVariable("TESSERACT_CMD", $tesseractExe, "Machine")
    Write-OK "Set machine env var TESSERACT_CMD=$tesseractExe"
}

try {
    $versionLine = & $tesseractExe --version | Select-Object -First 1
    if ($versionLine) {
        Write-Info $versionLine
    }
    Write-OK "Direct invocation succeeded."
} catch {
    Write-Fail "Direct invocation failed: $($_.Exception.Message)"
    exit 1
}

Write-Warn "Open a NEW terminal session (or sign out/in) for PATH changes to apply to new processes."
Write-OK "Done."
