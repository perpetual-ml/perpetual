$ErrorActionPreference = "Stop"

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

$RLib = Join-Path $projectRoot "package-r\R_library"
if (-not (Test-Path $RLib)) {
    New-Item -ItemType Directory -Force -Path $RLib
}

$env:R_LIBS_USER = $RLib
$env:R_LIBS_SITE = $RLib

# Try to find Rscript if not in path
if (-not (Get-Command "Rscript" -ErrorAction SilentlyContinue)) {
    Write-Host "Rscript not found in PATH, searching in C:\Program Files\R..." -ForegroundColor Yellow
    $RPath = Get-ChildItem -Path "C:\Program Files\R" -Recurse -Filter "Rscript.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    if ($RPath) {
        $global:RBinDir = Split-Path -Parent $RPath
        $env:PATH = "$global:RBinDir;$env:PATH"
        Write-Host "Found R at $global:RBinDir, added to PATH." -ForegroundColor Gray
    } else {
        Write-Error "Rscript.exe not found. Please install R and add it to your PATH."
    }
} else {
    $global:RBinDir = Split-Path -Parent (Get-Command "Rscript.exe").Source
}

$RscriptExe = Join-Path $global:RBinDir "Rscript.exe"
$Rexe = Join-Path $global:RBinDir "R.exe"
$env:PERPETUAL_ROOT = $projectRoot.Replace('\', '/')

Write-Host "Setting up R dependencies..." -ForegroundColor Cyan
& $RscriptExe scripts/setup_r_deps.R
Write-Host "Vendoring Rust dependencies and syncing core..." -ForegroundColor Cyan
python scripts/vendor_r.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Vendor script failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Remove potential stale Cargo.lock in package-r/src/rust to allow fresh resolution matching vendored deps
$RustDir = Join-Path $projectRoot "package-r\src\rust"
$StaleLock = Join-Path $RustDir "Cargo.lock"
if (Test-Path $StaleLock) {
    Write-Host "Removing stale $StaleLock..."
    Remove-Item -Path $StaleLock -Force
}

Write-Host "Building and installing perpetual package..." -ForegroundColor Cyan
$RLibUnix = $RLib.Replace('\', '/')

$LockDir = Join-Path $RLib "00LOCK-perpetual"
if (Test-Path $LockDir) {
    Write-Host "Removing stale lock directory..." -ForegroundColor Yellow
    Remove-Item -Path $LockDir -Recurse -Force
}
$LockDir2 = Join-Path $RLib "00LOCK-package-r"
if (Test-Path $LockDir2) {
    Write-Host "Removing stale lock directory (package-r)..." -ForegroundColor Yellow
    Remove-Item -Path $LockDir2 -Recurse -Force
}

Write-Host "Building and installing perpetual package..." -ForegroundColor Cyan
$RLibUnix = $RLib.Replace('\', '/')

# Run R CMD INSTALL directly to the console
& $Rexe CMD INSTALL package-r --library="$RLibUnix"

if ($LASTEXITCODE -ne 0) {
    Write-Error "R CMD INSTALL failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} else {
    Write-Host "Installation successful." -ForegroundColor Green
}

Write-Host "Running R tests..." -ForegroundColor Cyan
& $RscriptExe -e ".libPaths('$RLibUnix'); library(testthat); library(perpetual); test_dir('package-r/tests/testthat')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "R tests completed successfully!" -ForegroundColor Green
} else {
    Write-Error "R tests failed!"
    exit $LASTEXITCODE
}
