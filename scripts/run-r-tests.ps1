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

Write-Host "Setting up R dependencies..." -ForegroundColor Cyan
& $RscriptExe scripts/setup_r_deps.R

Write-Host "Building and installing perpetual package..." -ForegroundColor Cyan
$RLibUnix = $RLib.Replace('\', '/')

$installArgs = @("CMD", "INSTALL", "package-r", "--library=$RLibUnix")
$process = Start-Process -FilePath $Rexe -ArgumentList $installArgs -Wait -NoNewWindow -PassThru

if ($process.ExitCode -ne 0) {
    Write-Error "R CMD INSTALL failed with exit code $($process.ExitCode)"
    exit $process.ExitCode
}

Write-Host "Running R tests..." -ForegroundColor Cyan
Write-Host "Running R tests..." -ForegroundColor Cyan
& $RscriptExe -e ".libPaths('$RLibUnix'); library(testthat); library(perpetual); test_dir('package-r/tests/testthat')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "R tests completed successfully!" -ForegroundColor Green
} else {
    Write-Error "R tests failed!"
    exit $LASTEXITCODE
}
