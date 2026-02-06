#!/bin/bash
set -e

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

RLIB="$PROJECT_ROOT/package-r/R_library"
mkdir -p "$RLIB"

export R_LIBS_USER="$RLIB"
export R_LIBS_SITE="$RLIB"

echo "Setting up R dependencies..."
Rscript scripts/setup_r_deps.R

echo "Building and installing perpetual package..."
# R CMD INSTALL will trigger Makevars which handles the Rust build
R CMD INSTALL package-r --library="$RLIB"

echo "Running R tests..."
Rscript -e ".libPaths('$RLIB'); library(testthat); library(perpetual); test_dir('package-r/tests/testthat', stop_on_failure=TRUE)"

echo "R tests completed successfully!"
