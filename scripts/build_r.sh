#!/bin/bash
set -e

# Navigate to project root (assuming script is in scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

# Setup environment (if cargo is not in path)
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# R Library Path
# You can customize this or let R verify it
RLIB="$HOME/R/x86_64-pc-linux-gnu-library/4.3"
mkdir -p "$RLIB"

echo "Building Rust library..."
cd package-r/src/rust
cargo build --release
# Copy the static library to src/ where Makevars expects it
cp target/release/libperpetual_r.a ../
cd ../../..

echo "Installing package with R CMD INSTALL..."
R CMD INSTALL package-r --library="$RLIB"

echo "Running tests..."
Rscript -e ".libPaths('$RLIB'); library(testthat); library(perpetual); test_dir('package-r/tests/testthat')"

echo "Done!"
