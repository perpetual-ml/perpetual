#!/bin/bash
set -e

# PGO Optimization Script for PerpetualBooster
# Ensure cargo is in PATH (common issue in some environments)
export PATH="$PATH:$HOME/.cargo/bin:/c/Users/wwwmu/.cargo/bin"

BENCH_NAME="training_benchmark"

echo "Step 1: Checking Tooling..."

# Check for cargo-pgo
if ! cargo pgo --version &> /dev/null; then
    echo "cargo-pgo not found. Installing..."
    cargo install cargo-pgo
else
    echo "cargo-pgo is already installed."
fi

# Check for llvm-tools-preview
if ! rustup component list | grep "llvm-tools-preview (installed)" &> /dev/null; then
    echo "llvm-tools-preview not found. Installing..."
    rustup component add llvm-tools-preview
else
    echo "llvm-tools-preview is already installed."
fi

echo "Step 2: Establishing Baseline..."
# Run standard benchmark to save baseline
cargo bench --bench $BENCH_NAME -- --noplot --save-baseline base
# Backup criterion data because cargo clean will wipe it
echo "Backing up baseline data..."
cp -r target/criterion criterion_backup

echo "Step 3: Cleaning Target..."
cargo clean

echo "Step 4: Instrumenting Build (Stage 1)..."
# Use -- to pass args to cargo
cargo pgo build -- --bench $BENCH_NAME

echo "Step 5: Profiling (Stage 2)..."
cargo pgo bench -- --bench $BENCH_NAME

echo "Step 6: Optimizing Build (Stage 3)..."
cargo pgo optimize -- --bench $BENCH_NAME

echo "Restoring baseline data..."
# Ensure target/criterion exists
mkdir -p target/criterion
cp -r criterion_backup/* target/criterion/
rm -rf criterion_backup

echo "Step 7: Validation..."
# Find the optimized benchmark binary
# It will be in target/<target-triple>/release/deps/ or target/release/deps/
# We look for the most recent executable matching the benchmark name
BENCH_BIN=$(find target -name "${BENCH_NAME}-*" -type f -executable | grep "release/deps" | sort -r | head -n 1)

if [ -z "$BENCH_BIN" ]; then
    # Fallback for Windows if 'executable' check fails or extension is needed
    BENCH_BIN=$(find target -name "${BENCH_NAME}-*.exe" | grep "release/deps" | sort -r | head -n 1)
fi

if [ -f "$BENCH_BIN" ]; then
    echo "Found optimized benchmark binary: $BENCH_BIN"
    echo "Running comparison against baseline..."
    "$BENCH_BIN" --noplot --baseline base
else
    echo "Error: Could not find optimized benchmark binary."
    exit 1
fi

echo "PGO Workflow Completed Successfully!"
