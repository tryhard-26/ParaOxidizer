#!/usr/bin/env bash
set -e

CRATES=(
    "paraoxidizer-core"
    "paraoxidizer-quant"
    "paraoxidizer-format"
    "paraoxidizer-security"
    "paraoxidizer-calibration"
    "paraoxidizer-optimizer"
    "paraoxidizer-runtime"
    "paraoxidizer-bench"
    "paraoxidizer-serve"
    "paraoxidizer-cli"
)

echo "Publishing ParaOxidizer crates to crates.io..."

for crate in "${CRATES[@]}"; do
    echo "===================================================="
    echo "Publishing $crate..."
    echo "===================================================="
    cargo publish -p "$crate"
    echo "Waiting 15 seconds for crates.io index to update..."
    sleep 15
done

echo "===================================================="
echo "Publishing root meta-package: paraoxidizer..."
echo "===================================================="
cargo publish -p paraoxidizer

echo "All crates published successfully!"
