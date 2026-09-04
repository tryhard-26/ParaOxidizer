#!/usr/bin/env bash

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

echo "Publishing ParaOxidizer crates to crates.io with automated retry & rate limit handling..."

for crate in "${CRATES[@]}"; do
    while true; do
        echo "===================================================="
        echo "Attempting publication of: $crate..."
        echo "===================================================="
        out=$(cargo publish -p "$crate" 2>&1)
        res=$?
        echo "$out"
        if [ $res -eq 0 ]; then
            echo "Successfully published $crate!"
            echo "Sleeping 15 seconds for crates.io index update..."
            sleep 15
            break
        fi

        if echo "$out" | grep -q -i "already uploaded"; then
            echo "$crate is already published on crates.io. Proceeding..."
            break
        fi

        if echo "$out" | grep -q -i "429 Too Many Requests"; then
            echo "Encountered Crates.io new-crate rate limit. Waiting 60 seconds before retrying..."
            sleep 60
        else
            echo "Error publishing $crate. Waiting 10 seconds before retry..."
            sleep 10
        fi
    done
done

while true; do
    echo "===================================================="
    echo "Publishing root package: paraoxidizer..."
    echo "===================================================="
    out=$(cargo publish -p paraoxidizer 2>&1)
    res=$?
    echo "$out"
    if [ $res -eq 0 ] || echo "$out" | grep -q -i "already uploaded"; then
        echo "Successfully published root package paraoxidizer!"
        break
    fi
    echo "Retrying root package in 30 seconds..."
    sleep 30
done

echo "ALL CRATES PUBLISHED SUCCESSFULLY TO CRATES.IO!"
