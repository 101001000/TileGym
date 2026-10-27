#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run only the benchmarks that the tiletranslation backend implements
# Usage: ./run_all_tiletranslation.sh [OUTPUT_DIR]
#
# Like run_all.sh, but restricted to benchmark files whose backend list
# includes tiletranslation. The subset is auto-discovered: a bench_*.py file
# is selected when its ALL_BACKENDS table contains a ("tiletranslation", ...)
# entry, so new benchmarks are picked up without editing this script.

set -o pipefail

cd "$(dirname "$0")"

OUTPUT_DIR="${1:-.}"
mkdir -p "$OUTPUT_DIR"

# Check if output directory is writable
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory $OUTPUT_DIR is not writable" >&2
    exit 1
fi

# Auto-discover tiletranslation benchmarks: every bench_*.py that declares a
# ("tiletranslation", ...) backend in its backend table. Sorted for
# deterministic order; output names use the same path-flattening scheme as
# run_all.sh.
TILETRANSLATION_BENCHMARKS=()
while IFS= read -r file; do
    if grep -q '("tiletranslation"' "$file"; then
        TILETRANSLATION_BENCHMARKS+=("$file")
    fi
done < <(find . -name 'bench_*.py' -not -path './__pycache__/*' | sort)

echo "Running tiletranslation benchmarks sequentially (parallel execution disabled to ensure accurate results)..."
echo "Results will be saved to: $OUTPUT_DIR"
echo "Current directory: $(pwd)"
echo "tiletranslation benchmark files found: ${#TILETRANSLATION_BENCHMARKS[@]}"
for file in "${TILETRANSLATION_BENCHMARKS[@]}"; do
    echo "  $file"
done
echo ""

FAILED_BENCHMARKS=()

for file in "${TILETRANSLATION_BENCHMARKS[@]}"; do
    rel_path="${file#./}"
    safe_name="${rel_path//\//_}"
    safe_name="${safe_name%.py}"
    output_file="$OUTPUT_DIR/${safe_name}_results.txt"

    echo "=========================================="
    echo "Running $rel_path..."
    echo "=========================================="

    if python3 "$file" 2>&1 | tee "$output_file"; then
        chmod 644 "$output_file" 2>/dev/null || true
        echo "✓ PASSED: $rel_path"
        echo "  Results saved to: $output_file"
    else
        (echo "BENCHMARK FAILED"; echo ""; cat "$output_file") > "$output_file.new" 2>/dev/null && \
            mv "$output_file.new" "$output_file" 2>/dev/null || \
            echo "BENCHMARK FAILED" > "$output_file"
        chmod 644 "$output_file" 2>/dev/null || true
        echo "✗ FAILED: $rel_path"
        echo "  Error details saved to: $output_file"
        FAILED_BENCHMARKS+=("$rel_path")
    fi
    echo ""
done

echo "=========================================="
if [ ${#FAILED_BENCHMARKS[@]} -eq 0 ]; then
    echo "All tiletranslation benchmarks complete! ✓"
else
    echo "tiletranslation benchmarks complete with failures! ✗"
    echo "Failed benchmarks:"
    for failed in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $failed"
    done
fi
echo "Results directory: $OUTPUT_DIR"
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*_results.txt 2>/dev/null || echo "  No result files found"

# Generate the Google Sheets CSV (results + overhead formulas vs CuTile).
echo ""
echo "Generating CSV summary..."
if python3 bench_results_to_csv.py "$OUTPUT_DIR"; then
    echo "  CSV: $OUTPUT_DIR/tiletranslation_results.csv"
else
    echo "  Warning: CSV generation failed" >&2
fi
echo "=========================================="

# Exit with error if any benchmarks failed
if [ ${#FAILED_BENCHMARKS[@]} -gt 0 ]; then
    exit 1
fi
