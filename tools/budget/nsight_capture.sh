#!/usr/bin/env bash
# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

# Capture GPU timing and DRAM traffic on the development host for a single
# encode invocation, producing the per-stage measurements the analytic budget
# model consumes.
#
# Requires the NVIDIA Nsight Systems (nsys) and Nsight Compute (ncu) CLIs and a
# physical GPU, so it runs only on the development host, never in the CPU-only
# CI. Host measurements are projected onto the Orin Nano by budget_model.py
# using the factors in projection_sheet.json.
#
# Usage: nsight_capture.sh <output_dir> -- <command> [args...]
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <output_dir> -- <command> [args...]" >&2
    exit 2
fi

out_dir="$1"
shift
if [[ "$1" != "--" ]]; then
    echo "expected '--' before the command" >&2
    exit 2
fi
shift

mkdir -p "$out_dir"

for tool in nsys ncu; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: '$tool' not found; run on a host with Nsight installed" >&2
        exit 3
    fi
done

# System-wide timeline: kernel durations and host<->device memcpy volume.
nsys profile \
    --output "$out_dir/timeline" \
    --force-overwrite true \
    --trace cuda,nvtx \
    --stats true \
    "$@"

# Per-kernel DRAM read/write bytes, the input to the bandwidth constraint.
ncu \
    --set full \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
    --export "$out_dir/kernels" \
    --force-overwrite \
    "$@"

echo "Captured Nsight reports under $out_dir"
