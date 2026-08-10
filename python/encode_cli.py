# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the cujpegxl encoder over a synthetic NV12 frame.

Test/instrumentation harness for W6d: it generates a deterministic NV12 image on
the host, uploads and encodes it through the Python binding, and either writes
the `.jxl` bytes (so the determinism runner can double-run and byte-compare it)
or emits per-stage records in the budget model's input schema.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import pycujpegxl


def synthetic_nv12(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    luma = ((x * 3 + y * 5) & 0xFF).astype(np.uint8)
    cy, cx = np.mgrid[0 : height // 2, 0:width]
    chroma = ((cx * 7 + cy * 11) & 0xFF).astype(np.uint8)
    return np.concatenate([luma.reshape(-1), chroma.reshape(-1)])


def budget_document(stages: list[dict]) -> dict:
    total_us = sum(s["gpu_us"] + s["cpu_us"] for s in stages)
    fps = 1.0e6 / total_us if total_us > 0 else 0.0
    return {
        "fps": fps,
        "projection": {"gpu_time": 1.0, "cpu_time": 1.0, "bandwidth": 1.0},
        "stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", required=True, help="Frame size as WIDTHxHEIGHT.")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    sink = parser.add_mutually_exclusive_group(required=True)
    sink.add_argument("--output", help="Write the encoded .jxl bytes to this path.")
    sink.add_argument(
        "--stage-records",
        help="Write per-stage budget-model records (JSON) to this path.",
    )
    args = parser.parse_args(argv)

    width, height = (int(v) for v in args.dims.lower().split("x"))
    nv12 = synthetic_nv12(width, height)

    if args.output is not None:
        data = pycujpegxl.encode(nv12, width, height, args.distance, args.device)
        with open(args.output, "wb") as handle:
            handle.write(data)
        return 0

    # Discard a warmup encode so the measured records reflect steady state
    # rather than one-time CUDA context/JIT initialization.
    pycujpegxl.encode(nv12, width, height, args.distance, args.device)
    _, stages = pycujpegxl.encode_with_stats(nv12, width, height, args.distance, args.device)
    with open(args.stage_records, "w") as handle:
        json.dump(budget_document(stages), handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
