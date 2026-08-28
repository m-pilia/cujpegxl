# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the cujpegxl encoder over a synthetic NV12 frame.

Test/instrumentation harness for W6d: it generates a deterministic NV12 image on
the host, uploads and encodes it through the Python binding, and writes the
`.jxl` bytes (so the determinism runner can double-run and byte-compare it).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import pycujpegxl


def synthetic_nv12(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    luma = ((x * 3 + y * 5) & 0xFF).astype(np.uint8)
    cy, cx = np.mgrid[0 : height // 2, 0:width]
    chroma = ((cx * 7 + cy * 11) & 0xFF).astype(np.uint8)
    return np.concatenate([luma.reshape(-1), chroma.reshape(-1)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", required=True, help="Frame size as WIDTHxHEIGHT.")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True, help="Write the encoded .jxl bytes to this path.")
    args = parser.parse_args(argv)

    width, height = (int(v) for v in args.dims.lower().split("x"))
    nv12 = synthetic_nv12(width, height)

    data = pycujpegxl.encode(nv12, width, height, args.distance, args.device)
    with open(args.output, "wb") as handle:
        handle.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
