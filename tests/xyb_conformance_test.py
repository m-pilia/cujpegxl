# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""GPU conformance: NV12->XYB kernel vs the libjxl XYB oracle.

A flat tile has an exact chroma upsample, so it pins the color-conversion and
opsin math tightly against libjxl. Textured tiles additionally exercise the
texture-unit bilinear upsample (whose 1/256 weight quantization is the dominant
source of divergence), bounded by a looser tolerance.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import oracle
import reference

_DUMP_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _make_nv12(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> bytes:
    height, width = y.shape
    chroma = np.empty((height // 2, width), dtype=np.uint8)
    chroma[:, 0::2] = cb
    chroma[:, 1::2] = cr
    return y.astype(np.uint8).tobytes() + chroma.astype(np.uint8).tobytes()


def _kernel_xyb(nv12: bytes, width: int, height: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as scratch:
        in_path = pathlib.Path(scratch) / "in.nv12"
        out_path = pathlib.Path(scratch) / "out.f32"
        in_path.write_bytes(nv12)
        subprocess.run(
            [str(_DUMP_BIN), "dump", str(width), str(height), str(in_path), str(out_path)],
            check=True,
        )
        return np.fromfile(out_path, dtype="<f4").reshape(3, height, width)


def _oracle_xyb(nv12: bytes, width: int, height: int) -> np.ndarray:
    srgb = reference.nv12_to_srgb(nv12, width, height)
    return oracle.xyb_from_srgb(_ORACLE_BIN, srgb)


class XybConformanceTest(unittest.TestCase):
    def _max_abs_diff(self, nv12: bytes, width: int, height: int) -> float:
        got = _kernel_xyb(nv12, width, height)
        want = _oracle_xyb(nv12, width, height)
        return float(np.max(np.abs(got - want)))

    def test_flat_tile_matches_libjxl_tightly(self):
        y = np.full((64, 64), 140, dtype=np.uint8)
        cb = np.full((32, 32), 100, dtype=np.uint8)
        cr = np.full((32, 32), 170, dtype=np.uint8)
        self.assertLess(self._max_abs_diff(_make_nv12(y, cb, cr), 64, 64), 3e-4)

    def test_gradient_tile(self):
        width, height = 64, 64
        yy, xx = np.mgrid[0:height, 0:width]
        y = (16 + (xx + yy) * 1.5).astype(np.uint8)
        cyy, cxx = np.mgrid[0 : height // 2, 0 : width // 2]
        cb = (64 + cxx * 2).astype(np.uint8)
        cr = (192 - cyy * 2).astype(np.uint8)
        self.assertLess(self._max_abs_diff(_make_nv12(y, cb, cr), width, height), 2e-3)

    def test_random_tile_bounded(self):
        rng = np.random.default_rng(0)
        width, height = 64, 64
        y = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        cb = rng.integers(0, 256, size=(height // 2, width // 2), dtype=np.uint8)
        cr = rng.integers(0, 256, size=(height // 2, width // 2), dtype=np.uint8)
        self.assertLess(self._max_abs_diff(_make_nv12(y, cb, cr), width, height), 2e-3)


if __name__ == "__main__":
    unittest.main()
