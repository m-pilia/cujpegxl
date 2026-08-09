# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""GPU conformance: forward 8x8 DCT kernel vs the libjxl DCT oracle.

Both consume the same planar XYB float image and must produce identical
coefficients (same libjxl raster layout: index = horizontal_freq*8 + vertical_freq).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_DCT_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _coeffs(binary: pathlib.Path, xyb: np.ndarray) -> np.ndarray:
    _, height, width = xyb.shape
    with tempfile.TemporaryDirectory() as scratch:
        in_path = pathlib.Path(scratch) / "xyb.f32"
        out_path = pathlib.Path(scratch) / "coeff.f32"
        np.ascontiguousarray(xyb, dtype="<f4").tofile(in_path)
        subprocess.run(
            [str(binary), "dct", str(width), str(height), str(in_path), str(out_path)],
            check=True,
        )
        return np.fromfile(out_path, dtype="<f4")


class DctConformanceTest(unittest.TestCase):
    def _max_abs_diff(self, xyb: np.ndarray) -> float:
        mine = _coeffs(_DCT_BIN, xyb)
        want = _coeffs(_ORACLE_BIN, xyb)
        return float(np.max(np.abs(mine - want)))

    def test_random_xyb(self):
        rng = np.random.default_rng(3)
        xyb = np.empty((3, 64, 64), dtype=np.float32)
        xyb[0] = rng.uniform(-0.03, 0.03, (64, 64))
        xyb[1] = rng.uniform(0.0, 0.85, (64, 64))
        xyb[2] = rng.uniform(0.0, 0.85, (64, 64))
        self.assertLess(self._max_abs_diff(xyb), 3e-4)

    def test_gradient_xyb(self):
        yy, xx = np.mgrid[0:64, 0:64]
        ramp = ((xx + yy) / 128.0).astype(np.float32)
        xyb = np.stack([ramp * 0.05 - 0.025, ramp, ramp * 0.5], axis=0).astype(np.float32)
        self.assertLess(self._max_abs_diff(xyb), 3e-4)

    def test_partial_edge_blocks(self):
        # 24x40: exercises non-square block grids that still tile by 8.
        rng = np.random.default_rng(9)
        xyb = rng.uniform(0.0, 0.8, (3, 40, 24)).astype(np.float32)
        self.assertLess(self._max_abs_diff(xyb), 3e-4)


if __name__ == "__main__":
    unittest.main()
