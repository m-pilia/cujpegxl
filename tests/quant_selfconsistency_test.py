# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""GPU self-consistency: forward DCT + uniform quantization.

Two checks per the milestone:
  1. The kernel's quantized integers equal round(coeff / step) exactly (validates
     weights, layout, and rounding against the definition).
  2. Dequantizing and inverse-transforming reconstructs the XYB input within the
     per-pixel quantization-step bound (validates transform invertibility).

The forward DCT itself is separately validated against libjxl by
dct_conformance_test; here the matching NumPy transform is used as the inverse.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_QUANT_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _dct8_matrix() -> np.ndarray:
    a = np.zeros((8, 8))
    for k in range(8):
        g = 0.125 if k == 0 else np.sqrt(2.0) / 8.0
        for n in range(8):
            a[k, n] = g * np.cos(np.pi * (n + 0.5) * k / 8.0)
    forward = np.zeros((64, 64))
    for fx in range(8):
        for fy in range(8):
            for y in range(8):
                for x in range(8):
                    forward[fx * 8 + fy, y * 8 + x] = a[fy, y] * a[fx, x]
    return forward


def _weights() -> np.ndarray:
    with tempfile.TemporaryDirectory() as scratch:
        out = pathlib.Path(scratch) / "w.f32"
        subprocess.run([str(_ORACLE_BIN), "quantmatrix", str(out)], check=True)
        return np.fromfile(out, dtype="<f4").reshape(3, 64)


def _kernel_quant(xyb: np.ndarray, distance: float) -> np.ndarray:
    _, height, width = xyb.shape
    with tempfile.TemporaryDirectory() as scratch:
        in_path = pathlib.Path(scratch) / "xyb.f32"
        out_path = pathlib.Path(scratch) / "q.i32"
        np.ascontiguousarray(xyb, dtype="<f4").tofile(in_path)
        subprocess.run(
            [str(_QUANT_BIN), "quant", str(width), str(height), str(distance),
             str(in_path), str(out_path)],
            check=True,
        )
        return np.fromfile(out_path, dtype="<i4")


class QuantSelfConsistencyTest(unittest.TestCase):
    def _check(self, xyb: np.ndarray, distance: float) -> None:
        _, height, width = xyb.shape
        forward = _dct8_matrix()
        inverse = np.linalg.inv(forward)
        weights = _weights()
        plane = width * height
        blocks_x = width // 8
        q = _kernel_quant(xyb, distance)

        for c in range(3):
            step = weights[c] * distance
            for by in range(height // 8):
                for bx in range(blocks_x):
                    block = xyb[c, by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8].reshape(64)
                    coeff = forward @ block
                    q_ref = np.rint(coeff / step).astype(np.int32)
                    base = c * plane + (by * blocks_x + bx) * 64
                    q_block = q[base : base + 64]
                    np.testing.assert_array_equal(q_block, q_ref)

                    recon = inverse @ (q_block * step)
                    bound = np.abs(inverse) @ (0.5 * step)
                    self.assertTrue(np.all(np.abs(recon - block) <= bound + 1e-4))

    def test_random_xyb_distance1(self):
        rng = np.random.default_rng(4)
        xyb = np.empty((3, 32, 32), dtype=np.float32)
        xyb[0] = rng.uniform(-0.03, 0.03, (32, 32))
        xyb[1] = rng.uniform(0.0, 0.85, (32, 32))
        xyb[2] = rng.uniform(0.0, 0.85, (32, 32))
        self._check(xyb, 1.0)

    def test_random_xyb_distance3(self):
        rng = np.random.default_rng(5)
        xyb = np.empty((3, 16, 24), dtype=np.float32)
        xyb[0] = rng.uniform(-0.03, 0.03, (16, 24))
        xyb[1] = rng.uniform(0.0, 0.85, (16, 24))
        xyb[2] = rng.uniform(0.0, 0.85, (16, 24))
        self._check(xyb, 3.0)


if __name__ == "__main__":
    unittest.main()
