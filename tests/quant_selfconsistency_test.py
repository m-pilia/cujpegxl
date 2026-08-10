# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""GPU self-consistency: forward DCT + cjxl-calibrated quantization.

Two checks per the milestone:
  1. The kernel's quantized integers equal round(coeff / step) exactly, where the
     per-coefficient step matches libjxl's quantizer for the given distance (DC
     via the separate DC quantizer, AC via the DCT8 dequant weight and the
     per-block AC step). Validates weights, layout, calibration, and rounding.
  2. Dequantizing and inverse-transforming reconstructs the XYB input within the
     per-pixel quantization-step bound (validates transform invertibility).

The forward DCT itself is separately validated against libjxl by
dct_conformance_test; here the matching NumPy transform is used as the inverse.
The calibration below mirrors src/quant_calibration.h byte-for-byte.
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

# libjxl DC quantization reciprocals (kInvDCQuant), channels X, Y, B.
_DC_INV_QUANT = (4096.0, 512.0, 256.0)


def _calibrate(distance: float) -> dict:
    """Port of src/quant_calibration.h calibrate_quant (single-precision)."""
    f32 = np.float32
    k_ac_quant = f32(0.725)
    k_dc_quant = f32(1.095924047623553)
    k_dc_quant_pow = f32(0.83)
    k_dc_mul = f32(0.3)
    k_quant_field_target = f32(5.0)
    k_global_scale_denom = 1 << 16
    k_global_scale_numerator = 4096
    k_quant_max = 256

    d = f32(distance)
    quant_ac = f32(k_ac_quant / d)
    bt_dc = max(
        f32(0.5) * d,
        min(d, f32(k_dc_mul * f32((f32(1.0) / k_dc_mul) * d) ** k_dc_quant_pow)),
    )
    quant_dc_float = min(f32(k_dc_quant / bt_dc), f32(50.0))

    scale = f32(f32(k_global_scale_denom) * quant_ac / k_quant_field_target)
    scale = max(f32(1.0), min(scale, f32(1 << 15)))
    global_scale = int(scale)
    scaled_quant_dc = int(f32(quant_dc_float * k_global_scale_numerator * f32(1.6)))
    if global_scale > scaled_quant_dc:
        global_scale = 1 if scaled_quant_dc <= 0 else scaled_quant_dc

    inv_global_scale = f32(f32(1.0) * f32(k_global_scale_denom) / f32(global_scale))
    fval = min(f32(1 << 16), f32(quant_dc_float * inv_global_scale + f32(0.5)))
    quant_dc = int(fval)
    q = int(np.round(np.float64(quant_ac) * np.float64(inv_global_scale)))
    q = max(1, min(k_quant_max, q))

    return {
        "global_scale": global_scale,
        "quant_dc": quant_dc,
        "raw_quant_field": q,
        "global_scale_float": float(f32(global_scale) * f32(1.0 / k_global_scale_denom)),
        "inv_global_scale": float(inv_global_scale),
    }


def _coeff_steps(weights_c: np.ndarray, c: int, cal: dict) -> np.ndarray:
    """Per-coefficient dequant step for channel c (index 0=DC, 1..63=AC)."""
    step = weights_c * (cal["inv_global_scale"] / cal["raw_quant_field"])
    step[0] = 1.0 / (_DC_INV_QUANT[c] * cal["global_scale_float"] * cal["quant_dc"])
    return step


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
        cal = _calibrate(distance)
        plane = width * height
        blocks_x = width // 8
        q = _kernel_quant(xyb, distance)

        for c in range(3):
            step = _coeff_steps(weights[c], c, cal)
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
