# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity checks for the libjxl XYB oracle.

These do not assert exact values (that is the encoder's job to match); they
confirm the oracle links libjxl, runs, and produces XYB with the qualitative
structure of the opsin space.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

import oracle

_BINARY = pathlib.Path(sys.argv.pop(1)).resolve()


def _flat(value: float) -> np.ndarray:
    return np.full((3, 8, 8), value, dtype=np.float32)


class XybOracleTest(unittest.TestCase):
    def test_neutral_gray_has_zero_x_and_equal_yb(self):
        xyb = oracle.xyb_from_srgb(_BINARY, _flat(0.5))
        self.assertEqual(xyb.shape, (3, 8, 8))
        self.assertTrue(np.isfinite(xyb).all())
        np.testing.assert_allclose(xyb[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(xyb[1], xyb[2], atol=1e-4)

    def test_luma_increases_with_brightness(self):
        y_dark = oracle.xyb_from_srgb(_BINARY, _flat(0.2))[1].mean()
        y_bright = oracle.xyb_from_srgb(_BINARY, _flat(0.8))[1].mean()
        self.assertLess(y_dark, y_bright)

    def test_red_has_positive_x(self):
        red = np.zeros((3, 8, 8), dtype=np.float32)
        red[0] = 0.8
        xyb = oracle.xyb_from_srgb(_BINARY, red)
        self.assertGreater(xyb[0].mean(), 0.0)


if __name__ == "__main__":
    unittest.main()
