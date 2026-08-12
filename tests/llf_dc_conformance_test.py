# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: dc_from_llf vs libjxl's DCFromLowestFrequencies.

For each square DCT size the encoder's low-frequency -> DC derivation must match
libjxl exactly, so the DC-image entries the covered 8x8 positions take are the
ones djxl reconstructs. The coefficients are produced by the libjxl DCT oracle
(same layout the GPU forward transform emits). CPU-only.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_LAYOUT_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _oracle_dct(block_dim: int, pixels: np.ndarray) -> np.ndarray:
    # One block: reuse the 3-plane oracle dct by replicating the block into all
    # three planes and keeping the first plane's coefficients.
    n = block_dim
    planes = np.broadcast_to(pixels, (3, n, n)).astype("<f4")
    with tempfile.TemporaryDirectory() as scratch:
        in_path = pathlib.Path(scratch) / "xyb.f32"
        out_path = pathlib.Path(scratch) / "coeff.f32"
        np.ascontiguousarray(planes).tofile(in_path)
        subprocess.run(
            [str(_ORACLE_BIN), "dctn", str(n), str(n), str(n), str(in_path), str(out_path)],
            check=True,
        )
        return np.fromfile(out_path, dtype="<f4")[: n * n]


def _dc(binary: pathlib.Path, block_dim: int, coeffs: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as scratch:
        in_path = pathlib.Path(scratch) / "coeff.f32"
        out_path = pathlib.Path(scratch) / "dc.f32"
        np.ascontiguousarray(coeffs, dtype="<f4").tofile(in_path)
        args = [str(binary), "dcfromllf", str(block_dim), str(in_path), str(out_path)]
        subprocess.run(args, check=True)
        return np.fromfile(out_path, dtype="<f4")


class LlfDcConformanceTest(unittest.TestCase):
    def _check(self, block_dim: int, pixels: np.ndarray) -> None:
        coeffs = _oracle_dct(block_dim, pixels)
        mine = _dc(_LAYOUT_BIN, block_dim, coeffs)
        want = _dc(_ORACLE_BIN, block_dim, coeffs)
        side = block_dim // 8
        self.assertEqual(mine.shape, (side * side,))
        np.testing.assert_allclose(mine, want, atol=2e-4)

    def test_random(self):
        rng = np.random.default_rng(11)
        for block_dim in (8, 16, 32):
            with self.subTest(block_dim=block_dim):
                self._check(block_dim, rng.uniform(-0.5, 0.9, (block_dim, block_dim)).astype("<f4"))

    def test_gradient(self):
        for block_dim in (8, 16, 32):
            with self.subTest(block_dim=block_dim):
                yy, xx = np.mgrid[0:block_dim, 0:block_dim]
                ramp = ((xx + yy) / (2.0 * block_dim)).astype("<f4")
                self._check(block_dim, ramp)

    def test_matches_subblock_means(self):
        # DCFromLowestFrequencies of a smooth block approximates the per-8x8
        # sub-block means; for a constant block it is exact.
        for block_dim in (8, 16, 32):
            with self.subTest(block_dim=block_dim):
                pixels = np.full((block_dim, block_dim), 0.42, dtype="<f4")
                coeffs = _oracle_dct(block_dim, pixels)
                mine = _dc(_LAYOUT_BIN, block_dim, coeffs)
                np.testing.assert_allclose(mine, 0.42, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
