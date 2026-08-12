# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: natural_coeff_order vs libjxl's ComputeNaturalCoeffOrder.

For each square DCT size the encoder's LUT generator must reproduce libjxl's
natural coefficient order exactly (scan index -> transposed-raster index), so the
entropy stage tokenizes large-block coefficients in the order the decoder expects.
CPU-only.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_ORDER_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _order(binary: pathlib.Path, args: list[str]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as scratch:
        out_path = pathlib.Path(scratch) / "order.u32"
        subprocess.run([str(binary), *args, str(out_path)], check=True)
        return np.fromfile(out_path, dtype="<u4")


class CoeffOrderConformanceTest(unittest.TestCase):
    def test_matches_libjxl(self):
        for block_dim in (8, 16, 32):
            with self.subTest(block_dim=block_dim):
                mine = _order(_ORDER_BIN, [str(block_dim)])
                want = _order(_ORACLE_BIN, ["coefforder", str(block_dim)])
                self.assertEqual(mine.shape, (block_dim * block_dim,))
                np.testing.assert_array_equal(mine, want)

    def test_is_permutation(self):
        for block_dim in (8, 16, 32):
            with self.subTest(block_dim=block_dim):
                mine = _order(_ORDER_BIN, [str(block_dim)])
                np.testing.assert_array_equal(
                    np.sort(mine), np.arange(block_dim * block_dim, dtype="<u4")
                )


if __name__ == "__main__":
    unittest.main()
