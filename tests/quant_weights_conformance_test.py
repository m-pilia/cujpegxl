# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: baked DCT8/16/32 dequant matrices vs the libjxl quantmatrix oracle.

The committed weight headers must reproduce libjxl's DequantMatrices bit-for-bit,
so the encoder dequantizes to exactly what djxl reconstructs. CPU-only.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_WEIGHTS_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _weights(binary: pathlib.Path, block_dim: int, oracle: bool) -> np.ndarray:
    with tempfile.TemporaryDirectory() as scratch:
        out_path = pathlib.Path(scratch) / "w.f32"
        prefix = ["quantmatrix"] if oracle else []
        subprocess.run([str(binary), *prefix, str(block_dim), str(out_path)], check=True)
        return np.fromfile(out_path, dtype="<f4")


class QuantWeightsConformanceTest(unittest.TestCase):
    def test_matches_libjxl(self):
        for block_dim in (8, 16, 32):
            with self.subTest(block_dim=block_dim):
                mine = _weights(_WEIGHTS_BIN, block_dim, oracle=False)
                want = _weights(_ORACLE_BIN, block_dim, oracle=True)
                self.assertEqual(mine.shape, (3 * block_dim * block_dim,))
                np.testing.assert_array_equal(mine, want)


if __name__ == "__main__":
    unittest.main()
