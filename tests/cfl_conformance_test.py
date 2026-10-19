# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance: cfl_ytox_ratio / cfl_ytob_ratio vs libjxl's ColorCorrelation.

The encoder's map-value -> correlation-factor mapping must match libjxl exactly
for every signed-int8 map value, so the decoder reconstructs chroma with the same
factors the encoder decorrelated with. CPU-only.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_CFL_BIN = pathlib.Path(sys.argv.pop(1)).resolve()
_ORACLE_BIN = pathlib.Path(sys.argv.pop(1)).resolve()


def _ratios(binary: pathlib.Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as scratch:
        out_path = pathlib.Path(scratch) / "ratios.f32"
        subprocess.run([str(binary), "cfl", str(out_path)], check=True)
        return np.fromfile(out_path, dtype="<f4")


class CflConformanceTest(unittest.TestCase):
    def test_matches_libjxl(self):
        mine = _ratios(_CFL_BIN)
        want = _ratios(_ORACLE_BIN)
        self.assertEqual(mine.shape, (512,))
        np.testing.assert_allclose(mine, want, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
