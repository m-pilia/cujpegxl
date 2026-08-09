# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Full-ladder conformance: encode -> djxl -> RGB PSNR vs the NV12 input.

For every ladder rung and source frame the encoder (driven through the Python
binding) produces a .jxl that unmodified djxl decodes to 8-bit sRGB RGB. That
output is compared, in RGB PSNR, against the reference RGB the encoder derived
from the NV12 input (tests/conformance_util). The pass threshold per rung is the
worst measured PSNR minus a 2 dB margin, recorded in _THRESHOLDS_DB below. The
same run also checks per-rung determinism (byte-identical double encode).

gpu/manual: needs the 1080Ti and a system djxl; excluded from CPU CI.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

import conformance_util as cu
import corpus_prep as cp
import pycujpegxl

_DJXL = pathlib.Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else pathlib.Path("djxl")
_FRAMES = sorted(pathlib.Path("data").glob("*.png"))

_DISTANCE = 1.0

# Worst measured PSNR per rung minus a 2 dB margin (see module docstring).
# Measured on the GTX 1080Ti dev host at distance 1.0.
_THRESHOLDS_DB = {
    "2160p": 31.4,
    "1080p": 28.99,
    "720p": 28.9,
}


def _decode_png(jxl: bytes, work: pathlib.Path) -> np.ndarray:
    in_path = work / "frame.jxl"
    out_path = work / "frame.png"
    in_path.write_bytes(jxl)
    subprocess.run([str(_DJXL), str(in_path), str(out_path)], check=True,
                   capture_output=True)
    with Image.open(out_path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


class LadderConformanceTest(unittest.TestCase):
    def _run_rung(self, rung: cp.Rung) -> None:
        if not _FRAMES:
            self.skipTest("no source frames in data/")
        width, height = rung.width, rung.height
        worst = float("inf")
        for source in _FRAMES:
            rgb_4k = cp.load_rgb(source)
            scaled = cp.box_downscale(rgb_4k, rung.factor)
            nv12 = np.frombuffer(cp.rgb_to_nv12(scaled), dtype=np.uint8)

            jxl = pycujpegxl.encode(nv12, width, height, _DISTANCE)
            jxl2 = pycujpegxl.encode(nv12, width, height, _DISTANCE)
            self.assertEqual(jxl, jxl2, f"{source.name} {rung.name}: non-deterministic encode")

            with tempfile.TemporaryDirectory() as scratch:
                decoded = _decode_png(jxl, pathlib.Path(scratch))
            self.assertEqual(decoded.shape, (height, width, 3))

            reference = cu.nv12_to_reference_rgb(nv12, width, height)
            value = cu.psnr(decoded, reference)
            print(f"{rung.name} {source.stem}: PSNR={value:.2f} dB  ({len(jxl)} bytes)")
            worst = min(worst, value)

        print(f"{rung.name} worst PSNR={worst:.2f} dB  threshold={_THRESHOLDS_DB[rung.name]} dB")
        self.assertGreaterEqual(worst, _THRESHOLDS_DB[rung.name])

    def test_720p(self) -> None:
        self._run_rung(cp.Rung("720p", 3))

    def test_1080p(self) -> None:
        self._run_rung(cp.Rung("1080p", 2))

    def test_2160p(self) -> None:
        self._run_rung(cp.Rung("2160p", 1))


if __name__ == "__main__":
    unittest.main()
