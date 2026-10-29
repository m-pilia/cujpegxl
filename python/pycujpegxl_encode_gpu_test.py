# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

# Requires a physical CUDA device; tagged "gpu"/"manual" and excluded from the
# CPU-only CI. Exercises the Python encode wrapper (host NumPy NV12 upload), the
# determinism runner wired around the encoder, and the per-stage timing stats.

import unittest

import numpy as np

import determinism_runner as dr
import pycujpegxl

WIDTH = 512
HEIGHT = 512

JXL_CONTAINER_PREFIX = b"\x00\x00\x00\x0cJXL "


def synthetic_nv12(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    luma = ((x * 3 + y * 5) & 0xFF).astype(np.uint8)
    cy, cx = np.mgrid[0 : (height + 1) // 2, 0 : (width + 1) // 2 * 2]
    chroma = ((cx * 7 + cy * 11) & 0xFF).astype(np.uint8)
    return np.concatenate([luma.reshape(-1), chroma.reshape(-1)])


class PyCujpegxlEncodeTest(unittest.TestCase):
    def setUp(self):
        self.nv12 = synthetic_nv12(WIDTH, HEIGHT)

    def test_encode_returns_jxl_container(self):
        data = pycujpegxl.encode(self.nv12, WIDTH, HEIGHT)
        self.assertGreater(len(data), 0)
        self.assertTrue(data.startswith(JXL_CONTAINER_PREFIX))

    def test_encode_rejects_wrong_size(self):
        with self.assertRaises(ValueError):
            pycujpegxl.encode(self.nv12[:-1], WIDTH, HEIGHT)

    def test_encode_odd_size(self):
        width, height = 643, 389
        nv12 = synthetic_nv12(width, height)
        a = pycujpegxl.encode(nv12, width, height)
        b = pycujpegxl.encode(nv12, width, height)
        self.assertTrue(a.startswith(JXL_CONTAINER_PREFIX))
        self.assertEqual(a, b)

    def test_encode_rejects_odd_size_without_last_chroma_pair(self):
        width, height = 643, 389
        with self.assertRaises(ValueError):
            pycujpegxl.encode(np.zeros(width * height * 3 // 2, np.uint8), width, height)

    def test_determinism_via_runner(self):
        a = dr.RunArtifact(data=pycujpegxl.encode(self.nv12, WIDTH, HEIGHT))
        b = dr.RunArtifact(data=pycujpegxl.encode(self.nv12, WIDTH, HEIGHT))
        result = dr.compare(a, b)
        self.assertTrue(result.identical, msg=f"first diff at {result.first_diff_offset}")

    def test_encode_with_stats_returns_stage_timings(self):
        _, stages = pycujpegxl.encode_with_stats(self.nv12, WIDTH, HEIGHT)
        self.assertEqual([s["name"] for s in stages], ["frontend", "entropy", "assembly"])
        for stage in stages:
            self.assertGreaterEqual(stage["bytes_moved"], 0)
            self.assertGreaterEqual(stage["gpu_us"], 0.0)
            self.assertGreaterEqual(stage["cpu_us"], 0.0)


if __name__ == "__main__":
    unittest.main()
