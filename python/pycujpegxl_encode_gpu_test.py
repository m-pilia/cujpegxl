# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

# Requires a physical CUDA device; tagged "gpu"/"manual" and excluded from the
# CPU-only CI. Exercises the Python encode wrapper (host NumPy NV12 upload), the
# determinism runner wired around the encoder, and the budget-model stage-record
# emission.

import unittest

import numpy as np

import budget_model as bm
import determinism_runner as dr
import pycujpegxl

WIDTH = 512
HEIGHT = 512

JXL_CONTAINER_PREFIX = b"\x00\x00\x00\x0cJXL "


def synthetic_nv12(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    luma = ((x * 3 + y * 5) & 0xFF).astype(np.uint8)
    cy, cx = np.mgrid[0 : height // 2, 0:width]
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

    def test_pipelined_encode_preserves_sequence(self):
        encoder = pycujpegxl.Encoder(WIDTH, HEIGHT, pipeline_depth=2)
        first = encoder.submit(self.nv12, sequence=17)
        second = encoder.submit(self.nv12, sequence=18)

        self.assertEqual(first.sequence, 17)
        self.assertEqual(second.sequence, 18)
        self.assertTrue(first.result(timeout=60).startswith(JXL_CONTAINER_PREFIX))
        self.assertTrue(second.result(timeout=60).startswith(JXL_CONTAINER_PREFIX))
        self.assertTrue(first.ready)
        self.assertTrue(second.ready)

    def test_determinism_via_runner(self):
        a = dr.RunArtifact(data=pycujpegxl.encode(self.nv12, WIDTH, HEIGHT))
        b = dr.RunArtifact(data=pycujpegxl.encode(self.nv12, WIDTH, HEIGHT))
        result = dr.compare(a, b)
        self.assertTrue(result.identical, msg=f"first diff at {result.first_diff_offset}")

    def test_stage_records_feed_budget_model(self):
        _, stages = pycujpegxl.encode_with_stats(self.nv12, WIDTH, HEIGHT)
        self.assertEqual([s["name"] for s in stages], ["frontend", "entropy", "assembly"])
        for stage in stages:
            self.assertGreaterEqual(stage["bytes_moved"], 0)
            self.assertGreaterEqual(stage["gpu_us"], 0.0)
            self.assertGreaterEqual(stage["cpu_us"], 0.0)

        records = [bm.StageRecord.from_dict(s) for s in stages]
        report = bm.evaluate(records, fps=15.0, projection=bm.ProjectionFactors(1.0, 1.0, 1.0))
        self.assertEqual(len(report.constraints), 4)


if __name__ == "__main__":
    unittest.main()
