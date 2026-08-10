# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np

import corpus_prep as cp


class CorpusPrepTest(unittest.TestCase):
    def test_box_downscale_averages_block(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0] = [0, 0, 0]
        rgb[0, 1] = [10, 10, 10]
        rgb[1, 0] = [20, 20, 20]
        rgb[1, 1] = [30, 30, 30]
        scaled = cp.box_downscale(rgb, 2)
        self.assertEqual(scaled.shape, (1, 1, 3))
        self.assertEqual(scaled[0, 0, 0], 15)

    def test_downscale_factor_one_is_identity(self):
        rgb = (np.arange(2 * 2 * 3).reshape(2, 2, 3) % 256).astype(np.uint8)
        np.testing.assert_array_equal(cp.box_downscale(rgb, 1), rgb)

    def test_rgb_to_nv12_pure_red(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[:] = [255, 0, 0]
        nv12 = cp.rgb_to_nv12(rgb)
        # Y plane (4 bytes) + one interleaved Cb/Cr pair.
        self.assertEqual(len(nv12), 2 * 2 + 2)
        self.assertEqual(nv12[0], 54)
        self.assertEqual(nv12[4], 99)
        self.assertEqual(nv12[5], 255)

    def test_nv12_size_matches_geometry(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        nv12 = cp.rgb_to_nv12(rgb)
        self.assertEqual(len(nv12), 4 * 6 + (4 // 2) * 6)

    def test_ladder_geometry(self):
        self.assertEqual([r.width for r in cp.LADDER], [3840, 1920, 1280])
        self.assertEqual([r.height for r in cp.LADDER], [2160, 1080, 720])

    def test_ladder_dims_are_block_aligned(self):
        for rung in cp.LADDER:
            self.assertEqual(rung.width % 8, 0)
            self.assertEqual(rung.height % 8, 0)


if __name__ == "__main__":
    unittest.main()
