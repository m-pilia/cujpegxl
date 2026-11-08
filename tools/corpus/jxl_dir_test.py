# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import tempfile
import unittest

import jxl_dir


class JxlNameTest(unittest.TestCase):
    def test_distance_is_float_repr(self):
        self.assertEqual(jxl_dir.jxl_name("a", "2160p", 1.0), "a_2160p_d=1.0.jxl")
        self.assertEqual(jxl_dir.jxl_name("a", "1080p", 2.5), "a_1080p_d=2.5.jxl")
        self.assertEqual(jxl_dir.jxl_name("a", "1080p", 4), "a_1080p_d=4.0.jxl")


class JxlPathsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)

    def test_maps_present_files_and_ignores_others(self):
        (self.dir / "a_2160p_d=1.0.jxl").write_bytes(b"x")
        (self.dir / "unrelated.jxl").write_bytes(b"y")
        paths = jxl_dir.jxl_paths(self.dir, [("a", "2160p", 1.0)])
        self.assertEqual(paths, {("a", "2160p", 1.0): self.dir / "a_2160p_d=1.0.jxl"})

    def test_lists_every_missing_file(self):
        (self.dir / "a_2160p_d=1.0.jxl").write_bytes(b"x")
        with self.assertRaises(FileNotFoundError) as ctx:
            jxl_dir.jxl_paths(self.dir, [("a", "2160p", 1.0), ("a", "2160p", 0.5),
                                         ("b", "1080p", 1.0)])
        self.assertIn("2 files", str(ctx.exception))
        self.assertIn("a_2160p_d=0.5.jxl", str(ctx.exception))
        self.assertIn("b_1080p_d=1.0.jxl", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
