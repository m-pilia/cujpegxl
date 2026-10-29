# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pycujpegxl


class PyCujpegxlTest(unittest.TestCase):
    def test_api_version(self):
        self.assertEqual(pycujpegxl.api_version(), 5)
        self.assertEqual(pycujpegxl.API_VERSION, 5)

    def test_encode_api_is_exposed(self):
        self.assertTrue(callable(pycujpegxl.encode))
        self.assertTrue(callable(pycujpegxl.encode_with_stats))


if __name__ == "__main__":
    unittest.main()
