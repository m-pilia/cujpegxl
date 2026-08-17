# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pycujpegxl


class PyCujpegxlTest(unittest.TestCase):
    def test_api_version(self):
        self.assertEqual(pycujpegxl.api_version(), 3)
        self.assertEqual(pycujpegxl.API_VERSION, 3)

    def test_query_backend_is_exposed(self):
        self.assertTrue(callable(pycujpegxl.query_backend))

    def test_encode_api_is_exposed(self):
        self.assertTrue(callable(pycujpegxl.encode))
        self.assertTrue(callable(pycujpegxl.encode_with_stats))

    def test_query_backend_missing_device_raises(self):
        with self.assertRaises(RuntimeError):
            pycujpegxl.query_backend(9999)


if __name__ == "__main__":
    unittest.main()
