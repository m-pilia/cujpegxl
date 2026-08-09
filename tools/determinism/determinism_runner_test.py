# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import sys
import unittest

import determinism_runner as dr


class DeterminismRunnerTest(unittest.TestCase):
    def test_deterministic_stdout_matches(self):
        result = dr.run_twice([sys.executable, "-c", "print('fixed')"], None)
        self.assertTrue(result.identical)
        self.assertIsNone(result.first_diff_offset)

    def test_nondeterministic_stdout_detected(self):
        result = dr.run_twice(
            [sys.executable, "-c", "import random; print(random.random())"], None
        )
        self.assertFalse(result.identical)
        self.assertIsNotNone(result.first_diff_offset)

    def test_output_file_mode_matches(self):
        result = dr.run_twice(
            [sys.executable, "-c", "open('{output}', 'w').write('payload')"],
            "output",
        )
        self.assertTrue(result.identical)


if __name__ == "__main__":
    unittest.main()
