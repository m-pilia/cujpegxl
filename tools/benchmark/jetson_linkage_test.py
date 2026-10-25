# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import subprocess
import sys
import unittest


class JetsonLinkageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elf = subprocess.run(
            [sys.argv[2], "--file-header", "--dynamic-table", sys.argv[1]],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        ).stdout

    def test_aarch64(self):
        self.assertRegex(self.elf, r"Machine:\s+AArch64\b")

    def test_no_cuda_shared_libraries(self):
        needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", self.elf)
        self.assertTrue(needed, "Expected dynamically linked system libraries")
        sdk_libraries = [
            name for name in needed if name.lower().startswith(("libcuda", "libnv"))
        ]
        self.assertEqual(sdk_libraries, [])

    def test_no_runpath(self):
        self.assertNotRegex(self.elf, r"\((?:RPATH|RUNPATH)\)")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
