# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import sys

import budget_model as bm


def main() -> None:
    stages, fps, projection = bm.load_input(pathlib.Path(sys.argv[1]))
    report = bm.evaluate(stages, fps, projection)
    print(bm.format_report(report))
    assert report.passed, "sample stage records must fit the Orin Nano budget"


if __name__ == "__main__":
    main()
