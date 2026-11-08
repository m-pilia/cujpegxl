# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Flat directory of externally encoded cujpegxl .jxl files.

Each file is named `<frame>_<rung>_d=<distance>.jxl`, with the frame's source
PNG stem, the corpus rung name and the distance as Python float repr (`1.0`,
`0.5`, `2.5`). Files outside the requested matrix are ignored.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

# (frame stem, rung name, distance)
JxlKey = tuple[str, str, float]


def jxl_name(frame: str, rung: str, distance: float) -> str:
    return f"{frame}_{rung}_d={float(distance)!r}.jxl"


def jxl_paths(jxl_dir: pathlib.Path, keys: Iterable[JxlKey]) -> dict[JxlKey, pathlib.Path]:
    """Map every requested key to its file; FileNotFoundError lists all missing ones."""
    paths = {key: jxl_dir / jxl_name(*key) for key in keys}
    missing = sorted(path.name for path in paths.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"{jxl_dir} lacks {len(missing)} files: {', '.join(missing)}")
    return paths
