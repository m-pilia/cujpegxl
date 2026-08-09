# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Python front-end to the libjxl validation oracle.

The oracle binary exchanges planar float32 buffers (3 planes, row-major, no
padding) on disk. This module handles that IO and shells out to the binary so
tests can obtain libjxl reference intermediates as NumPy arrays.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

PLANAR_DTYPE = np.dtype("<f4")


def write_planar(path: pathlib.Path, planes: np.ndarray) -> None:
    assert planes.ndim == 3 and planes.shape[0] == 3
    np.ascontiguousarray(planes, dtype=PLANAR_DTYPE).tofile(path)


def read_planar(path: pathlib.Path, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=PLANAR_DTYPE)
    assert data.size == 3 * height * width
    return data.reshape(3, height, width)


def xyb_from_srgb(binary: pathlib.Path, srgb: np.ndarray) -> np.ndarray:
    """Reference libjxl opsin XYB for an sRGB-encoded RGB image in [0, 1].

    `srgb` has shape (3, H, W); the returned XYB has the same shape.
    """
    _, height, width = srgb.shape
    with tempfile.TemporaryDirectory() as scratch:
        in_path = pathlib.Path(scratch) / "srgb.f32"
        out_path = pathlib.Path(scratch) / "xyb.f32"
        write_planar(in_path, srgb)
        subprocess.run(
            [str(binary), "xyb", str(width), str(height), str(in_path), str(out_path)],
            check=True,
        )
        return read_planar(out_path, height, width)
