# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Reference front-end math shared by GPU conformance tests.

Reproduces the encoder's pre-XYB path (NV12 -> chroma upsample -> BT.709
full-range Y'CbCr -> R'G'B') so the resulting sRGB image can be handed to the
libjxl XYB oracle. This is the simple, low-risk arithmetic half; the perceptual
XYB transform itself is validated against libjxl, not against this file.
"""

from __future__ import annotations

import numpy as np

# BT.709 full-range Y'CbCr -> R'G'B' (inverse of the corpus generator matrix).
CR_TO_R = 1.5748
CB_TO_G = 0.1873242729
CR_TO_G = 0.4681242729
CB_TO_B = 1.8556


def upsample_chroma(plane: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centered bilinear 4:2:0 -> 4:4:4 upsample matching the CUDA texture path.

    Luma pixel x samples chroma continuous coordinate (x + 0.5) / 2, i.e. index
    (x - 0.5) / 2 with clamp-to-edge, and likewise for y.
    """
    ch, cw = plane.shape
    xs = (np.arange(width) + 0.5) / 2.0 - 0.5
    ys = (np.arange(height) + 0.5) / 2.0 - 0.5
    x0 = np.floor(xs)
    y0 = np.floor(ys)
    fx = xs - x0
    fy = ys - y0
    # Clamp each tap independently (clamp-to-edge), matching the texture unit;
    # deriving the second tap from the already-clamped first would blend across
    # the edge instead of repeating it.
    x0i = np.clip(x0.astype(np.int64), 0, cw - 1)
    x1i = np.clip(x0.astype(np.int64) + 1, 0, cw - 1)
    y0i = np.clip(y0.astype(np.int64), 0, ch - 1)
    y1i = np.clip(y0.astype(np.int64) + 1, 0, ch - 1)

    top = plane[y0i][:, x0i] * (1 - fx) + plane[y0i][:, x1i] * fx
    bot = plane[y1i][:, x0i] * (1 - fx) + plane[y1i][:, x1i] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def nv12_to_srgb(nv12: bytes, width: int, height: int) -> np.ndarray:
    """NV12 (BT.709 full range) -> sRGB-encoded R'G'B' in [0, 1], shape (3,H,W)."""
    buffer = np.frombuffer(nv12, dtype=np.uint8)
    luma = buffer[: width * height].reshape(height, width).astype(np.float64)
    chroma = buffer[width * height :].reshape(height // 2, width).astype(np.float64)
    cb = upsample_chroma(chroma[:, 0::2], width, height)
    cr = upsample_chroma(chroma[:, 1::2], width, height)

    y = luma / 255.0
    cb = cb / 255.0 - 128.0 / 255.0
    cr = cr / 255.0 - 128.0 / 255.0

    r = np.clip(y + CR_TO_R * cr, 0.0, 1.0)
    g = np.clip(y - CB_TO_G * cb - CR_TO_G * cr, 0.0, 1.0)
    b = np.clip(y + CB_TO_B * cb, 0.0, 1.0)
    return np.stack([r, g, b], axis=0).astype(np.float32)
