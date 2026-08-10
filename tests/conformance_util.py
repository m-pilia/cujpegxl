# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Reference NV12->RGB path and PSNR for the ladder conformance test.

The encoder consumes NV12 (BT.709, full range, 4:2:0). djxl decodes the produced
.jxl back to 8-bit sRGB RGB. To measure the codec's own loss we compare that
against the RGB the encoder itself derived from the NV12 input: the same
centered-bilinear chroma upsample and BT.709 full-range inverse used by
src/xyb.cu's nv12_to_xyb kernel (before the sRGB->linear->opsin step, which djxl
inverts). This reference is a defined comparison target, so a small mismatch with
the encoder's texture-unit filtering only shifts the per-rung PSNR baseline the
test thresholds are recorded against.
"""

from __future__ import annotations

import numpy as np

# BT.709 full-range Y'CbCr -> R'G'B', matching src/xyb.cu.
_CR_TO_R = 1.5748
_CB_TO_G = 0.1873242729
_CR_TO_G = 0.4681242729
_CB_TO_B = 1.8556
_CHROMA_CENTER = 128.0 / 255.0


def _upsample_bilinear(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Centered bilinear 2x upsample matching the kernel's chroma texture fetch.

    Luma pixel x samples the chroma texel-index coordinate (x + 0.5) / 2 - 0.5,
    with clamp-to-edge addressing (cudaAddressModeClamp)."""
    in_h, in_w = plane.shape

    def _axis(out_n: int, in_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coord = (np.arange(out_n) + 0.5) * 0.5 - 0.5
        i0 = np.floor(coord).astype(np.int64)
        frac = coord - i0
        lo = np.clip(i0, 0, in_n - 1)
        hi = np.clip(i0 + 1, 0, in_n - 1)
        return lo, hi, frac

    ylo, yhi, yfrac = _axis(out_h, in_h)
    xlo, xhi, xfrac = _axis(out_w, in_w)

    top = plane[ylo][:, xlo] * (1 - xfrac) + plane[ylo][:, xhi] * xfrac
    bot = plane[yhi][:, xlo] * (1 - xfrac) + plane[yhi][:, xhi] * xfrac
    return top * (1 - yfrac)[:, None] + bot * yfrac[:, None]


def nv12_to_reference_rgb(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    """8-bit sRGB RGB the encoder derives from an NV12 buffer (H, W, 3)."""
    luma = nv12[: width * height].reshape(height, width).astype(np.float64) / 255.0
    chroma = nv12[width * height :].reshape(height // 2, width).astype(np.float64) / 255.0
    cb = _upsample_bilinear(chroma[:, 0::2], height, width) - _CHROMA_CENTER
    cr = _upsample_bilinear(chroma[:, 1::2], height, width) - _CHROMA_CENTER

    r = np.clip(luma + _CR_TO_R * cr, 0.0, 1.0)
    g = np.clip(luma - _CB_TO_G * cb - _CR_TO_G * cr, 0.0, 1.0)
    b = np.clip(luma + _CB_TO_B * cb, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.rint(rgb * 255.0).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))
