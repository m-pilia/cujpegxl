# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the ladder-resolution NV12 test corpus from source RGB frames.

For every source frame and every rung of the fixed resolution ladder the tool
produces a raw NV12 (BT.709, full range, 4:2:0) buffer that can be uploaded
directly as a device input, plus a manifest recording geometry and a SHA-256
checksum so CI can prove the generation is reproducible.

Conventions (a defined encoder input, not a perceptual master):
  * downscale: integer box-average in 8-bit sRGB (gamma) space;
  * matrix:    BT.709, full range (0-255), round-to-nearest (half to even);
  * chroma:    4:2:0 by 2x2 box average of the full-resolution chroma planes;
  * layout:    Y plane, then interleaved Cb/Cr (NV12).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
from PIL import Image

FULL_4K_WIDTH: int = 3840
FULL_4K_HEIGHT: int = 2160


@dataclasses.dataclass(frozen=True)
class Rung:
    name: str
    factor: int

    @property
    def width(self) -> int:
        return FULL_4K_WIDTH // self.factor

    @property
    def height(self) -> int:
        return FULL_4K_HEIGHT // self.factor


# The 540p rung (960x540) is excluded: 540 is not a multiple of 8, which the
# VarDCT block pipeline requires. Every supported rung is a clean multiple of 8.
LADDER: tuple[Rung, ...] = (
    Rung("2160p", 1),
    Rung("1080p", 2),
    Rung("720p", 3),
)

# BT.709 full-range RGB -> Y'CbCr.
_RGB_TO_Y = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_RGB_TO_CB = np.array([-0.114572, -0.385428, 0.5], dtype=np.float64)
_RGB_TO_CR = np.array([0.5, -0.454153, -0.045847], dtype=np.float64)


def load_rgb(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def box_downscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return rgb
    height, width, channels = rgb.shape
    assert height % factor == 0 and width % factor == 0
    blocks = rgb.reshape(
        height // factor, factor, width // factor, factor, channels
    ).astype(np.float64)
    averaged = blocks.mean(axis=(1, 3))
    return np.rint(averaged).astype(np.uint8)


def _round_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def rgb_to_nv12(rgb: np.ndarray) -> bytes:
    height, width, _ = rgb.shape
    assert height % 2 == 0 and width % 2 == 0
    pixels = rgb.astype(np.float64)

    luma = _round_u8(pixels @ _RGB_TO_Y)
    cb_full = pixels @ _RGB_TO_CB + 128.0
    cr_full = pixels @ _RGB_TO_CR + 128.0

    cb = _round_u8(cb_full.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)))
    cr = _round_u8(cr_full.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)))

    chroma = np.empty((height // 2, width), dtype=np.uint8)
    chroma[:, 0::2] = cb
    chroma[:, 1::2] = cr

    return luma.tobytes() + chroma.tobytes()


@dataclasses.dataclass(frozen=True)
class Asset:
    source: str
    name: str
    rung: str
    width: int
    height: int
    nv12_bytes: int
    sha256: str


def build_asset(source: pathlib.Path, rgb_4k: np.ndarray, rung: Rung,
                out_dir: pathlib.Path) -> Asset:
    scaled = box_downscale(rgb_4k, rung.factor)
    nv12 = rgb_to_nv12(scaled)
    name = f"{source.stem}_{rung.name}.nv12"
    (out_dir / name).write_bytes(nv12)
    return Asset(
        source=source.name,
        name=name,
        rung=rung.name,
        width=rung.width,
        height=rung.height,
        nv12_bytes=len(nv12),
        sha256=hashlib.sha256(nv12).hexdigest(),
    )


def build_corpus(data_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(data_dir.glob("*.png"))
    assets: list[Asset] = []
    for source in sources:
        rgb_4k = load_rgb(source)
        assert rgb_4k.shape[:2] == (FULL_4K_HEIGHT, FULL_4K_WIDTH), (
            f"{source.name}: expected 3840x2160, got "
            f"{rgb_4k.shape[1]}x{rgb_4k.shape[0]}"
        )
        for rung in LADDER:
            assets.append(build_asset(source, rgb_4k, rung, out_dir))
    return {
        "colorspace": "bt709",
        "range": "full",
        "format": "nv12",
        "chroma": "420",
        "assets": [dataclasses.asdict(a) for a in assets],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_corpus(args.data_dir, args.out_dir)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
