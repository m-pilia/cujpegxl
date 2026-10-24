# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Rate-distortion quality benchmark: cujpegxl vs nvJPEG vs libjxl.

Encodes the corpus at a --resolution rung (2160p by default, 1080p for the
eval corpus) with all three codecs across a quality/distance ladder, decodes
each output back to RGB, and scores perceptual quality (butteraugli,
ssimulacra2) and PSNR against a shared reference. Inputs are always 4K master
PNGs; the rung frame is derived with the same integer box downscale the corpus
uses. The reference is the NV12-reconstructed RGB, i.e. the same
4:2:0-subsampled signal the GPU encoders see, so the comparison isolates codec
distortion from the shared chroma subsampling and gives libjxl no full-chroma
advantage. Chroma is upsampled with the same bilinear kernel the decoders
reconstruct, so a lossless codec scores ~0 rather than paying a
chroma-upsampling mismatch floor.

cujpegxl and libjxl share the butteraugli-distance axis (so they are directly
comparable at a given distance); nvJPEG's quality axis is reported as its own
curve. Compression ratio uses the raw RGB byte count (W*H*3) as the baseline so
it is consistent across codecs and independent of PNG's lossless compression.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

import corpus_prep as cp
import pycujpegxl
import pylibjxl
import pynvjpeg

RESOLUTIONS = ("1080p", "2160p")

DEFAULT_DISTANCES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_QUALITIES: tuple[int, ...] = (70, 80, 90, 95)


@dataclass
class Row:
    image: str
    resolution: str
    codec: str
    param: str
    coded_bytes: int
    bpp: float
    ratio: float
    psnr: float
    butteraugli: float
    ssimulacra2: float

    def as_dict(self) -> dict:
        return {
            "image": self.image,
            "resolution": self.resolution,
            "codec": self.codec,
            "param": self.param,
            "coded_bytes": self.coded_bytes,
            "bpp": self.bpp,
            "ratio": self.ratio,
            "psnr": self.psnr,
            "butteraugli": self.butteraugli,
            "ssimulacra2": self.ssimulacra2,
        }


@dataclass
class BenchConfig:
    distances: tuple[float, ...] = DEFAULT_DISTANCES
    qualities: tuple[int, ...] = DEFAULT_QUALITIES
    device: int = 0


def bits_per_pixel(coded_bytes: int, width: int, height: int) -> float:
    return 8.0 * coded_bytes / (width * height)


def compression_ratio(coded_bytes: int, width: int, height: int) -> float:
    return (width * height * 3) / coded_bytes


def _ycbcr_to_rgb(ycbcr_centered: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(ycbcr_centered @ cp.RGB_FROM_YCBCR.T), 0, 255).astype(np.uint8)


def _score(reference: np.ndarray, distorted: np.ndarray) -> dict:
    return {
        "psnr": pylibjxl.psnr(reference, distorted),
        "butteraugli": pylibjxl.butteraugli(reference, distorted),
        "ssimulacra2": pylibjxl.ssimulacra2(reference, distorted),
    }


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode nvJPEG output to RGB on the pipeline's BT.709 full-range convention.

    nvJPEG emits a bare JFIF stream, so libjpeg's default RGB decode applies the
    JFIF/BT.601 inverse. But the stored Y'CbCr was produced by corpus_prep's
    BT.709 full-range matrix, so that mismatch imposes a fixed chroma rotation on
    every pixel (a butteraugli floor independent of quality). `draft("YCbCr")`
    makes libjpeg hand back the raw stored samples without color conversion, and
    inverting them with _RGB_FROM_YCBCR keeps nvJPEG on the same convention as
    the reference and the other codecs.
    """
    with Image.open(io.BytesIO(jpeg_bytes)) as handle:
        handle.draft("YCbCr", handle.size)
        if handle.mode != "YCbCr":
            raise RuntimeError(
                f"expected raw YCbCr decode, got mode {handle.mode!r}; the BT.709 "
                "inverse would be applied to the wrong color space"
            )
        ycbcr = np.asarray(handle, dtype=np.float64)
    ycbcr[..., 1:] -= 128.0
    return _ycbcr_to_rgb(ycbcr)


def ensure_shape(rgb: np.ndarray, width: int, height: int, name: str) -> None:
    if rgb.shape[1] != width or rgb.shape[0] != height:
        raise ValueError(
            f"{name}: expected {width}x{height}, got "
            f"{rgb.shape[1]}x{rgb.shape[0]}"
        )


def bench_image(
    source: pathlib.Path,
    rung: cp.Rung,
    cfg: BenchConfig,
    cached: dict[tuple[str, str, str, str], Row] | None = None,
) -> list[Row]:
    name = source.name
    rgb = cp.load_rgb(source)
    ensure_shape(rgb, cp.FULL_4K_WIDTH, cp.FULL_4K_HEIGHT, name)
    width = rung.width
    height = rung.height
    nv12 = cp.rgb_to_nv12(cp.box_downscale(rgb, rung.factor))
    nv12_arr = np.frombuffer(nv12, dtype=np.uint8)
    reference = cp.nv12_to_rgb(nv12, width, height)

    rows: list[Row] = []

    for distance in cfg.distances:
        jxl = pycujpegxl.encode(nv12_arr, width, height, distance, cfg.device)
        decoded = pylibjxl.decode(jxl)
        rows.append(_row(name, rung.name, "cujpegxl", f"d={distance}", len(jxl),
                         width, height, reference, decoded))

    if cached is None:
        for distance in cfg.distances:
            jxl = pylibjxl.encode(reference, distance)
            decoded = pylibjxl.decode(jxl)
            rows.append(_row(name, rung.name, "libjxl", f"d={distance}", len(jxl),
                             width, height, reference, decoded))

        for quality in cfg.qualities:
            jpg = pynvjpeg.encode(nv12_arr, width, height, quality, cfg.device)
            decoded = decode_jpeg(jpg)
            rows.append(_row(name, rung.name, "nvjpeg", f"q={quality}", len(jpg),
                             width, height, reference, decoded))
    else:
        rows.extend(cached[(rung.name, name, "libjxl", f"d={d}")] for d in cfg.distances)
        rows.extend(cached[(rung.name, name, "nvjpeg", f"q={q}")] for q in cfg.qualities)

    return rows


def _row(image: str, resolution: str, codec: str, param: str, coded_bytes: int,
         width: int, height: int, reference: np.ndarray, decoded: np.ndarray) -> Row:
    scores = _score(reference, decoded)
    return Row(
        image=image,
        resolution=resolution,
        codec=codec,
        param=param,
        coded_bytes=coded_bytes,
        bpp=bits_per_pixel(coded_bytes, width, height),
        ratio=compression_ratio(coded_bytes, width, height),
        **scores,
    )


def load_cached_rows(
    path: pathlib.Path, frames: list[pathlib.Path], cfg: BenchConfig, resolution: str
) -> dict[tuple[str, str, str, str], Row]:
    cache: dict[tuple[str, str, str, str], Row] = {}
    for raw in json.loads(path.read_text()):
        if raw["codec"] == "cujpegxl":
            continue
        if "resolution" not in raw:
            raise ValueError(
                f"{path}: row {raw['image']} {raw['codec']} {raw['param']} has no "
                "resolution field; the cache predates per-resolution rows and "
                "must be regenerated without --cached-input."
            )
        row = Row(**raw)
        cache[(row.resolution, row.image, row.codec, row.param)] = row
    missing = [
        f"{resolution} {frame.name} {codec} {param}"
        for frame in frames
        for codec, param in [
            *(("libjxl", f"d={distance}") for distance in cfg.distances),
            *(("nvjpeg", f"q={quality}") for quality in cfg.qualities),
        ]
        if (resolution, frame.name, codec, param) not in cache
    ]
    if missing:
        raise ValueError(
            f"{path} has no reference rows for: {', '.join(missing)}. "
            "Regenerate the cache without --cached-input."
        )
    return cache


def _format_table(rows: list[Row]) -> str:
    header = (
        f"{'image':<22} {'codec':<9} {'param':<8} {'bytes':>10} "
        f"{'bpp':>6} {'ratio':>6} {'psnr':>6} {'butter':>7} {'ssim2':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.image[:22]:<22} {r.codec:<9} {r.param:<8} {r.coded_bytes:>10} "
            f"{r.bpp:>6.2f} {r.ratio:>6.1f} {r.psnr:>6.2f} "
            f"{r.butteraugli:>7.3f} {r.ssimulacra2:>7.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory with source PNG frames (default: data).",
    )
    parser.add_argument(
        "--resolution",
        choices=RESOLUTIONS,
        default="2160p",
        help="Corpus rung to benchmark; masters are 4K PNGs downscaled to it.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISTANCES),
        help="Butteraugli distances for cujpegxl and libjxl.",
    )
    parser.add_argument(
        "--qualities",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUALITIES),
        help="JPEG qualities for nvJPEG.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Write the results as JSON to this path.",
    )
    parser.add_argument(
        "--cached-input",
        type=pathlib.Path,
        default=None,
        help="JSON file previously written with --output; reuse its libjxl and "
        "nvJPEG rows (must cover the requested images, distances, and "
        "qualities) and only recompute cujpegxl.",
    )
    args = parser.parse_args(argv)

    cfg = BenchConfig(
        distances=tuple(args.distances),
        qualities=tuple(args.qualities),
        device=args.device,
    )
    rung = cp.rung_by_name(args.resolution)
    frames = cp.data_frames(args.data_dir)
    cached = (
        load_cached_rows(args.cached_input, frames, cfg, rung.name)
        if args.cached_input is not None
        else None
    )

    all_rows: list[Row] = []
    for source in frames:
        all_rows.extend(bench_image(source, rung, cfg, cached))

    print(_format_table(all_rows))

    if args.output is not None:
        args.output.write_text(
            json.dumps([r.as_dict() for r in all_rows], indent=2) + "\n"
        )
        print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
