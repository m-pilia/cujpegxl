# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Rate-distortion quality benchmark: cujpegxl vs nvJPEG vs libjxl at 4K.

Encodes the 2160p corpus with all three codecs across a quality/distance
ladder, decodes each output back to RGB, and scores perceptual quality
(butteraugli, ssimulacra2) and PSNR against a shared reference. The reference
is the NV12-reconstructed RGB, i.e. the same 4:2:0-subsampled signal the GPU
encoders see, so the comparison isolates codec distortion from the shared
chroma subsampling and gives libjxl no full-chroma advantage. Chroma is
upsampled with the same bilinear kernel the decoders reconstruct, so a lossless
codec scores ~0 rather than paying a chroma-upsampling mismatch floor.

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

FULL_4K_WIDTH: int = cp.FULL_4K_WIDTH
FULL_4K_HEIGHT: int = cp.FULL_4K_HEIGHT

# BT.709 full-range Y'CbCr->RGB, the exact inverse of corpus_prep._RGB_TO_*.
_RGB_FROM_YCBCR = np.linalg.inv(
    np.array(
        [
            cp._RGB_TO_Y,
            cp._RGB_TO_CB,
            cp._RGB_TO_CR,
        ],
        dtype=np.float64,
    )
)

DEFAULT_DISTANCES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_QUALITIES: tuple[int, ...] = (70, 80, 90, 95)


@dataclass
class Row:
    image: str
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


def _linear_upsample_2x(plane: np.ndarray) -> np.ndarray:
    """Upsample a 2D chroma plane 2x per axis to match cujpegxl's decode path.

    Replicates CUDA `cudaFilterModeLinear` texture sampling with clamp
    addressing at the coordinate the fused front-end uses ((g+0.5)*0.5), i.e.
    the half-texel phase B = 0.5*g - 0.25. This is the smooth kernel the decoder
    reconstructs from, so a lossless codec scores ~0 instead of paying the
    nearest-replication mismatch floor (butteraugli ~1-2) that biased the
    perceptual metrics against every GPU codec.
    """
    result = plane.astype(np.float64)
    for axis in (0, 1):
        n = result.shape[axis]
        coord = 0.5 * np.arange(2 * n) - 0.25
        i0 = np.floor(coord).astype(np.intp)
        alpha = (coord - i0).reshape([2 * n if a == axis else 1 for a in range(result.ndim)])
        lo = np.take(result, np.clip(i0, 0, n - 1), axis=axis)
        hi = np.take(result, np.clip(i0 + 1, 0, n - 1), axis=axis)
        result = (1.0 - alpha) * lo + alpha * hi
    return result


def nv12_to_rgb(nv12: bytes | np.ndarray, width: int, height: int) -> np.ndarray:
    """Reconstruct RGB from an NV12 byte stream.

    Upsamples the 4:2:0 chroma with the same bilinear kernel cujpegxl's decode
    path reconstructs (see `_linear_upsample_2x`) and applies the inverse of
    corpus_prep.rgb_to_nv12's BT.709 full-range matrix.
    """
    if isinstance(nv12, np.ndarray):
        buf = np.asarray(nv12, dtype=np.uint8).ravel()
    else:
        buf = np.frombuffer(nv12, dtype=np.uint8)
    luma = buf[: width * height].reshape(height, width).astype(np.float64)
    inter = buf[width * height :].reshape(height // 2, width)
    cb_up = _linear_upsample_2x(inter[:, 0::2])
    cr_up = _linear_upsample_2x(inter[:, 1::2])
    ycbcr = np.stack([luma, cb_up - 128.0, cr_up - 128.0], axis=-1)
    rgb = np.clip(np.rint(ycbcr @ _RGB_FROM_YCBCR.T), 0, 255).astype(np.uint8)
    return rgb


def _score(reference: np.ndarray, distorted: np.ndarray) -> dict:
    return {
        "psnr": pylibjxl.psnr(reference, distorted),
        "butteraugli": pylibjxl.butteraugli(reference, distorted),
        "ssimulacra2": pylibjxl.ssimulacra2(reference, distorted),
    }


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg_bytes)) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def _ensure_4k(rgb: np.ndarray, name: str) -> None:
    if rgb.shape[1] != FULL_4K_WIDTH or rgb.shape[0] != FULL_4K_HEIGHT:
        raise ValueError(
            f"{name}: expected {FULL_4K_WIDTH}x{FULL_4K_HEIGHT}, got "
            f"{rgb.shape[1]}x{rgb.shape[0]}"
        )


def bench_image(
    source: pathlib.Path,
    cfg: BenchConfig,
    cached: dict[tuple[str, str, str], Row] | None = None,
) -> list[Row]:
    name = source.name
    rgb = cp.load_rgb(source)
    _ensure_4k(rgb, name)
    width = FULL_4K_WIDTH
    height = FULL_4K_HEIGHT
    nv12 = cp.rgb_to_nv12(rgb)
    nv12_arr = np.frombuffer(nv12, dtype=np.uint8)
    reference = nv12_to_rgb(nv12, width, height)

    rows: list[Row] = []

    for distance in cfg.distances:
        jxl = pycujpegxl.encode(nv12_arr, width, height, distance, cfg.device)
        decoded = pylibjxl.decode(jxl)
        rows.append(_row(name, "cujpegxl", f"d={distance}", len(jxl), width, height,
                         reference, decoded))

    if cached is None:
        for distance in cfg.distances:
            jxl = pylibjxl.encode(reference, distance)
            decoded = pylibjxl.decode(jxl)
            rows.append(_row(name, "libjxl", f"d={distance}", len(jxl), width, height,
                             reference, decoded))

        for quality in cfg.qualities:
            jpg = pynvjpeg.encode(nv12_arr, width, height, quality, cfg.device)
            decoded = _decode_jpeg(jpg)
            rows.append(_row(name, "nvjpeg", f"q={quality}", len(jpg), width, height,
                             reference, decoded))
    else:
        rows.extend(cached[(name, "libjxl", f"d={d}")] for d in cfg.distances)
        rows.extend(cached[(name, "nvjpeg", f"q={q}")] for q in cfg.qualities)

    return rows


def _row(image: str, codec: str, param: str, coded_bytes: int, width: int, height: int,
         reference: np.ndarray, decoded: np.ndarray) -> Row:
    scores = _score(reference, decoded)
    return Row(
        image=image,
        codec=codec,
        param=param,
        coded_bytes=coded_bytes,
        bpp=bits_per_pixel(coded_bytes, width, height),
        ratio=compression_ratio(coded_bytes, width, height),
        **scores,
    )


def _load_cached_rows(
    path: pathlib.Path, frames: list[pathlib.Path], cfg: BenchConfig
) -> dict[tuple[str, str, str], Row]:
    cache = {
        (row["image"], row["codec"], row["param"]): Row(**row)
        for row in json.loads(path.read_text())
        if row["codec"] != "cujpegxl"
    }
    missing = [
        f"{frame.name} {codec} {param}"
        for frame in frames
        for codec, param in [
            *(("libjxl", f"d={distance}") for distance in cfg.distances),
            *(("nvjpeg", f"q={quality}") for quality in cfg.qualities),
        ]
        if (frame.name, codec, param) not in cache
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


def _data_frames(data_dir: pathlib.Path) -> list[pathlib.Path]:
    frames = sorted(data_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no *.png frames found under {data_dir}")
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory with source PNG frames (default: data).",
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
    frames = _data_frames(args.data_dir)
    cached = (
        _load_cached_rows(args.cached_input, frames, cfg)
        if args.cached_input is not None
        else None
    )

    all_rows: list[Row] = []
    for source in frames:
        all_rows.extend(bench_image(source, cfg, cached))

    print(_format_table(all_rows))

    if args.output is not None:
        args.output.write_text(
            json.dumps([r.as_dict() for r in all_rows], indent=2) + "\n"
        )
        print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
