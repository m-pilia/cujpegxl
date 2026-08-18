# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the quality benchmark helpers and the pylibjxl metric contract.

GPU-free: only the helpers and the CPU-side libjxl metrics/round-trip are
exercised. The codec encode paths (cujpegxl, nvJPEG) are not driven here.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
from collections.abc import Iterator

import numpy as np

import corpus_prep as cp
import pylibjxl

import quality_benchmark as qb


def _gradient(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    r = (x * 3) & 0xFF
    g = (y * 5) & 0xFF
    b = ((x + y) * 2) & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def test_bits_per_pixel_and_ratio_are_consistent() -> None:
    width, height = 3840, 2160
    coded = 1_000_000
    assert qb.bits_per_pixel(coded, width, height) == 8.0 * coded / (width * height)
    assert qb.compression_ratio(coded, width, height) == (width * height * 3) / coded
    bpp = qb.bits_per_pixel(coded, width, height)
    assert abs(bpp - 8.0 / qb.compression_ratio(coded, width, height)) < 1e-9


def test_nv12_to_rgb_round_trips_luma_and_stays_bounded() -> None:
    rgb = _gradient(64, 64)
    nv12 = cp.rgb_to_nv12(rgb)
    recon = qb.nv12_to_rgb(nv12, 64, 64)
    assert recon.shape == (64, 64, 3)
    assert recon.dtype == np.uint8
    luma = (rgb.astype(np.float64) @ cp._RGB_TO_Y).astype(np.uint8)
    recon_luma = (recon.astype(np.float64) @ cp._RGB_TO_Y).astype(np.uint8)
    assert np.mean(np.abs(luma.astype(int) - recon_luma.astype(int))) < 1.0
    assert recon.mean() < 256 and recon.min() >= 0


def _jpeg_from_bt709_ycbcr(rgb: np.ndarray) -> bytes:
    """Emulate nvJPEG: store BT.709 full-range Y'CbCr in a bare JFIF stream."""
    from io import BytesIO

    from PIL import Image

    pixels = rgb.astype(np.float64)
    y = np.rint(pixels @ cp._RGB_TO_Y)
    cb = np.rint(pixels @ cp._RGB_TO_CB + 128.0)
    cr = np.rint(pixels @ cp._RGB_TO_CR + 128.0)
    ycbcr = np.clip(np.stack([y, cb, cr], axis=-1), 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(ycbcr, mode="YCbCr").save(buf, "JPEG", quality=100, subsampling=0)
    return buf.getvalue()


def test_decode_jpeg_uses_bt709_not_jfif_bt601() -> None:
    rgb = _gradient(64, 64)
    jpeg = _jpeg_from_bt709_ycbcr(rgb)

    recovered = qb._decode_jpeg(jpeg)
    assert recovered.shape == rgb.shape
    assert recovered.dtype == np.uint8

    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(jpeg)) as handle:
        jfif_rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)

    bt709_err = np.mean(np.abs(recovered.astype(int) - rgb.astype(int)))
    bt601_err = np.mean(np.abs(jfif_rgb.astype(int) - rgb.astype(int)))
    assert bt709_err < 2.0
    assert bt601_err > 3.0 * bt709_err


def test_identical_images_score_as_a_perfect_match() -> None:
    img = _gradient(64, 64)
    assert abs(pylibjxl.psnr(img, img) - 99.0) < 1e-9
    assert pylibjxl.butteraugli(img, img) < 0.05
    assert pylibjxl.ssimulacra2(img, img) > 95.0


def test_metric_scores_worsen_monotonically_with_distortion() -> None:
    original = _gradient(64, 64)
    mild = np.clip(original.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    severe = np.clip(original.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    assert pylibjxl.psnr(original, mild) > pylibjxl.psnr(original, severe)
    assert pylibjxl.butteraugli(original, mild) < pylibjxl.butteraugli(original, severe)
    assert pylibjxl.ssimulacra2(original, mild) > pylibjxl.ssimulacra2(original, severe)


def test_jxl_encode_decode_round_trips_through_libjxl() -> None:
    img = _gradient(64, 64)
    encoded = pylibjxl.encode(img, 1.0)
    assert isinstance(encoded, (bytes, bytearray))
    assert len(encoded) > 0
    decoded = pylibjxl.decode(encoded)
    assert decoded.shape == img.shape
    assert decoded.dtype == np.uint8
    assert pylibjxl.psnr(img, decoded) > 30.0


def _row_dict(image: str, codec: str, param: str) -> dict:
    return {
        "image": image,
        "codec": codec,
        "param": param,
        "coded_bytes": 1_000,
        "bpp": 1.0,
        "ratio": 8.0,
        "psnr": 40.0,
        "butteraugli": 1.0,
        "ssimulacra2": 90.0,
    }


@contextlib.contextmanager
def _expect_value_error(*needles: str) -> Iterator[None]:
    try:
        yield
    except ValueError as err:
        for needle in needles:
            assert needle in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_load_cached_rows_skips_cujpegxl_and_round_trips_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = pathlib.Path(tmp) / "cache.json"
        cache_path.write_text(
            json.dumps([
                _row_dict("a.png", "cujpegxl", "d=0.5"),
                _row_dict("a.png", "libjxl", "d=0.5"),
                _row_dict("a.png", "nvjpeg", "q=70"),
            ])
        )
        cfg = qb.BenchConfig(distances=(0.5,), qualities=(70,))
        cache = qb._load_cached_rows(cache_path, [pathlib.Path(tmp) / "a.png"], cfg)
    assert set(cache) == {("a.png", "libjxl", "d=0.5"), ("a.png", "nvjpeg", "q=70")}
    assert cache[("a.png", "libjxl", "d=0.5")] == qb.Row(
        image="a.png",
        codec="libjxl",
        param="d=0.5",
        coded_bytes=1_000,
        bpp=1.0,
        ratio=8.0,
        psnr=40.0,
        butteraugli=1.0,
        ssimulacra2=90.0,
    )


def test_load_cached_rows_rejects_cache_missing_requested_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = pathlib.Path(tmp) / "cache.json"
        cache_path.write_text(
            json.dumps([
                _row_dict("a.png", "libjxl", "d=0.5"),
                _row_dict("a.png", "nvjpeg", "q=70"),
            ])
        )
        cfg = qb.BenchConfig(distances=(0.5, 1.0), qualities=(70,))
        with _expect_value_error("d=1.0"):
            qb._load_cached_rows(cache_path, [pathlib.Path(tmp) / "a.png"], cfg)
        with _expect_value_error("b.png"):
            qb._load_cached_rows(cache_path, [pathlib.Path(tmp) / "b.png"], cfg)
