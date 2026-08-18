# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the quality benchmark helpers and the pylibjxl metric contract.

GPU-free: only the helpers and the CPU-side libjxl metrics/round-trip are
exercised. The codec encode paths (cujpegxl, nvJPEG) are not driven here.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

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


class BitsPerPixelTest(unittest.TestCase):
    def test_bpp_and_ratio_are_consistent(self) -> None:
        width, height = 3840, 2160
        coded = 1_000_000
        self.assertEqual(
            qb.bits_per_pixel(coded, width, height), 8.0 * coded / (width * height)
        )
        self.assertEqual(
            qb.compression_ratio(coded, width, height), (width * height * 3) / coded
        )
        bpp = qb.bits_per_pixel(coded, width, height)
        self.assertLess(abs(bpp - 24.0 / qb.compression_ratio(coded, width, height)), 1e-9)


class Nv12ToRgbTest(unittest.TestCase):
    def test_round_trips_luma_and_stays_bounded(self) -> None:
        rgb = _gradient(64, 64)
        nv12 = cp.rgb_to_nv12(rgb)
        recon = qb.nv12_to_rgb(nv12, 64, 64)
        self.assertEqual(recon.shape, (64, 64, 3))
        self.assertEqual(recon.dtype, np.uint8)
        luma = (rgb.astype(np.float64) @ cp._RGB_TO_Y).astype(np.uint8)
        recon_luma = (recon.astype(np.float64) @ cp._RGB_TO_Y).astype(np.uint8)
        self.assertLess(np.mean(np.abs(luma.astype(int) - recon_luma.astype(int))), 1.0)
        self.assertLess(recon.mean(), 256)
        self.assertGreaterEqual(recon.min(), 0)

    def test_half_resolution_frame(self) -> None:
        rgb = _gradient(96, 48)
        nv12 = cp.rgb_to_nv12(rgb)
        recon = qb.nv12_to_rgb(nv12, 96, 48)
        self.assertEqual(recon.shape, (48, 96, 3))
        luma = (rgb.astype(np.float64) @ cp._RGB_TO_Y).astype(np.uint8)
        recon_luma = (recon.astype(np.float64) @ cp._RGB_TO_Y).astype(np.uint8)
        self.assertLess(np.mean(np.abs(luma.astype(int) - recon_luma.astype(int))), 1.0)


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


class DecodeJpegTest(unittest.TestCase):
    def test_uses_bt709_not_jfif_bt601(self) -> None:
        from io import BytesIO

        from PIL import Image

        rgb = _gradient(64, 64)
        jpeg = _jpeg_from_bt709_ycbcr(rgb)

        recovered = qb._decode_jpeg(jpeg)
        self.assertEqual(recovered.shape, rgb.shape)
        self.assertEqual(recovered.dtype, np.uint8)

        with Image.open(BytesIO(jpeg)) as handle:
            jfif_rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)

        bt709_err = np.mean(np.abs(recovered.astype(int) - rgb.astype(int)))
        bt601_err = np.mean(np.abs(jfif_rgb.astype(int) - rgb.astype(int)))
        self.assertLess(bt709_err, 2.0)
        self.assertGreater(bt601_err, 3.0 * bt709_err)


class MetricContractTest(unittest.TestCase):
    def test_identical_images_score_as_a_perfect_match(self) -> None:
        img = _gradient(64, 64)
        self.assertAlmostEqual(pylibjxl.psnr(img, img), 99.0)
        self.assertLess(pylibjxl.butteraugli(img, img), 0.05)
        self.assertGreater(pylibjxl.ssimulacra2(img, img), 95.0)

    def test_scores_worsen_monotonically_with_distortion(self) -> None:
        original = _gradient(64, 64)
        mild = np.clip(original.astype(np.int16) + 12, 0, 255).astype(np.uint8)
        severe = np.clip(original.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        self.assertGreater(
            pylibjxl.psnr(original, mild), pylibjxl.psnr(original, severe)
        )
        self.assertLess(
            pylibjxl.butteraugli(original, mild),
            pylibjxl.butteraugli(original, severe),
        )
        self.assertGreater(
            pylibjxl.ssimulacra2(original, mild),
            pylibjxl.ssimulacra2(original, severe),
        )

    def test_jxl_encode_decode_round_trips_through_libjxl(self) -> None:
        img = _gradient(64, 64)
        encoded = pylibjxl.encode(img, 1.0)
        self.assertIsInstance(encoded, (bytes, bytearray))
        self.assertGreater(len(encoded), 0)
        decoded = pylibjxl.decode(encoded)
        self.assertEqual(decoded.shape, img.shape)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertGreater(pylibjxl.psnr(img, decoded), 30.0)


class RungForTest(unittest.TestCase):
    def test_maps_ladder_names(self) -> None:
        rung = qb._rung_for("1080p")
        self.assertEqual((rung.width, rung.height), (1920, 1080))
        self.assertEqual(rung.factor, 2)
        self.assertEqual(qb._rung_for("2160p").factor, 1)

    def test_rejects_unknown_resolution(self) -> None:
        with self.assertRaises(ValueError):
            qb._rung_for("540p")


class EnsureShapeTest(unittest.TestCase):
    def test_rejects_wrong_geometry(self) -> None:
        rgb = np.zeros((54, 96, 3), dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            qb._ensure_shape(rgb, cp.FULL_4K_WIDTH, cp.FULL_4K_HEIGHT, "img.png")
        self.assertIn("expected 3840x2160", str(ctx.exception))
        self.assertIn("96x54", str(ctx.exception))
        qb._ensure_shape(rgb, 96, 54, "img.png")


def _row_dict(
    image: str, codec: str, param: str, resolution: str | None = "2160p"
) -> dict:
    row = {
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
    if resolution is not None:
        row["resolution"] = resolution
    return row


class LoadCachedRowsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)
        self.cfg = qb.BenchConfig(distances=(0.5,), qualities=(70,))
        self.frames = [self.tmp / "a.png"]

    def _write_cache(self, rows: list[dict]) -> pathlib.Path:
        cache_path = self.tmp / "cache.json"
        cache_path.write_text(json.dumps(rows))
        return cache_path

    def test_skips_cujpegxl_and_round_trips_fields(self):
        cache_path = self._write_cache([
            _row_dict("a.png", "cujpegxl", "d=0.5"),
            _row_dict("a.png", "libjxl", "d=0.5"),
            _row_dict("a.png", "nvjpeg", "q=70"),
        ])
        cache = qb._load_cached_rows(cache_path, self.frames, self.cfg, "2160p")
        self.assertEqual(
            set(cache),
            {("2160p", "a.png", "libjxl", "d=0.5"), ("2160p", "a.png", "nvjpeg", "q=70")},
        )
        self.assertEqual(
            cache[("2160p", "a.png", "libjxl", "d=0.5")],
            qb.Row(
                image="a.png",
                resolution="2160p",
                codec="libjxl",
                param="d=0.5",
                coded_bytes=1_000,
                bpp=1.0,
                ratio=8.0,
                psnr=40.0,
                butteraugli=1.0,
                ssimulacra2=90.0,
            ),
        )

    def test_keys_on_resolution(self):
        cache_path = self._write_cache([
            _row_dict("a.png", "libjxl", "d=0.5"),
            _row_dict("a.png", "libjxl", "d=0.5", resolution="1080p"),
            _row_dict("a.png", "nvjpeg", "q=70", resolution="1080p"),
        ])
        cache = qb._load_cached_rows(cache_path, self.frames, self.cfg, "1080p")
        self.assertEqual(
            set(cache),
            {
                ("2160p", "a.png", "libjxl", "d=0.5"),
                ("1080p", "a.png", "libjxl", "d=0.5"),
                ("1080p", "a.png", "nvjpeg", "q=70"),
            },
        )

    def test_rejects_rows_without_resolution_field(self):
        cache_path = self._write_cache([
            _row_dict("a.png", "libjxl", "d=0.5", resolution=None),
            _row_dict("a.png", "nvjpeg", "q=70"),
        ])
        with self.assertRaises(ValueError) as ctx:
            qb._load_cached_rows(cache_path, self.frames, self.cfg, "2160p")
        self.assertIn("no resolution field", str(ctx.exception))
        self.assertIn("a.png libjxl d=0.5", str(ctx.exception))

    def test_rejects_cache_missing_requested_entries(self):
        cache_path = self._write_cache([
            _row_dict("a.png", "libjxl", "d=0.5"),
            _row_dict("a.png", "nvjpeg", "q=70"),
        ])
        cfg = qb.BenchConfig(distances=(0.5, 1.0), qualities=(70,))
        with self.assertRaises(ValueError) as ctx:
            qb._load_cached_rows(cache_path, self.frames, cfg, "2160p")
        self.assertIn("d=1.0", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            qb._load_cached_rows(
                cache_path, [self.tmp / "b.png"], cfg, "2160p"
            )
        self.assertIn("b.png", str(ctx.exception))

    def test_rejects_cache_lacking_requested_resolution(self):
        cache_path = self._write_cache([
            _row_dict("a.png", "libjxl", "d=0.5"),
            _row_dict("a.png", "nvjpeg", "q=70", resolution="1080p"),
        ])
        with self.assertRaises(ValueError) as ctx:
            qb._load_cached_rows(cache_path, self.frames, self.cfg, "2160p")
        self.assertIn("2160p a.png nvjpeg q=70", str(ctx.exception))


class RowDictTest(unittest.TestCase):
    def test_as_dict_includes_resolution(self):
        row = qb.Row(
            image="a.png",
            resolution="1080p",
            codec="cujpegxl",
            param="d=1.0",
            coded_bytes=10,
            bpp=1.0,
            ratio=8.0,
            psnr=40.0,
            butteraugli=1.0,
            ssimulacra2=90.0,
        )
        self.assertEqual(row.as_dict()["resolution"], "1080p")


if __name__ == "__main__":
    unittest.main()
