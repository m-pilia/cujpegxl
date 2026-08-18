# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Network-free tests for prepare_dataset.py (downloads use file:// or fakes)."""

from __future__ import annotations

import hashlib
import email.message
import json
import pathlib
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

import numpy as np
from PIL import Image

import corpus_prep as cp

import prepare_dataset as pd


def _entry(name: str, path: pathlib.Path, mime: str = "image/png") -> dict:
    return {
        "name": name,
        "page": f"https://commons.wikimedia.org/wiki/File:{name}",
        "url": path.as_uri(),
        "width": 64,
        "height": 36,
        "mime": mime,
        "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        "author": "Test Author",
        "license": "CC BY-SA 4.0",
    }


def _write_png(path: pathlib.Path, width: int, height: int) -> None:
    y, x = np.mgrid[0:height, 0:width]
    blue = np.full_like(x, 128)
    rgb = np.stack([(x * 7) & 0xFF, (y * 5) & 0xFF, blue], axis=-1).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


class CachePathTest(unittest.TestCase):
    def test_extension_from_mime(self):
        entry = {"name": "img", "mime": "image/tiff"}
        self.assertEqual(
            pd.cache_path(entry, pathlib.Path("/cache")),
            pathlib.Path("/cache/img.tif"),
        )
        entry = {"name": "img", "mime": "image/jpeg"}
        self.assertEqual(
            pd.cache_path(entry, pathlib.Path("/cache")),
            pathlib.Path("/cache/img.jpg"),
        )


class Crop169Test(unittest.TestCase):
    def test_exact_aspect_is_unchanged(self):
        image = Image.new("RGB", (160, 90), "red")
        self.assertEqual(pd._crop_16_9(image).size, (160, 90))

    def test_wider_trims_sides_centered(self):
        image = Image.new("RGB", (200, 90), "green")
        for x in range(20):
            for y in range(90):
                image.putpixel((x, y), (255, 0, 0))
                image.putpixel((199 - x, y), (0, 0, 255))
        cropped = pd._crop_16_9(image)
        self.assertEqual(cropped.size, (160, 90))
        self.assertLess(cropped.getpixel((0, 45))[0], 128)
        self.assertEqual(cropped.getpixel((80, 45)), (0, 128, 0))

    def test_taller_trims_top_and_bottom_centered(self):
        image = Image.new("RGB", (160, 100), "green")
        for y in range(5):
            for x in range(160):
                image.putpixel((x, y), (255, 0, 0))
                image.putpixel((x, 99 - y), (0, 0, 255))
        cropped = pd._crop_16_9(image)
        self.assertEqual(cropped.size, (160, 90))
        self.assertLess(cropped.getpixel((80, 0))[0], 128)
        self.assertEqual(cropped.getpixel((80, 45)), (0, 128, 0))


class DownloadOriginalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.payload = b"png-bytes-0123456789"
        self.source = self.dir / "src.png"
        self.source.write_bytes(self.payload)
        self.entry = _entry("file", self.source)

    def test_downloads_verifies_and_skips_on_second_call(self):
        calls = []

        def fetch(url):
            calls.append(url)
            return self.payload

        cache_dir = self.dir / "cache"
        with mock.patch("time.sleep"):
            first = pd.download_original(self.entry, cache_dir, pd._RateLimiter(), fetch)
            second = pd.download_original(self.entry, cache_dir, pd._RateLimiter(), fetch)
        self.assertEqual(first.read_bytes(), self.payload)
        self.assertEqual(second, first)
        self.assertEqual(len(calls), 1)

    def test_sha1_mismatch_is_a_hard_error(self):
        def fetch(url):
            return b"corrupted"

        with self.assertRaises(RuntimeError) as ctx:
            pd.download_original(self.entry, self.dir / "cache", pd._RateLimiter(), fetch)
        self.assertIn("sha1 mismatch", str(ctx.exception))

    def test_retries_transient_failures(self):
        attempts = []

        def fetch(url):
            attempts.append(url)
            if len(attempts) < 3:
                raise urllib.error.URLError("transient")
            return self.payload

        with mock.patch("time.sleep"):
            dest = pd.download_original(self.entry, self.dir / "cache", pd._RateLimiter(), fetch)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(dest.read_bytes(), self.payload)


class RateLimitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.payload = b"png-bytes-0123456789"
        self.source = self.dir / "src.png"
        self.source.write_bytes(self.payload)
        self.entry = _entry("file", self.source)

    @staticmethod
    def _http_error(code, retry_after=None):
        headers = email.message.Message()
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        return urllib.error.HTTPError("url", code, "err", headers, None)

    def test_backoff_honors_retry_after_header(self):
        attempts = []

        def fetch(url):
            attempts.append(url)
            if len(attempts) < 3:
                raise self._http_error(429, retry_after="42")
            return self.payload

        sleeps = []
        with mock.patch("time.sleep", side_effect=sleeps.append):
            pd.download_original(self.entry, self.dir / "cache", pd._RateLimiter(), fetch)
        # Retry-After is a floor: attempt 0 sleeps 42 (> 30), attempt 1 sleeps 60 (> 42).
        self.assertEqual(sleeps, [42.0, 60.0, pd.DOWNLOAD_PAUSE_S])

    def test_consecutive_rate_limits_trip_cooldown(self):
        attempts = []

        def fetch(url):
            attempts.append(url)
            if len(attempts) < 4:
                raise self._http_error(429)
            return self.payload

        sleeps = []
        with mock.patch("time.sleep", side_effect=sleeps.append):
            pd.download_original(self.entry, self.dir / "cache", pd._RateLimiter(), fetch)
        self.assertEqual(
            sleeps,
            [
                pd.BACKOFF_BASE_S,
                2 * pd.BACKOFF_BASE_S,
                4 * pd.BACKOFF_BASE_S + pd.RATE_LIMIT_COOLDOWN_S,
                pd.DOWNLOAD_PAUSE_S,
            ],
        )

    def test_streak_is_shared_and_resets_on_success(self):
        limiter = pd._RateLimiter()
        self.assertEqual(limiter.note_rate_limit(), 0.0)
        self.assertEqual(limiter.note_rate_limit(), 0.0)
        limiter.note_success()
        self.assertEqual(limiter.note_rate_limit(), 0.0)
        self.assertEqual(limiter.note_rate_limit(), 0.0)
        self.assertEqual(limiter.note_rate_limit(), pd.RATE_LIMIT_COOLDOWN_S)
        self.assertEqual(limiter.note_rate_limit(), 0.0)


class BuildMasterTest(unittest.TestCase):
    def test_produces_4k_master_honoring_exif_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "portrait.jpg"
            image = Image.new("RGB", (90, 200), (255, 0, 0))
            for y in range(200):
                for x in range(45, 90):
                    image.putpixel((x, y), (0, 0, 255))
            exif = Image.Exif()
            exif[274] = 6  # Orientation: rotate 90 CW for display.
            image.save(source, format="JPEG", exif=exif, quality=95, subsampling=0)

            master = root / "masters" / "portrait.png"
            master.parent.mkdir()
            pd.build_master(source, master)

            with Image.open(master) as handle:
                self.assertEqual(handle.size, (3840, 2160))
                top = handle.getpixel((1920, 200))
                bottom = handle.getpixel((1920, 1960))
            # After the 90-degree rotation the red half must be on top.
            self.assertGreater(top[0], 200)
            self.assertLess(top[2], 64)
            self.assertGreater(bottom[2], 200)
            self.assertLess(bottom[0], 64)


class ReusableAssetTest(unittest.TestCase):
    def _manifest_entry(self, **overrides):
        entry = {
            "source": "img.png",
            "name": "img_1080p.nv12",
            "rung": "1080p",
            "width": 1920,
            "height": 1080,
            "nv12_bytes": 3110400,
            "sha256": "0" * 64,
        }
        entry.update(overrides)
        return entry

    def test_reuses_when_file_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            frame = out_dir / "img_1080p.nv12"
            payload = b"\x01\x02\x03"
            frame.write_bytes(payload)
            entry = self._manifest_entry(sha256=hashlib.sha256(payload).hexdigest())
            asset = pd._reusable_asset([entry], "img.png", pd._rung_for("1080p"), out_dir)
            self.assertEqual(asset, cp.Asset(**entry))

    def test_rejects_stale_or_foreign_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            frame = out_dir / "img_1080p.nv12"
            frame.write_bytes(b"stale")
            self.assertIsNone(
                pd._reusable_asset(
                    [self._manifest_entry()], "img.png", pd._rung_for("1080p"), out_dir
                )
            )
            self.assertIsNone(
                pd._reusable_asset(
                    [self._manifest_entry()], "other.png", pd._rung_for("1080p"), out_dir
                )
            )
            self.assertIsNone(
                pd._reusable_asset([], "img.png", pd._rung_for("1080p"), out_dir)
            )


class PrepareTest(unittest.TestCase):
    """End-to-end pipeline over a file:// source, both rungs, idempotency."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.source = self.root / "tiny.png"
        _write_png(self.source, 64, 36)
        self.entry = _entry("tiny", self.source)
        self.dirs = {
            "cache_dir": self.root / "cache",
            "masters_dir": self.root / "masters",
            "out_dir": self.root / "out",
        }

    def _prepare(self, resolution):
        return pd.prepare(
            [self.entry],
            resolution,
            sources_ref={"path": "sources.json", "sha256": "0" * 64},
            **self.dirs,
        )

    def test_full_pipeline_1080p_then_2160p_merges_manifest(self):
        doc_1080 = self._prepare("1080p")
        rungs = {a["rung"] for a in doc_1080["assets"]}
        self.assertEqual(rungs, {"1080p"})
        frame_1080 = self.dirs["out_dir"] / "tiny_1080p.nv12"
        self.assertEqual(frame_1080.stat().st_size, 1920 * 1080 * 3 // 2)
        entry_1080 = doc_1080["assets"][0]
        self.assertEqual(
            entry_1080["sha256"],
            hashlib.sha256(frame_1080.read_bytes()).hexdigest(),
        )

        doc_2160 = self._prepare("2160p")
        by_rung = {a["rung"]: a for a in doc_2160["assets"]}
        self.assertEqual(set(by_rung), {"1080p", "2160p"})
        self.assertEqual(by_rung["2160p"]["width"], 3840)
        self.assertEqual(by_rung["2160p"]["height"], 2160)
        self.assertEqual(
            (self.dirs["out_dir"] / "tiny_2160p.nv12").stat().st_size,
            3840 * 2160 * 3 // 2,
        )
        for key in ("colorspace", "range", "format", "chroma", "sources", "versions"):
            self.assertIn(key, doc_2160)
        self.assertEqual(doc_2160["colorspace"], "bt709")
        self.assertEqual(doc_2160["format"], "nv12")

    def test_second_run_reuses_outputs(self):
        self._prepare("1080p")
        frame = self.dirs["out_dir"] / "tiny_1080p.nv12"
        master = self.dirs["masters_dir"] / "tiny.png"
        before = (frame.stat().st_mtime_ns, master.stat().st_mtime_ns)
        time.sleep(0.01)
        self._prepare("1080p")
        after = (frame.stat().st_mtime_ns, master.stat().st_mtime_ns)
        self.assertEqual(before, after)

    def test_attributions_written(self):
        self._prepare("1080p")
        text = (self.dirs["out_dir"] / pd.ATTRIBUTIONS_NAME).read_text()
        self.assertIn("tiny: Test Author, CC BY-SA 4.0, https://commons.wikimedia.org/wiki/File:tiny", text)

    def test_download_failures_are_aggregated(self):
        def fetch(url):
            raise urllib.error.URLError("down")

        with mock.patch("time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                pd.prepare(
                    [self.entry],
                    "1080p",
                    sources_ref={"path": "sources.json", "sha256": "0" * 64},
                    fetch=fetch,
                    **self.dirs,
                )
        self.assertIn("tiny", str(ctx.exception))
        self.assertIn("downloads failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
