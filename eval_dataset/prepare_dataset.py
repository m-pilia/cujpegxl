# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Download and prepare the Commons eval corpus at a fixed rung resolution.

Consumes eval_dataset/sources.json (see query.py) in three idempotent stages:
originals are fetched once into a sha1-verified cache, 4K PNG masters are
derived (EXIF orientation, center 16:9 crop, LANCZOS resample to 3840x2160),
and the selected rung's NV12 frames plus corpus_manifest.json are produced
through the shared tools/corpus conventions, so the output directory is a
drop-in sibling of //data. Because every source satisfies width >= 3840 and
height >= 2160, the 16:9 crop is never smaller than the master target and
the resample is always a downscale.

When run through `bazelisk run`, pass absolute --* paths: the py_binary's
working directory is its runfiles tree, not the workspace.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable

import numpy as np
from PIL import Image, ImageOps

import corpus_prep as cp

USER_AGENT = "cujpegxl-eval-dataset/1.0"
MIME_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/tiff": "tif"}
# upload.wikimedia.org rate-limits bulk original downloads (HTTP 429) and asks
# for a serial, paced access pattern: requests are strictly sequential and
# spaced by DOWNLOAD_PAUSE_S; 429/503 responses back off exponentially while
# honoring Retry-After, and a streak of them trips a long client-wide cooldown.
DOWNLOAD_PAUSE_S = 30.0
RETRIES = 5
BACKOFF_BASE_S = 30.0
BACKOFF_MAX_S = 600.0
RATE_LIMIT_STREAK = 3
RATE_LIMIT_COOLDOWN_S = 300.0
RESOLUTIONS = ("1080p", "2160p")
MANIFEST_NAME = "corpus_manifest.json"
ATTRIBUTIONS_NAME = "ATTRIBUTIONS.txt"

Fetch = Callable[[str], bytes]


@dataclasses.dataclass
class _RateLimiter:
    """Tracks the client-wide streak of consecutive 429/503 responses."""

    streak: int = 0

    def note_rate_limit(self) -> float:
        self.streak += 1
        if self.streak >= RATE_LIMIT_STREAK:
            self.streak = 0
            return RATE_LIMIT_COOLDOWN_S
        return 0.0

    def note_success(self) -> None:
        self.streak = 0


def _hash_file(path: pathlib.Path, algo: str) -> str:
    digest = hashlib.new(algo)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def cache_path(entry: dict, cache_dir: pathlib.Path) -> pathlib.Path:
    return cache_dir / f"{entry['name']}.{MIME_EXT[entry['mime']]}"


def download_original(
    entry: dict,
    cache_dir: pathlib.Path,
    rate_limiter: _RateLimiter,
    fetch: Fetch = _http_get,
) -> pathlib.Path:
    dest = cache_path(entry, cache_dir)
    if dest.exists() and (not entry["sha1"] or _hash_file(dest, "sha1") == entry["sha1"]):
        return dest
    for attempt in range(RETRIES):
        try:
            data = fetch(entry["url"])
        except urllib.error.HTTPError as err:
            if err.code in (429, 503) and attempt < RETRIES - 1:
                retry_after = err.headers.get("Retry-After")
                time.sleep(
                    min(
                        max(
                            float(retry_after) if retry_after is not None else 0.0,
                            BACKOFF_BASE_S * 2.0**attempt,
                        ),
                        BACKOFF_MAX_S,
                    )
                    + rate_limiter.note_rate_limit()
                )
                continue
            raise RuntimeError(f"{entry['name']}: download failed: {err}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt < RETRIES - 1:
                time.sleep(2.0**attempt)
                continue
            raise RuntimeError(f"{entry['name']}: download failed: {err}") from err
        if entry["sha1"] and hashlib.sha1(data).hexdigest() != entry["sha1"]:
            raise RuntimeError(f"{entry['name']}: sha1 mismatch for {entry['url']}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        rate_limiter.note_success()
        time.sleep(DOWNLOAD_PAUSE_S)
        return dest
    raise RuntimeError("unreachable")


def _crop_16_9(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width * 9 > height * 16:
        crop_width = int(round(height * 16 / 9))
        left = (width - crop_width) // 2
        return image.crop((left, 0, left + crop_width, height))
    if width * 9 < height * 16:
        crop_height = int(round(width * 9 / 16))
        top = (height - crop_height) // 2
        return image.crop((0, top, width, top + crop_height))
    return image


def build_master(source_path: pathlib.Path, dest_path: pathlib.Path) -> None:
    with Image.open(source_path) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
        master = _crop_16_9(rgb).resize(
            (cp.FULL_4K_WIDTH, cp.FULL_4K_HEIGHT), Image.Resampling.LANCZOS
        )
        tmp = dest_path.with_name(dest_path.name + ".part")
        master.save(tmp, format="PNG")
        tmp.replace(dest_path)


def _rung_for(resolution: str) -> cp.Rung:
    for rung in cp.LADDER:
        if rung.name == resolution:
            return rung
    raise ValueError(f"unsupported resolution: {resolution}")


def _reusable_asset(
    manifest_entries: Iterable[dict], source_name: str, rung: cp.Rung, out_dir: pathlib.Path
) -> cp.Asset | None:
    for entry in manifest_entries:
        if entry["rung"] != rung.name or entry["source"] != source_name:
            continue
        if entry["width"] != rung.width or entry["height"] != rung.height:
            continue
        frame = out_dir / entry["name"]
        if frame.exists() and _hash_file(frame, "sha256") == entry["sha256"]:
            return cp.Asset(**entry)
    return None


def build_nv12(master_path: pathlib.Path, rung: cp.Rung, out_dir: pathlib.Path) -> cp.Asset:
    rgb = cp.load_rgb(master_path)
    assert rgb.shape[:2] == (cp.FULL_4K_HEIGHT, cp.FULL_4K_WIDTH), master_path
    return cp.build_asset(master_path, rgb, rung, out_dir)


def _download_all(entries: list[dict], cache_dir: pathlib.Path, fetch: Fetch) -> list[pathlib.Path]:
    failures: list[str] = []
    cached: list[pathlib.Path] = []
    rate_limiter = _RateLimiter()
    for entry in entries:
        try:
            cached.append(download_original(entry, cache_dir, rate_limiter, fetch))
        except RuntimeError as err:
            failures.append(str(err))
    if failures:
        raise RuntimeError("downloads failed:\n  " + "\n  ".join(failures))
    cached.sort(key=lambda path: path.name)
    return cached


def prepare(
    entries: list[dict],
    resolution: str,
    cache_dir: pathlib.Path,
    masters_dir: pathlib.Path,
    out_dir: pathlib.Path,
    sources_ref: dict,
    fetch: Fetch = _http_get,
) -> dict:
    rung = _rung_for(resolution)
    for directory in (cache_dir, masters_dir, out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cached = _download_all(entries, cache_dir, fetch)
    print(f"stage A: {len(cached)} originals in {cache_dir}", file=sys.stderr)

    masters: list[pathlib.Path] = []
    for index, path in enumerate(cached):
        master = masters_dir / f"{path.stem}.png"
        if not master.exists():
            build_master(path, master)
        masters.append(master)
        print(f"stage B: {index + 1}/{len(cached)} {master.name}", file=sys.stderr)

    manifest_path = out_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    existing = manifest.get("assets", [])
    source_names = {master.name for master in masters}
    kept = [
        entry
        for entry in existing
        if entry["rung"] != rung.name and entry["source"] in source_names
    ]
    assets = [cp.Asset(**entry) for entry in kept]
    reused = 0
    for index, master in enumerate(masters):
        asset = _reusable_asset(existing, master.name, rung, out_dir)
        if asset is None:
            asset = build_nv12(master, rung, out_dir)
        else:
            reused += 1
        assets.append(asset)
        print(f"stage C: {index + 1}/{len(masters)} {asset.name}", file=sys.stderr)
    assets.sort(key=lambda a: a.name)

    doc = {
        "colorspace": "bt709",
        "range": "full",
        "format": "nv12",
        "chroma": "420",
        "sources": sources_ref,
        "versions": {"pillow": Image.__version__, "numpy": np.__version__},
        "assets": [dataclasses.asdict(a) for a in assets],
    }
    manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    lines = [
        f"{entry['name']}: {entry['author'] or 'uncredited'}, {entry['license']}, {entry['page']}"
        for entry in sorted(entries, key=lambda e: e["name"])
    ]
    (out_dir / ATTRIBUTIONS_NAME).write_text("\n".join(lines) + "\n")
    print(
        f"stage C: {len(assets)} assets ({reused} reused) in {out_dir}", file=sys.stderr
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=pathlib.Path,
        default=pathlib.Path("eval_dataset/sources.json"),
    )
    parser.add_argument("--resolution", required=True, choices=RESOLUTIONS)
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("eval_dataset/out")
    )
    parser.add_argument(
        "--cache-dir", type=pathlib.Path, default=pathlib.Path("eval_dataset/cache")
    )
    parser.add_argument(
        "--masters-dir",
        type=pathlib.Path,
        default=pathlib.Path("eval_dataset/masters"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    doc = json.loads(args.sources.read_text())
    entries = doc["files"][: args.limit] if args.limit else doc["files"]
    if not entries:
        raise SystemExit(f"no entries in {args.sources}")
    sources_ref = {
        "path": str(args.sources),
        "sha256": _hash_file(args.sources, "sha256"),
    }
    prepare(
        entries,
        args.resolution,
        args.cache_dir,
        args.masters_dir,
        args.out_dir,
        sources_ref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
