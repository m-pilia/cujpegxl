# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Build eval_dataset/sources.json from Wikimedia Commons featured pictures.

Selects direct file members of the "Featured pictures on Wikimedia Commons"
category whose resolution is at least 3840x2160 and whose mime type is a
supported raster format, and records permalinks plus author/license metadata
so prepare_dataset.py can fetch the originals on demand. The committed
sources.json snapshot is the source of truth; re-running the query against a
live Commons may legitimately produce a different set.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import html
import html.parser
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "cujpegxl-eval-dataset/1.0 (cujpegxl project metadata query)"
CATEGORY_TITLE = "Category:Featured pictures on Wikimedia Commons"
MIN_WIDTH = 3840
MIN_HEIGHT = 2160
ALLOWED_MIMES = ("image/jpeg", "image/png", "image/tiff")
MEMBERS_PAGE_SIZE = 500
IMAGEINFO_BATCH = 50
REQUEST_PAUSE_S = 1.0
RETRIES = 5
NAME_MAX_CHARS = 64


@dataclasses.dataclass(frozen=True)
class FileRecord:
    title: str
    page: str
    url: str
    width: int
    height: int
    mime: str
    sha1: str
    author: str
    license: str


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def _strip_html(markup):
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    return " ".join("".join(parser.parts).split())


def _strip_tracking(url):
    parts = urllib.parse.urlsplit(url)
    pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query)
        if not key.startswith("utm_")
    ]
    return urllib.parse.urlunsplit(
        parts._replace(query=urllib.parse.urlencode(pairs))
    )


def _sanitize_name(title):
    stem = title.removeprefix("File:").rsplit(".", 1)[0]
    replaced = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    return replaced.strip("_")[:NAME_MAX_CHARS].rstrip("_") or "file"


def _batched(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_json(params):
    query = {**params, "format": "json", "maxlag": "5"}
    url = f"{API_URL}?{urllib.parse.urlencode(query)}"
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as err:
            if err.code in (429, 503) and attempt < RETRIES - 1:
                retry_after = err.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after is not None
                    else 2.0**attempt * 5.0
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Commons API request failed: {err}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt < RETRIES - 1:
                time.sleep(2.0**attempt)
                continue
            raise RuntimeError(f"Commons API request failed: {err}") from err
        error = payload.get("error")
        if error is not None:
            if error.get("code") == "maxlag" and attempt < RETRIES - 1:
                time.sleep(2.0**attempt)
                continue
            raise RuntimeError(f"Commons API error: {error.get('info', error)}")
        time.sleep(REQUEST_PAUSE_S)
        return payload
    raise RuntimeError("unreachable")


def _imageinfo_params(titles):
    return {
        "action": "query",
        "prop": "imageinfo",
        "titles": "|".join(titles),
        "iiprop": "url|size|mime|sha1|extmetadata",
        "iiextmetadatafilter": "Artist|LicenseShortName",
    }


def parse_imageinfo(response):
    records = []
    for page in response["query"]["pages"].values():
        infos = page.get("imageinfo")
        if not infos:
            continue
        info = infos[0]
        if info.get("mime") not in ALLOWED_MIMES:
            continue
        if info.get("width", 0) < MIN_WIDTH or info.get("height", 0) < MIN_HEIGHT:
            continue
        ext = info.get("extmetadata", {})
        records.append(
            FileRecord(
                title=page["title"],
                page=info["descriptionurl"],
                url=_strip_tracking(info["url"]),
                width=info["width"],
                height=info["height"],
                mime=info["mime"],
                sha1=info.get("sha1", ""),
                author=_strip_html(ext.get("Artist", {}).get("value", "")),
                license=ext.get("LicenseShortName", {}).get("value", "") or "unknown",
            )
        )
    return records


def collect_records(fetch, num_images):
    records: list[FileRecord] = []
    cmcontinue = None
    pages = 0
    while len(records) < num_images:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": CATEGORY_TITLE,
            "cmtype": "file",
            "cmlimit": str(MEMBERS_PAGE_SIZE),
        }
        if cmcontinue is not None:
            params["cmcontinue"] = cmcontinue
        response = fetch(params)
        pages += 1
        titles = [m["title"] for m in response["query"]["categorymembers"]]
        for batch in _batched(titles, IMAGEINFO_BATCH):
            records.extend(parse_imageinfo(fetch(_imageinfo_params(batch))))
        print(
            f"category page {pages}: +{len(titles)} files, "
            f"{len(records)} passing filters",
            file=sys.stderr,
        )
        cmcontinue = response.get("continue", {}).get("cmcontinue")
        if cmcontinue is None:
            break
    return records


def build_sources_doc(records, num_images, generated):
    selected = sorted(records, key=lambda r: r.title)[:num_images]
    used: dict[str, int] = {}
    files = []
    for record in selected:
        base = _sanitize_name(record.title)
        count = used.get(base, 0)
        used[base] = count + 1
        files.append(
            {
                "name": base if count == 0 else f"{base}_{count + 1}",
                "page": record.page,
                "url": record.url,
                "width": record.width,
                "height": record.height,
                "mime": record.mime,
                "sha1": record.sha1,
                "author": record.author,
                "license": record.license,
            }
        )
    return {
        "generated": generated,
        "constraints": {
            "category": CATEGORY_TITLE,
            "min_width": MIN_WIDTH,
            "min_height": MIN_HEIGHT,
            "mimes": list(ALLOWED_MIMES),
        },
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("eval_dataset/sources.json"),
    )
    args = parser.parse_args(argv)
    if args.num_images < 1:
        parser.error("--num-images must be >= 1")

    records = collect_records(fetch_json, args.num_images)
    if len(records) < args.num_images:
        print(
            f"warning: only {len(records)} files matched, "
            f"requested {args.num_images}",
            file=sys.stderr,
        )
    doc = build_sources_doc(
        records, args.num_images, datetime.date.today().isoformat()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(doc['files'])} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
