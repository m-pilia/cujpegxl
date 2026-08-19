# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Network-free tests for query.py against committed API-response fixtures."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import query

TESTDATA = pathlib.Path(__file__).parent / "testdata"


def _load(name: str) -> dict:
    return json.loads((TESTDATA / name).read_text())


class _FakeApi:
    """Serves categorymembers pages in order and imageinfo per title batch."""

    def __init__(self, member_pages: list[dict], imageinfo_pages: list[dict]):
        self._member_pages = list(member_pages)
        self._imageinfo_pages = list(imageinfo_pages)
        self.member_calls = 0
        self.imageinfo_calls = 0
        self.imageinfo_batch_sizes: list[int] = []

    def __call__(self, params: dict) -> dict:
        if params.get("list") == "categorymembers":
            self.member_calls += 1
            return self._member_pages.pop(0)
        assert params.get("prop") == "imageinfo"
        self.imageinfo_calls += 1
        titles = params["titles"].split("|")
        self.imageinfo_batch_sizes.append(len(titles))
        assert len(titles) <= query.IMAGEINFO_BATCH
        assert params["iiextmetadatafilter"] == "Artist|LicenseShortName"
        combined = {"query": {"pages": {}}}
        remaining = list(titles)
        for page in self._imageinfo_pages:
            for key, value in page["query"]["pages"].items():
                if value["title"] in remaining:
                    combined["query"]["pages"][key] = value
                    remaining.remove(value["title"])
        assert not remaining, f"unexpected titles requested: {remaining}"
        return combined


def _fake_api() -> _FakeApi:
    return _FakeApi(
        [
            _load("categorymembers_page1.json"),
            _load("categorymembers_page2.json"),
        ],
        [
            _load("imageinfo_page1.json"),
            _load("imageinfo_page2.json"),
        ],
    )


class StripHtmlTest(unittest.TestCase):
    def test_strips_tags_decodes_entities_and_collapses_whitespace(self):
        self.assertEqual(
            query._strip_html(
                '<a href="x" title="t">&quot;Jane&lt;Doe&gt;&quot;</a> from  Zurich'
            ),
            '"Jane<Doe>" from Zurich',
        )
        self.assertEqual(
            query._strip_html("Harbour &amp; Co.\nPhotography"),
            "Harbour & Co. Photography",
        )
        self.assertEqual(query._strip_html("Diego Delso"), "Diego Delso")


class SanitizeNameTest(unittest.TestCase):
    def test_lowercases_and_replaces_non_alphanumerics(self):
        self.assertEqual(
            query._sanitize_name("File:Antiguo faro de Akranes, Vesturland.JPG"),
            "antiguo_faro_de_akranes_vesturland",
        )

    def test_strips_extension_and_non_ascii(self):
        self.assertEqual(query._sanitize_name("File:Überlöwe.png"), "berl_we")

    def test_truncates_to_limit_without_trailing_underscore(self):
        name = query._sanitize_name("File:" + "a" * 100 + ".png")
        self.assertEqual(name, "a" * query.NAME_MAX_CHARS)

    def test_fallback_for_nameless_files(self):
        self.assertEqual(query._sanitize_name("File:???.png"), "file")


class ParseImageinfoTest(unittest.TestCase):
    def test_page1_filters_and_fields(self):
        records = query.parse_imageinfo(_load("imageinfo_page1.json"))
        by_title = {r.title: r for r in records}
        # Too-small PNG and vanished (empty imageinfo) files filtered out.
        self.assertEqual(
            set(by_title),
            {
                "File:Zurich skyline panorama.jpg",
                "File:Antiguo faro de Akranes, Vesturland.JPG",
            },
        )
        zurich = by_title["File:Zurich skyline panorama.jpg"]
        self.assertEqual(zurich.author, '"Jane<Doe>" from Zurich')
        self.assertEqual(zurich.license, "CC BY-SA 4.0")
        self.assertEqual(
            zurich.page,
            "https://commons.wikimedia.org/wiki/File:Zurich_skyline_panorama.jpg",
        )
        self.assertTrue(zurich.url.startswith("https://upload.wikimedia.org/"))
        self.assertNotIn("utm_", zurich.url)
        self.assertTrue(zurich.url.endswith(".jpg"))
        akranes = by_title["File:Antiguo faro de Akranes, Vesturland.JPG"]
        self.assertEqual(akranes.author, "Diego Delso")

    def test_page2_fields(self):
        records = query.parse_imageinfo(_load("imageinfo_page2.json"))
        by_title = {r.title: r for r in records}
        self.assertEqual(
            set(by_title), {"File:Matterton Glacier.tif", "File:Blue hour harbour.jpg"}
        )
        # Missing extmetadata falls back to empty author and "unknown" license.
        glacier = by_title["File:Matterton Glacier.tif"]
        self.assertEqual(glacier.author, "")
        self.assertEqual(glacier.license, "unknown")
        self.assertEqual(
            by_title["File:Blue hour harbour.jpg"].author,
            "Harbour & Co. Photography",
        )

    def test_filters_non_raster_mimes(self):
        records = query.parse_imageinfo(_load("imageinfo_rejected.json"))
        self.assertEqual(records, [])

    def test_filters_oversized_images(self):
        records = query.parse_imageinfo(_load("imageinfo_oversized.json"))
        self.assertEqual([r.title for r in records], ["File:Vertical stitch.png"])


class CollectRecordsTest(unittest.TestCase):
    def test_paginates_and_filters(self):
        api = _fake_api()
        records = query.collect_records(api, num_images=4)
        self.assertEqual(api.member_calls, 2)
        self.assertEqual(len(records), 4)
        self.assertEqual(api.imageinfo_batch_sizes, [4, 2])

    def test_stops_early_once_pool_is_large_enough(self):
        api = _fake_api()
        records = query.collect_records(api, num_images=2)
        # First member page alone yields two passing files.
        self.assertEqual(api.member_calls, 1)
        self.assertEqual(len(records), 2)


class BuildSourcesDocTest(unittest.TestCase):
    def _records(self):
        return query.parse_imageinfo(
            _load("imageinfo_page1.json")
        ) + query.parse_imageinfo(_load("imageinfo_page2.json"))

    def test_sorts_by_title_and_limits(self):
        doc = query.build_sources_doc(self._records(), 3, "2026-08-18")
        self.assertEqual(
            [f["name"] for f in doc["files"]],
            [
                "antiguo_faro_de_akranes_vesturland",
                "blue_hour_harbour",
                "matterton_glacier",
            ],
        )
        self.assertEqual(doc["generated"], "2026-08-18")
        self.assertEqual(doc["constraints"]["min_width"], 3840)
        self.assertEqual(doc["constraints"]["min_height"], 2160)
        self.assertEqual(doc["constraints"]["max_width"], 8000)
        self.assertEqual(doc["constraints"]["max_height"], 8000)
        self.assertEqual(doc["constraints"]["mimes"], list(query.ALLOWED_MIMES))

    def test_deduplicates_identical_names(self):
        zurich = query.parse_imageinfo(_load("imageinfo_page1.json"))[0]

        def with_title(title):
            return zurich.__class__(**{**zurich.__dict__, "title": title})

        aaa = with_title("File:Aaa.JPG")
        same_base = with_title("File:aAA.JPG")
        zzz = with_title("File:Zzz.JPG")
        doc = query.build_sources_doc([same_base, aaa, zzz], 3, "2026-08-18")
        self.assertEqual(
            [f["name"] for f in doc["files"]], ["aaa", "zzz", "aaa_2"]
        )

    def test_entries_carry_required_metadata(self):
        doc = query.build_sources_doc(self._records(), 10, "2026-08-18")
        for entry in doc["files"]:
            for key in (
                "name",
                "page",
                "url",
                "width",
                "height",
                "mime",
                "sha1",
                "author",
                "license",
            ):
                self.assertIn(key, entry)
            self.assertGreaterEqual(entry["width"], 3840)
            self.assertGreaterEqual(entry["height"], 2160)
            self.assertLessEqual(entry["width"], query.MAX_WIDTH)
            self.assertLessEqual(entry["height"], query.MAX_HEIGHT)


class MainTest(unittest.TestCase):
    def test_writes_sources_json(self):
        api = _fake_api()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "sources.json"
            with mock.patch.object(query, "fetch_json", side_effect=api):
                self.assertEqual(
                    query.main(["--num-images", "3", "--output", str(out)]), 0
                )
            doc = json.loads(out.read_text())
        self.assertEqual(len(doc["files"]), 3)
        self.assertEqual(
            [f["name"] for f in doc["files"]],
            [
                "antiguo_faro_de_akranes_vesturland",
                "blue_hour_harbour",
                "matterton_glacier",
            ],
        )

    def test_rejects_non_positive_num_images(self):
        with self.assertRaises(SystemExit):
            query.main(["--num-images", "0", "--output", "/dev/null"])


if __name__ == "__main__":
    unittest.main()
