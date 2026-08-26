# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
import tempfile
import unittest

import golden_hash as gh


def _job(frame: str, rung: str, distance: float) -> gh.Job:
    return gh.Job(frame=frame, rung=rung, width=1920, height=1080,
                  distance=distance, nv12=pathlib.Path(f"/dev/null/{frame}"))


def _fake_encoder(payload: bytes = b"") -> gh.EncodeFn:
    def encode(nv12: pathlib.Path, width: int, height: int, distance: float) -> bytes:
        return f"{nv12.name}|{width}x{height}|{distance}".encode() + payload
    return encode


class ComputeEntriesTest(unittest.TestCase):
    def test_entry_records_size_and_hash(self):
        [entry] = gh.compute_entries([_job("a", "1080p", 1.0)], _fake_encoder())
        self.assertEqual(entry["frame"], "a")
        self.assertEqual(entry["distance"], 1.0)
        self.assertEqual(entry["jxl_bytes"], len(b"a|1920x1080|1.0"))
        self.assertEqual(len(entry["sha256"]), 64)


class CompareEntriesTest(unittest.TestCase):
    def test_identical_entries_have_no_mismatch(self):
        jobs = [_job("a", "1080p", 1.0), _job("b", "2160p", 4.0)]
        entries = gh.compute_entries(jobs, _fake_encoder())
        self.assertEqual(gh.compare_entries(entries, entries), [])

    def test_changed_output_is_reported(self):
        jobs = [_job("a", "1080p", 1.0)]
        golden = gh.compute_entries(jobs, _fake_encoder())
        drifted = gh.compute_entries(jobs, _fake_encoder(payload=b"drift"))
        mismatches = gh.compare_entries(drifted, golden)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("sha256", mismatches[0])

    def test_missing_and_extra_keys_are_reported(self):
        golden = gh.compute_entries([_job("a", "1080p", 1.0)], _fake_encoder())
        built = gh.compute_entries([_job("b", "1080p", 1.0)], _fake_encoder())
        mismatches = gh.compare_entries(built, golden)
        self.assertTrue(any("not re-encoded" in m for m in mismatches))
        self.assertTrue(any("absent from golden manifest" in m for m in mismatches))


class RunTest(unittest.TestCase):
    def test_capture_then_verify_round_trip(self):
        jobs = [_job("a", "1080p", 1.0), _job("a", "2160p", 2.0)]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = pathlib.Path(tmp) / "golden.json"
            self.assertEqual(gh.run("capture", jobs, _fake_encoder(), manifest), 0)
            self.assertEqual(len(json.loads(manifest.read_text())["entries"]), 2)
            self.assertEqual(gh.run("verify", jobs, _fake_encoder(), manifest), 0)

    def test_verify_fails_on_drift(self):
        jobs = [_job("a", "1080p", 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = pathlib.Path(tmp) / "golden.json"
            gh.run("capture", jobs, _fake_encoder(), manifest)
            self.assertEqual(gh.run("verify", jobs, _fake_encoder(payload=b"x"), manifest), 1)


if __name__ == "__main__":
    unittest.main()
