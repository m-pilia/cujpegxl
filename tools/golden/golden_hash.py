# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Golden-hash gate: capture reference .jxl SHA-256 and assert byte-identity.

The primary non-regression gate for bit-exact encoder changes. Every corpus
frame is encoded at each (rung, distance) of the tracked matrix; the SHA-256 of
the emitted .jxl is recorded once as a golden reference. After any change the
same matrix is re-encoded and asserted byte-identical to the golden manifest.
Any diff fails the gate.

The NV12 corpus is generated from data/*.png with tools/corpus/corpus_prep and
each frame is emitted to a .jxl by //tools/golden:encode_to_file. Encoding is
deterministic per device, so one encode per (frame, rung, distance) suffices.

Usage:
  bazelisk run //tools/golden:golden_hash -- capture --manifest $PWD/golden.json
  bazelisk run //tools/golden:golden_hash -- verify  --manifest $PWD/golden.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Callable

import corpus_prep as cp

DEFAULT_RUNGS: tuple[str, ...] = ("1080p", "2160p")
DEFAULT_DISTANCES: tuple[float, ...] = (1.0,)

# Encode one NV12 file (path, width, height, distance) to its .jxl bytes.
EncodeFn = Callable[[pathlib.Path, int, int, float], bytes]


@dataclasses.dataclass(frozen=True)
class Job:
    frame: str
    rung: str
    width: int
    height: int
    distance: float
    nv12: pathlib.Path


def entry_for(job: Job, encode: EncodeFn) -> dict:
    data = encode(job.nv12, job.width, job.height, job.distance)
    return {
        "frame": job.frame,
        "rung": job.rung,
        "width": job.width,
        "height": job.height,
        "distance": job.distance,
        "jxl_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def compute_entries(jobs: list[Job], encode: EncodeFn) -> list[dict]:
    return [entry_for(job, encode) for job in jobs]


def _entry_key(entry: dict) -> tuple[str, str, float]:
    return (entry["frame"], entry["rung"], float(entry["distance"]))


def compare_entries(built: list[dict], reference: list[dict]) -> list[str]:
    """Return human-readable mismatches of re-encoded `built` vs golden `reference`.

    An empty list means every re-encoded output is byte-identical to its golden
    reference over the same (frame, rung, distance) matrix.
    """
    built_by = {_entry_key(e): e for e in built}
    reference_by = {_entry_key(e): e for e in reference}

    mismatches: list[str] = []
    for key in sorted(reference_by.keys() - built_by.keys()):
        mismatches.append(f"{key}: in golden manifest but not re-encoded")
    for key in sorted(built_by.keys() - reference_by.keys()):
        mismatches.append(f"{key}: re-encoded but absent from golden manifest")
    for key in sorted(built_by.keys() & reference_by.keys()):
        got = built_by[key]
        want = reference_by[key]
        if got["sha256"] != want["sha256"]:
            mismatches.append(
                f"{key}: sha256 {got['sha256']} != golden {want['sha256']} "
                f"({got['jxl_bytes']} vs {want['jxl_bytes']} bytes)"
            )
    return mismatches


def resolve_encoder() -> pathlib.Path:
    candidate = pathlib.Path(__file__).resolve().parent / "encode_to_file"
    if candidate.exists():
        return candidate
    workspace_root = pathlib.Path(__file__).resolve().parents[2]
    candidate = workspace_root / "bazel-bin" / "tools" / "golden" / "encode_to_file"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "encode_to_file binary not found; build it with "
        "`bazelisk build //tools/golden:encode_to_file`"
    )


def make_encoder(binary: pathlib.Path, device: int) -> EncodeFn:
    def encode(nv12: pathlib.Path, width: int, height: int, distance: float) -> bytes:
        with tempfile.TemporaryDirectory() as scratch:
            out = pathlib.Path(scratch) / "out.jxl"
            cmd = [
                str(binary),
                f"--dims={width}x{height}",
                f"--distance={distance}",
                f"--device={device}",
                f"--output={out}",
                str(nv12),
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"encode_to_file failed (exit {completed.returncode}):\n{completed.stderr}"
                )
            return out.read_bytes()

    return encode


def data_frames(data_dir: pathlib.Path) -> list[pathlib.Path]:
    frames = sorted(data_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no *.png frames found under {data_dir}")
    return frames


def build_jobs(frames: list[pathlib.Path], rungs: list[cp.Rung], distances: list[float],
               nv12_dir: pathlib.Path) -> list[Job]:
    jobs: list[Job] = []
    for rung in rungs:
        rung_dir = nv12_dir / rung.name
        rung_dir.mkdir(parents=True, exist_ok=True)
        for source in frames:
            scaled = cp.box_downscale(cp.load_rgb(source), rung.factor)
            nv12_path = rung_dir / f"{source.stem}_{rung.name}.nv12"
            nv12_path.write_bytes(cp.rgb_to_nv12(scaled))
            for distance in distances:
                jobs.append(Job(
                    frame=source.stem,
                    rung=rung.name,
                    width=rung.width,
                    height=rung.height,
                    distance=distance,
                    nv12=nv12_path,
                ))
    return jobs


def _resolve_rungs(names: list[str]) -> list[cp.Rung]:
    by_name = {r.name: r for r in cp.LADDER}
    rungs: list[cp.Rung] = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"unknown rung {name!r}; known: {sorted(by_name)}")
        rungs.append(by_name[name])
    return rungs


def run(mode: str, jobs: list[Job], encode: EncodeFn, manifest_path: pathlib.Path) -> int:
    entries = compute_entries(jobs, encode)
    if mode == "capture":
        manifest = {"codec": "cujpegxl", "entries": entries}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"captured {len(entries)} golden hashes to {manifest_path}")
        return 0

    reference = json.loads(manifest_path.read_text())
    mismatches = compare_entries(entries, reference["entries"])
    if mismatches:
        print(f"golden-hash gate FAILED against {manifest_path}:")
        for line in mismatches:
            print(f"  {line}")
        return 1
    print(f"golden-hash gate OK: {len(entries)} outputs byte-identical to {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["capture", "verify"])
    parser.add_argument("--manifest", type=pathlib.Path, required=True,
                        help="Golden manifest to write (capture) or check against (verify).")
    parser.add_argument("--rungs", nargs="+", default=list(DEFAULT_RUNGS),
                        help=f"Ladder rungs to cover (default: {' '.join(DEFAULT_RUNGS)}).")
    parser.add_argument("--distances", nargs="+", type=float, default=list(DEFAULT_DISTANCES),
                        help="Butteraugli distances to cover "
                        f"(default: {' '.join(str(d) for d in DEFAULT_DISTANCES)}).")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"),
                        help="Directory with source PNG frames (default: data).")
    parser.add_argument("--nv12-dir", type=pathlib.Path, default=None,
                        help="Persist the generated NV12 corpus here instead of a temp dir.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--encoder", type=pathlib.Path, default=None,
                        help="Path to the encode_to_file binary (auto-detected if omitted).")
    args = parser.parse_args(argv)

    frames = data_frames(args.data_dir)
    rungs = _resolve_rungs(args.rungs)
    binary = args.encoder if args.encoder is not None else resolve_encoder()
    if not binary.exists():
        parser.error(f"encode_to_file binary not found at {binary}")
    encode = make_encoder(binary, args.device)

    if args.nv12_dir is not None:
        jobs = build_jobs(frames, rungs, args.distances, args.nv12_dir)
        return run(args.mode, jobs, encode, args.manifest)
    with tempfile.TemporaryDirectory() as tmp:
        jobs = build_jobs(frames, rungs, args.distances, pathlib.Path(tmp))
        return run(args.mode, jobs, encode, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
