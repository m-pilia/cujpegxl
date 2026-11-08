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
With --jxl-dir the .jxl files are instead read from a directory of files encoded
elsewhere (e.g. on another device), named as in tools/corpus/jxl_dir.

Usage:
  bazelisk run //tools/golden:golden_hash -- capture --manifest $PWD/golden.json
  bazelisk run //tools/golden:golden_hash -- verify  --manifest $PWD/golden.json
  bazelisk run //tools/golden:golden_hash -- verify  --manifest $PWD/golden.json \
      --jxl-dir $PWD/encoded
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
import jxl_dir

DEFAULT_RUNGS: tuple[str, ...] = ("1080p", "2160p")
DEFAULT_DISTANCES: tuple[float, ...] = (1.0,)


@dataclasses.dataclass(frozen=True)
class Job:
    frame: str
    rung: str
    width: int
    height: int
    distance: float


# Produce the .jxl bytes of one job, by encoding it or by reading a pre-encoded file.
EncodeFn = Callable[[Job], bytes]


def entry_for(job: Job, encode: EncodeFn) -> dict:
    data = encode(job)
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


def make_encoder(binary: pathlib.Path, device: int, frames: list[pathlib.Path],
                 nv12_dir: pathlib.Path) -> EncodeFn:
    """Encode each job with encode_to_file from its NV12 frame, generated into `nv12_dir`."""
    sources = {source.stem: source for source in frames}
    nv12_paths: dict[tuple[str, str], pathlib.Path] = {}

    def nv12_for(job: Job) -> pathlib.Path:
        key = (job.frame, job.rung)
        if key not in nv12_paths:
            rung_dir = nv12_dir / job.rung
            rung_dir.mkdir(parents=True, exist_ok=True)
            source = sources[job.frame]
            nv12_paths[key], _ = cp.write_rung_nv12(source, cp.load_rgb(source),
                                                    cp.rung_by_name(job.rung), rung_dir)
        return nv12_paths[key]

    def encode(job: Job) -> bytes:
        with tempfile.TemporaryDirectory() as scratch:
            out = pathlib.Path(scratch) / "out.jxl"
            cmd = [
                str(binary),
                f"--dims={job.width}x{job.height}",
                f"--distance={job.distance}",
                f"--device={device}",
                f"--output={out}",
                str(nv12_for(job)),
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"encode_to_file failed (exit {completed.returncode}):\n{completed.stderr}"
                )
            return out.read_bytes()

    return encode


def make_reader(directory: pathlib.Path, jobs: list[Job]) -> EncodeFn:
    """Read each job's .jxl from `directory`; every job's file must exist."""
    paths = jxl_dir.jxl_paths(directory, [(job.frame, job.rung, job.distance) for job in jobs])

    def read(job: Job) -> bytes:
        return paths[(job.frame, job.rung, job.distance)].read_bytes()

    return read


def build_jobs(frames: list[pathlib.Path], rungs: list[cp.Rung],
               distances: list[float]) -> list[Job]:
    return [
        Job(frame=source.stem, rung=rung.name, width=rung.width, height=rung.height,
            distance=distance)
        for rung in rungs
        for source in frames
        for distance in distances
    ]


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
    parser.add_argument("--jxl-dir", type=pathlib.Path, default=None,
                        help="Hash the .jxl files in this directory instead of encoding.")
    args = parser.parse_args(argv)

    frames = cp.data_frames(args.data_dir)
    rungs = [cp.rung_by_name(name) for name in args.rungs]
    jobs = build_jobs(frames, rungs, args.distances)
    if args.jxl_dir is not None:
        return run(args.mode, jobs, make_reader(args.jxl_dir, jobs), args.manifest)

    binary = args.encoder if args.encoder is not None else resolve_encoder()
    if not binary.exists():
        parser.error(f"encode_to_file binary not found at {binary}")
    if args.nv12_dir is not None:
        return run(args.mode, jobs, make_encoder(binary, args.device, frames, args.nv12_dir),
                   args.manifest)
    with tempfile.TemporaryDirectory() as tmp:
        return run(args.mode, jobs, make_encoder(binary, args.device, frames, pathlib.Path(tmp)),
                   args.manifest)


if __name__ == "__main__":
    sys.exit(main())
