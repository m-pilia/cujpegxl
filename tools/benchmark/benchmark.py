# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the GPU encode time of the 4K data corpus against cujpegxl or nvJPEG.

This is the convenience driver for //tools/benchmark:encode_bench. It generates
the NV12 corpus (BT.709, full range, 4:2:0) from the source PNG frames under
data/ using tools/corpus/corpus_prep.py, then invokes the C++ encode_bench
binary with one NV12 file per source frame so the timed loop alternates across
images. The H2D upload and NV12 deinterleave (for nvJPEG) happen once at
startup and are excluded from the timed window inside encode_bench.

Profiling for cujpegxl is exposed via --profile: it forwards to encode_bench's
--profile flag (single iteration, NVTX ranges, per-stage StageTiming breakdown)
so a follow-up `nsys profile` or `ncu` capture is clean.

Usage:
  bazelisk run //tools/benchmark:benchmark -- --codec=cujpegxl
  bazelisk run //tools/benchmark:benchmark -- --codec=nvjpeg --iterations=10
  bazelisk run //tools/benchmark:benchmark -- --codec=cujpegxl --profile \\
      && nsys profile ./bazel-bin/tools/benchmark/encode_bench ...
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import corpus_prep as cp


def _resolve_encode_bench() -> pathlib.Path:
    candidate = pathlib.Path(__file__).resolve().parent / "encode_bench"
    if candidate.exists():
        return candidate
    workspace_root = pathlib.Path(__file__).resolve().parents[2]
    candidate = workspace_root / "bazel-bin" / "tools" / "benchmark" / "encode_bench"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "encode_bench binary not found; build it with "
        "`bazelisk build //tools/benchmark:encode_bench`"
    )


def _data_frames(data_dir: pathlib.Path) -> list[pathlib.Path]:
    frames = sorted(data_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no *.png frames found under {data_dir}")
    return frames


def _generate_corpus(frames: list[pathlib.Path], rung: cp.Rung,
                     out_dir: pathlib.Path) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nv12_files: list[pathlib.Path] = []
    for source in frames:
        rgb_4k = cp.load_rgb(source)
        scaled = cp.box_downscale(rgb_4k, rung.factor)
        nv12 = cp.rgb_to_nv12(scaled)
        path = out_dir / f"{source.stem}_{rung.name}.nv12"
        path.write_bytes(nv12)
        nv12_files.append(path)
    return nv12_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", required=True, choices=["cujpegxl", "nvjpeg"])
    parser.add_argument(
        "--rung",
        default="2160p",
        choices=[r.name for r in cp.LADDER],
        help="Ladder rung to benchmark (2160p = 3840x2160). Default: 2160p.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--distance", type=float, default=1.0, help="cujpegxl only.")
    parser.add_argument("--quality", type=int, default=90, help="nvJPEG only.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="cujpegxl only: reduce iterations to 1, emit NVTX ranges, and print "
        "per-stage timings (intended for nsys/ncu capture).",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory with source PNG frames (default: data).",
    )
    parser.add_argument(
        "--nv12-dir",
        type=pathlib.Path,
        default=None,
        help="Reuse a pre-generated NV12 corpus from this directory instead of "
        "regenerating into a temp dir.",
    )
    parser.add_argument(
        "--encode-bench",
        type=pathlib.Path,
        default=None,
        help="Path to the encode_bench binary (auto-detected if omitted).",
    )
    args = parser.parse_args(argv)

    if args.profile and args.codec != "cujpegxl":
        parser.error("--profile is only supported with --codec=cujpegxl")

    rung = next(r for r in cp.LADDER if r.name == args.rung)
    bench = args.encode_bench if args.encode_bench is not None else _resolve_encode_bench()
    if not bench.exists():
        parser.error(f"encode_bench binary not found at {bench}")

    frames = _data_frames(args.data_dir)

    if args.nv12_dir is not None:
        nv12_files = _generate_corpus(frames, rung, args.nv12_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            nv12_files = _generate_corpus(frames, rung, pathlib.Path(tmp))
            return _invoke_bench(bench, nv12_files, args, rung)
    return _invoke_bench(bench, nv12_files, args, rung)


def _invoke_bench(bench: pathlib.Path, nv12_files: list[pathlib.Path], args: argparse.Namespace,
                  rung: cp.Rung) -> int:
    cmd = [
        str(bench),
        "--codec", args.codec,
        "--dims", f"{rung.width}x{rung.height}",
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup),
        "--device", str(args.device),
    ]
    if args.codec == "cujpegxl":
        cmd += ["--distance", str(args.distance)]
    else:
        cmd += ["--quality", str(args.quality)]
    if args.profile:
        cmd.append("--profile")
    cmd += [str(p) for p in nv12_files]

    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
