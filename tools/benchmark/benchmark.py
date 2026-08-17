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
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass

import corpus_prep as cp


@dataclass
class BenchStats:
    """Parsed per-encode timing statistics from one encode_bench invocation."""

    codec: str
    setting: str  # "d=1.0" (cujpegxl) or "q=90" (nvJPEG)
    width: int
    height: int
    images: int
    iterations: int
    warmup: int
    n: int
    total_us: float
    mean_us: float
    fps: float
    stddev_us: float
    min_us: float
    p50_us: float
    p99_us: float
    max_us: float

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1.0e6

    @property
    def megapixels_per_s(self) -> float:
        return self.megapixels * self.fps

    def as_dict(self) -> dict:
        data = asdict(self)
        data["megapixels_per_s"] = self.megapixels_per_s
        return data


def parse_stats(output: str) -> BenchStats:
    """Parse the human-readable encode_bench report into a BenchStats."""

    def field(pattern: str) -> re.Match:
        match = re.search(pattern, output)
        if match is None:
            raise ValueError(f"could not parse encode_bench output for /{pattern}/:\n{output}")
        return match

    codec = field(r"codec:\s+(\S+)").group(1)
    width = int(field(r"dims:\s+(\d+)x\d+").group(1))
    height = int(field(r"dims:\s+\d+x(\d+)").group(1))
    if codec == "cujpegxl":
        setting = f"d={float(field(r'distance:\s+([\d.]+)').group(1))}"
    else:
        setting = f"q={int(field(r'quality:\s+(\d+)').group(1))}"
    mean_match = field(r"mean:\s+([\d.]+)\s+\(([\d.]+) fps\)")
    return BenchStats(
        codec=codec,
        setting=setting,
        width=width,
        height=height,
        images=int(field(r"images:\s+(\d+)").group(1)),
        iterations=int(field(r"iterations:\s+(\d+)").group(1)),
        warmup=int(field(r"warmup:\s+(\d+)").group(1)),
        n=int(field(r"\bn:\s+(\d+)").group(1)),
        total_us=float(field(r"total:\s+([\d.]+)").group(1)),
        mean_us=float(mean_match.group(1)),
        fps=float(mean_match.group(2)),
        stddev_us=float(field(r"stddev:\s+([\d.]+)").group(1)),
        min_us=float(field(r"min:\s+([\d.]+)").group(1)),
        p50_us=float(field(r"p50:\s+([\d.]+)").group(1)),
        p99_us=float(field(r"p99:\s+([\d.]+)").group(1)),
        max_us=float(field(r"max:\s+([\d.]+)").group(1)),
    )


def resolve_encode_bench() -> pathlib.Path:
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


def data_frames(data_dir: pathlib.Path) -> list[pathlib.Path]:
    frames = sorted(data_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no *.png frames found under {data_dir}")
    return frames


def generate_corpus(frames: list[pathlib.Path], rung: cp.Rung,
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


def build_command(bench: pathlib.Path, nv12_files: list[pathlib.Path], rung: cp.Rung, *,
                  codec: str, iterations: int, warmup: int, device: int, distance: float = 1.0,
                  quality: int = 90, profile: bool = False) -> list[str]:
    cmd = [
        str(bench),
        "--codec", codec,
        "--dims", f"{rung.width}x{rung.height}",
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--device", str(device),
    ]
    cmd += ["--distance", str(distance)] if codec == "cujpegxl" else ["--quality", str(quality)]
    if profile:
        cmd.append("--profile")
    cmd += [str(p) for p in nv12_files]
    return cmd


def benchmark_codec(bench: pathlib.Path, nv12_files: list[pathlib.Path], rung: cp.Rung, *,
                    codec: str, iterations: int, warmup: int, device: int, distance: float = 1.0,
                    quality: int = 90) -> BenchStats:
    """Run one encode_bench invocation and return its parsed timing statistics.

    Each call includes its own warmup iterations (excluded from the timed window
    by encode_bench), so a codec is never measured cold.
    """
    cmd = build_command(bench, nv12_files, rung, codec=codec, iterations=iterations, warmup=warmup,
                        device=device, distance=distance, quality=quality)
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"encode_bench failed (exit {completed.returncode}):\n{completed.stderr}"
        )
    return parse_stats(completed.stdout)


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
    bench = args.encode_bench if args.encode_bench is not None else resolve_encode_bench()
    if not bench.exists():
        parser.error(f"encode_bench binary not found at {bench}")

    frames = data_frames(args.data_dir)

    if args.nv12_dir is not None:
        nv12_files = generate_corpus(frames, rung, args.nv12_dir)
        return _invoke_bench(bench, nv12_files, args, rung)
    with tempfile.TemporaryDirectory() as tmp:
        nv12_files = generate_corpus(frames, rung, pathlib.Path(tmp))
        return _invoke_bench(bench, nv12_files, args, rung)


def _invoke_bench(bench: pathlib.Path, nv12_files: list[pathlib.Path], args: argparse.Namespace,
                  rung: cp.Rung) -> int:
    cmd = build_command(bench, nv12_files, rung, codec=args.codec, iterations=args.iterations,
                        warmup=args.warmup, device=args.device, distance=args.distance,
                        quality=args.quality, profile=args.profile)
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
