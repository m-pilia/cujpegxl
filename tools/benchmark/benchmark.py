# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark encode time on the 4K data corpus.

The cujpegxl and nvJPEG paths use the NV12 corpus and exclude input upload from
their timed window. Libjxl and gjxl receive RGB reconstructed from those same
NV12 frames. Gjxl uses its public C API, so its timed call includes input
conversion, upload, encoding, and output publication.

Profiling for cujpegxl is exposed via --profile: it forwards to encode_bench's
--profile flag (single iteration, NVTX ranges, per-stage StageTiming breakdown)
so a follow-up `nsys profile` or `ncu` capture is clean.

Usage:
  bazelisk run //tools/benchmark:benchmark -- --codec=cujpegxl
  bazelisk run //tools/benchmark:benchmark -- --codec=nvjpeg --iterations=10
  bazelisk run //tools/benchmark:benchmark -- --codec=libjxl --effort=7
  bazelisk run --config=gjxl //tools/benchmark:benchmark -- --codec=gjxl --effort=7
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
    setting: str
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
    if codec in ("cujpegxl", "libjxl", "gjxl"):
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


def resolve_encode_bench(codec: str = "cujpegxl") -> pathlib.Path:
    binary = {"libjxl": "rgb_bench", "gjxl": "gjxl_bench"}.get(codec, "encode_bench")
    candidate = pathlib.Path(__file__).resolve().parent / binary
    if candidate.exists():
        return candidate
    for root in (pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parents[2]):
        candidate = root / "bazel-bin" / "tools" / "benchmark" / binary
        if candidate.exists():
            return candidate
    if codec == "gjxl":
        raise FileNotFoundError(
            f"{binary} binary not found; run the benchmark with `--config=gjxl`"
        )
    raise FileNotFoundError(
        f"{binary} binary not found; build it with "
        f"`bazelisk build //tools/benchmark:{binary}`"
    )


def generate_corpus(frames: list[pathlib.Path], rung: cp.Rung,
                    out_dir: pathlib.Path) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [cp.write_rung_nv12(source, cp.load_rgb(source), rung, out_dir)[0]
            for source in frames]


def generate_rgb_corpus(nv12_files: list[pathlib.Path], rung: cp.Rung) -> list[pathlib.Path]:
    rgb_files = []
    for nv12_file in nv12_files:
        rgb_file = nv12_file.with_suffix(".rgb")
        rgb_file.write_bytes(cp.nv12_to_rgb(nv12_file.read_bytes(), rung.width, rung.height).tobytes())
        rgb_files.append(rgb_file)
    return rgb_files


def build_command(bench: pathlib.Path, nv12_files: list[pathlib.Path], rung: cp.Rung, *,
                  codec: str, iterations: int, warmup: int, device: int, distance: float = 1.0,
                  quality: int = 90, effort: int = 7, profile: bool = False) -> list[str]:
    if codec in ("libjxl", "gjxl"):
        cmd = [str(bench), str(rung.width), str(rung.height), str(iterations), str(warmup),
               str(distance), str(effort)]
        if codec == "gjxl":
            cmd.append(str(device))
        return [*cmd, *map(str, nv12_files)]
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
                    quality: int = 90, effort: int = 7) -> BenchStats:
    """Run one encode_bench invocation and return its parsed timing statistics.

    Each call includes its own warmup iterations (excluded from the timed window
    by encode_bench), so a codec is never measured cold.
    """
    cmd = build_command(bench, nv12_files, rung, codec=codec, iterations=iterations, warmup=warmup,
                        device=device, distance=distance, quality=quality, effort=effort)
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"encode_bench failed (exit {completed.returncode}):\n{completed.stderr}"
        )
    return parse_stats(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", required=True,
                        choices=["cujpegxl", "nvjpeg", "libjxl", "gjxl"])
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
    parser.add_argument("--effort", type=int, default=None,
                        help="libjxl and gjxl only (1..10; default 7).")
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
        help="Persist the generated NV12 corpus here instead of a temp dir.",
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
    if args.effort is not None and args.codec not in ("libjxl", "gjxl"):
        parser.error("--effort is only supported with --codec=libjxl or --codec=gjxl")
    if args.effort is not None and not 1 <= args.effort <= 10:
        parser.error("--effort must be in [1, 10]")
    if args.codec == "gjxl" and args.device != 0:
        parser.error("gjxl uses CUDA device 0; set CUDA_VISIBLE_DEVICES to select another GPU")

    rung = cp.rung_by_name(args.rung)
    bench = args.encode_bench if args.encode_bench is not None else resolve_encode_bench(args.codec)
    if not bench.exists():
        parser.error(f"encode_bench binary not found at {bench}")

    frames = cp.data_frames(args.data_dir)

    if args.nv12_dir is not None:
        nv12_files = generate_corpus(frames, rung, args.nv12_dir)
        files = generate_rgb_corpus(nv12_files, rung) if args.codec in ("libjxl", "gjxl") else nv12_files
        return _invoke_bench(bench, files, args, rung)
    with tempfile.TemporaryDirectory() as tmp:
        nv12_files = generate_corpus(frames, rung, pathlib.Path(tmp))
        files = generate_rgb_corpus(nv12_files, rung) if args.codec in ("libjxl", "gjxl") else nv12_files
        return _invoke_bench(bench, files, args, rung)


def _invoke_bench(bench: pathlib.Path, nv12_files: list[pathlib.Path], args: argparse.Namespace,
                  rung: cp.Rung) -> int:
    cmd = build_command(bench, nv12_files, rung, codec=args.codec, iterations=args.iterations,
                        warmup=args.warmup, device=args.device, distance=args.distance,
                        quality=args.quality, effort=args.effort or 7, profile=args.profile)
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
