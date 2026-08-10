# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Compare cujpegxl vs nvJPEG encode runtime at comparable quality operating points.

Current quality settings:

  nvJPEG q=90 (bpp 2.26, ssimulacra2 79.5)  ~  cujpegxl d=1.0 (bpp 2.05, ssim2 82.3)
  nvJPEG q=80 (bpp 1.68, ssimulacra2 74.5)  ~  cujpegxl d=1.5 (bpp 1.67, ssim2 76.5)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import benchmark as bm
import corpus_prep as cp

# (nvJPEG quality, comparable cujpegxl distance); see the module docstring.
DEFAULT_PAIRS: tuple[tuple[int, float], ...] = ((90, 1.0), (80, 1.5))


def run_comparison(pairs: tuple[tuple[int, float], ...], *, data_dir: pathlib.Path, rung: cp.Rung,
                   iterations: int, warmup: int, device: int) -> list[tuple[bm.BenchStats,
                                                                            bm.BenchStats]]:
    bench = bm.resolve_encode_bench()
    if not bench.exists():
        raise FileNotFoundError(f"encode_bench binary not found at {bench}")
    frames = bm.data_frames(data_dir)

    results: list[tuple[bm.BenchStats, bm.BenchStats]] = []
    with tempfile.TemporaryDirectory() as tmp:
        nv12_files = bm.generate_corpus(frames, rung, pathlib.Path(tmp))
        for quality, distance in pairs:
            nvjpeg = bm.benchmark_codec(bench, nv12_files, rung, codec="nvjpeg",
                                        iterations=iterations, warmup=warmup, device=device,
                                        quality=quality)
            cujpegxl = bm.benchmark_codec(bench, nv12_files, rung, codec="cujpegxl",
                                          iterations=iterations, warmup=warmup, device=device,
                                          distance=distance)
            results.append((nvjpeg, cujpegxl))
    return results


def _format_table(results: list[tuple[bm.BenchStats, bm.BenchStats]]) -> str:
    header = (
        f"{'codec':9} {'setting':8} {'mean_us':>9} {'p50_us':>9} {'p99_us':>9} "
        f"{'fps':>7} {'MP/s':>8}"
    )
    lines = [header, "-" * len(header)]
    for nvjpeg, cujpegxl in results:
        for stats in (nvjpeg, cujpegxl):
            lines.append(
                f"{stats.codec:9} {stats.setting:8} {stats.mean_us:9.0f} {stats.p50_us:9.0f} "
                f"{stats.p99_us:9.0f} {stats.fps:7.1f} {stats.megapixels_per_s:8.1f}"
            )
        speedup = nvjpeg.mean_us / cujpegxl.mean_us if cujpegxl.mean_us > 0 else float("nan")
        faster = "cujpegxl" if speedup >= 1.0 else "nvJPEG"
        factor = speedup if speedup >= 1.0 else 1.0 / speedup
        lines.append(
            f"  -> {nvjpeg.setting} vs {cujpegxl.setting}: {faster} faster by {factor:.2f}x"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rung",
        default="2160p",
        choices=[r.name for r in cp.LADDER],
        help="Ladder rung to benchmark (2160p = 3840x2160). Default: 2160p.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations per codec, excluded from timing.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory with source PNG frames (default: data).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Write the results as JSON to this path.",
    )
    args = parser.parse_args(argv)

    rung = next(r for r in cp.LADDER if r.name == args.rung)
    results = run_comparison(DEFAULT_PAIRS, data_dir=args.data_dir, rung=rung,
                             iterations=args.iterations, warmup=args.warmup, device=args.device)

    print(_format_table(results))

    if args.output is not None:
        payload = [
            {"nvjpeg": nvjpeg.as_dict(), "cujpegxl": cujpegxl.as_dict(),
             "speedup_cujpegxl_over_nvjpeg": nvjpeg.mean_us / cujpegxl.mean_us}
            for nvjpeg, cujpegxl in results
        ]
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
