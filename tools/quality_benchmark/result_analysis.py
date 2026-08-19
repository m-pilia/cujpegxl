# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Compare cujpegxl against libjxl and nvJPEG on a quality-benchmark sweep.

Reads the JSON written by //tools/quality_benchmark:quality_benchmark with
--output. For cujpegxl vs nvJPEG and cujpegxl vs libjxl, reports:
  * quality delta at fixed bpp, anchored at every cujpegxl operating point
  * bpp delta at fixed quality, anchored at every cujpegxl operating point
and saves, per metric (PSNR, Butteraugli, SSIMulacra2), one bpp-vs-quality plot
with the mean curve of each codec across images and one scatter plot with all
per-image samples.

Positive delta means cujpegxl is better, negative means worse.
"""

import argparse
import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, stdev

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

METRICS = ("psnr", "butteraugli", "ssimulacra2")
METRIC_LABELS = {
    "psnr": "PSNR (dB)",
    "butteraugli": "Butteraugli (axis inverted: higher = better)",
    "ssimulacra2": "SSIMulacra2",
}
HIGHER_IS_BETTER = {"psnr": True, "butteraugli": False, "ssimulacra2": True}
METRIC_DECIMALS = {"psnr": 2, "butteraugli": 3, "ssimulacra2": 2}

CODEC_ORDER = ("cujpegxl", "libjxl", "nvjpeg")
CODEC_DISPLAY = {"cujpegxl": "cujpegxl", "libjxl": "libjxl", "nvjpeg": "nvJPEG"}
CODEC_STYLE = {
    "cujpegxl": {"color": "tab:blue", "marker": "o"},
    "libjxl": {"color": "tab:orange", "marker": "s"},
    "nvjpeg": {"color": "tab:green", "marker": "^"},
}


@dataclass(frozen=True)
class Record:
    codec: str
    image: str
    param: str
    param_value: float
    bpp: float
    metrics: dict


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for entry in json.load(handle):
            records.append(
                Record(
                    codec=entry["codec"],
                    image=entry["image"],
                    param=entry["param"],
                    param_value=float(entry["param"].split("=", 1)[1]),
                    bpp=entry["bpp"],
                    metrics={m: entry[m] for m in METRICS},
                )
            )
    return records


def orient(metric, value):
    return value if HIGHER_IS_BETTER[metric] else -value


def merge_sorted_pairs(pairs):
    xs, ys, counts = [], [], []
    for x, y in sorted(pairs):
        if xs and x == xs[-1]:
            counts[-1] += 1
            ys[-1] += (y - ys[-1]) / counts[-1]
        else:
            xs.append(x)
            ys.append(y)
            counts.append(1)
    return xs, ys


def interp(xs, ys, x):
    if len(xs) < 2 or x < xs[0] or x > xs[-1]:
        return None
    i = bisect.bisect_left(xs, x)
    if xs[i] == x:
        return ys[i]
    x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def fmt(value, decimals, signed=False):
    if value is None:
        return "-"
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def print_table(header, rows):
    all_rows = [header] + rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(header))]
    for row in all_rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def summarize(values):
    return {
        "n": len(values),
        "wins": sum(v > 0 for v in values),
        "mean": fmean(values) if values else None,
        "median": median(values) if values else None,
        "std": stdev(values) if len(values) >= 2 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def win_rate(stats):
    if not stats["n"]:
        return "-"
    return f"{100 * stats['wins'] / stats['n']:.0f}%"


def compare_codec_pair(anchor_by_image, other_by_image, metric):
    """Per cujpegxl anchor point, interpolate the other codec on this image.

    Returns (quality_at_bpp, bpp_at_quality) lists of
    (anchor record, delta or None) tuples.
    """
    quality_at_bpp = []
    bpp_at_quality = []
    for image in sorted(anchor_by_image):
        anchor_records = anchor_by_image[image]
        other_records = other_by_image.get(image)
        if not other_records:
            continue
        # Orienting the metric before interpolation keeps a single monotone
        # quality axis for both "higher is better" and "lower is better"
        # metrics, so bpp-at-quality lookups always search increasing quality.
        bpps, qualities = merge_sorted_pairs(
            (r.bpp, orient(metric, r.metrics[metric])) for r in other_records
        )
        qualities_sorted, bpps_sorted = merge_sorted_pairs(
            (orient(metric, r.metrics[metric]), r.bpp) for r in other_records
        )
        for anchor in anchor_records:
            other_quality = interp(bpps, qualities, anchor.bpp)
            if other_quality is None:
                quality_at_bpp.append((anchor, None))
            else:
                quality_at_bpp.append(
                    (anchor, orient(metric, anchor.metrics[metric]) - other_quality)
                )
            other_bpp = interp(
                qualities_sorted, bpps_sorted, orient(metric, anchor.metrics[metric])
            )
            if other_bpp is None:
                bpp_at_quality.append((anchor, None))
            else:
                bpp_at_quality.append((anchor, other_bpp - anchor.bpp))
    return quality_at_bpp, bpp_at_quality


def print_comparison(anchor_by_image, other_by_image, metric):
    quality_at_bpp, bpp_at_quality = compare_codec_pair(
        anchor_by_image, other_by_image, metric
    )
    decimals = METRIC_DECIMALS[metric]
    anchor_params = [
        param
        for _, param in sorted({(a.param_value, a.param) for a, _ in quality_at_bpp})
    ]
    param_or_all = anchor_params + [None]

    def rows_for(comparisons):
        rows = []
        for param in param_or_all:
            selected = [
                (anchor, delta)
                for anchor, delta in comparisons
                if param is None or anchor.param == param
            ]
            # Anchor bpp is averaged over all anchors of the param, including
            # skipped ones, so the column keeps meaning when entries drop out.
            bpp_mean = (
                fmean(anchor.bpp for anchor, _ in selected) if selected else None
            )
            deltas = [delta for _, delta in selected if delta is not None]
            stats = summarize(deltas)
            rows.append((param or "all", bpp_mean, selected, stats))
        return rows

    quality_rows = []
    for label, bpp_mean, selected, stats in rows_for(quality_at_bpp):
        quality_rows.append(
            [
                label,
                fmt(bpp_mean, 3),
                str(stats["n"]),
                str(len(selected) - stats["n"]),
                win_rate(stats),
                fmt(stats["mean"], decimals, signed=True),
                fmt(stats["median"], decimals, signed=True),
                fmt(stats["std"], decimals),
                fmt(stats["min"], decimals, signed=True),
                fmt(stats["max"], decimals, signed=True),
            ]
        )

    bpp_rows = []
    for label, bpp_mean, selected, stats in rows_for(bpp_at_quality):
        evaluated = [(anchor, delta) for anchor, delta in selected if delta is not None]
        pct = (
            fmean(delta / anchor.bpp * 100 for anchor, delta in evaluated)
            if evaluated
            else None
        )
        bpp_rows.append(
            [
                label,
                fmt(bpp_mean, 3),
                str(stats["n"]),
                str(len(selected) - stats["n"]),
                win_rate(stats),
                fmt(stats["mean"], 4, signed=True),
                fmt(stats["median"], 4, signed=True),
                fmt(stats["std"], 4),
                fmt(pct, 1, signed=True),
            ]
        )

    print(f"\n--- {metric} ---\n")
    print("quality at fixed bpp (positive delta = cujpegxl better)")
    print_table(
        ["anchor", "bpp", "n", "skip", "win%", "mean", "median", "std", "min", "max"],
        quality_rows,
    )
    print("\nbpp at fixed quality (positive delta = cujpegxl uses fewer bits)")
    print_table(
        ["anchor", "bpp", "n", "skip", "win%", "mean", "median", "std", "pct%"],
        bpp_rows,
    )


def plot_metric_mean(by_param, metric, num_images, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for codec in CODEC_ORDER:
        params = sorted(
            by_param[codec].values(), key=lambda records: records[0].param_value
        )
        bpps = [fmean(r.bpp for r in records) for records in params]
        qualities = [fmean(r.metrics[metric] for r in records) for records in params]
        stds = [
            stdev(r.metrics[metric] for r in records) if len(records) >= 2 else 0.0
            for records in params
        ]
        style = CODEC_STYLE[codec]
        ax.errorbar(
            bpps,
            qualities,
            yerr=stds,
            marker=style["marker"],
            color=style["color"],
            label=CODEC_DISPLAY[codec],
            capsize=3,
        )
    finish_metric_plot(
        fig, ax, metric, f"Mean over {num_images} images", output_dir, ""
    )


def plot_metric_samples(by_param, metric, num_images, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for codec in CODEC_ORDER:
        records = [r for params in by_param[codec].values() for r in params]
        style = CODEC_STYLE[codec]
        ax.scatter(
            [r.bpp for r in records],
            [r.metrics[metric] for r in records],
            marker=style["marker"],
            color=style["color"],
            s=25,
            alpha=0.5,
            edgecolors="none",
            label=CODEC_DISPLAY[codec],
        )
    finish_metric_plot(
        fig, ax, metric, f"Samples over {num_images} images", output_dir, "_scatter"
    )


def finish_metric_plot(fig, ax, metric, title, output_dir, suffix):
    if not HIGHER_IS_BETTER[metric]:
        ax.invert_yaxis()
    ax.set_xlabel("bits per pixel (bpp)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"bpp_vs_{metric}{suffix}.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"saved {output_path}")


def print_dataset_overview(records, by_param):
    images = {r.image for r in records}
    print(f"dataset: {len(records)} records, {len(images)} images")
    for codec in CODEC_ORDER:
        param_means = sorted(
            (
                fmean(r.bpp for r in records)
                for records in by_param[codec].values()
            ),
        )
        print(
            f"  {CODEC_DISPLAY[codec]:8s} {len(by_param[codec]):2d} params, "
            f"mean bpp range {param_means[0]:.3f} - {param_means[-1]:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    records = load_records(args.input)
    images = {r.image for r in records}
    by_image = {codec: defaultdict(list) for codec in CODEC_ORDER}
    by_param = {codec: defaultdict(list) for codec in CODEC_ORDER}
    for record in records:
        by_image[record.codec][record.image].append(record)
        by_param[record.codec][record.param].append(record)

    print_dataset_overview(records, by_param)
    for other_codec in ("nvjpeg", "libjxl"):
        print("\n" + "=" * 78)
        print(f"cujpegxl vs {CODEC_DISPLAY[other_codec]}".center(78))
        print("=" * 78)
        for metric in METRICS:
            print_comparison(by_image["cujpegxl"], by_image[other_codec], metric)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric_mean(by_param, metric, len(images), args.output_dir)
        plot_metric_samples(by_param, metric, len(images), args.output_dir)


if __name__ == "__main__":
    main()
