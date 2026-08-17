# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Run a command twice and byte-compare a designated output.

The runner is codec-agnostic: it executes an arbitrary command line two times
and compares either its stdout or a file it produces. This is the determinism
harness used to enforce the per-device byte-identical determinism requirement in
CI.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import pathlib
import subprocess
import sys
import tempfile


@dataclasses.dataclass(frozen=True)
class RunArtifact:
    data: bytes

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclasses.dataclass(frozen=True)
class Comparison:
    identical: bool
    size_a: int
    size_b: int
    digest_a: str
    digest_b: str
    first_diff_offset: int | None


def _first_diff_offset(a: bytes, b: bytes) -> int | None:
    if a == b:
        return None
    limit = min(len(a), len(b))
    for offset in range(limit):
        if a[offset] != b[offset]:
            return offset
    return limit


def compare(a: RunArtifact, b: RunArtifact) -> Comparison:
    return Comparison(
        identical=a.data == b.data,
        size_a=len(a.data),
        size_b=len(b.data),
        digest_a=a.digest,
        digest_b=b.digest,
        first_diff_offset=_first_diff_offset(a.data, b.data),
    )


def _run_once(command: list[str], output_file: str | None) -> RunArtifact:
    if output_file is None:
        completed = subprocess.run(command, capture_output=True, check=True)
        return RunArtifact(data=completed.stdout)

    with tempfile.TemporaryDirectory() as scratch:
        target = pathlib.Path(scratch) / "output"
        rendered = [arg.replace("{output}", str(target)) for arg in command]
        subprocess.run(rendered, check=True)
        return RunArtifact(data=target.read_bytes())


def run_twice(command: list[str], output_file: str | None) -> Comparison:
    return compare(
        _run_once(command, output_file),
        _run_once(command, output_file),
    )


def format_report(result: Comparison) -> str:
    lines = [
        f"identical: {result.identical}",
        f"size_a: {result.size_a}",
        f"size_b: {result.size_b}",
        f"digest_a: {result.digest_a}",
        f"digest_b: {result.digest_b}",
    ]
    if not result.identical:
        lines.append(f"first_diff_offset: {result.first_diff_offset}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stdout",
        action="store_true",
        help="Compare the command's stdout across the two runs.",
    )
    mode.add_argument(
        "--output-file",
        action="store_true",
        help=(
            "Compare a file the command writes. The literal token {output} in "
            "the command is replaced by a fresh path for each run."
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command provided after options")

    result = run_twice(command, output_file="output" if args.output_file else None)
    print(format_report(result))
    return 0 if result.identical else 1


if __name__ == "__main__":
    sys.exit(main())
