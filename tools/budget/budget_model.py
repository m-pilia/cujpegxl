# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic resource-budget model for the Orin Nano 8GB target.

The encoder is developed and profiled on a discrete host GPU. This tool ingests
per-stage host measurements (bytes moved, GPU microseconds, CPU microseconds per
frame), projects them onto the Orin Nano with a set of scaling factors, and
reports the projected steady-state resource use against the device budget:

  * DRAM bandwidth  <= 17 GB/s   (25% of ~68 GB/s shared LPDDR5)
  * SM time         <= 25%       (fraction of wall-clock the GPU is busy)
  * CPU             <= 1.5 cores
  * throughput      >= 15 fps    at 3840x2160

Projection factors are placeholders calibrated on the host and refined once
real device measurements are available; they live in a separate sheet so the
model itself stays policy-free.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

ORIN_SHARED_BANDWIDTH_BPS: float = 68.0e9
BANDWIDTH_SHARE: float = 0.25
SM_TIME_SHARE: float = 0.25
CPU_CORE_BUDGET: float = 1.5
TARGET_FPS: float = 15.0


@dataclasses.dataclass(frozen=True)
class StageRecord:
    name: str
    bytes_moved: float
    gpu_us: float
    cpu_us: float

    @staticmethod
    def from_dict(raw: dict) -> "StageRecord":
        return StageRecord(
            name=str(raw["name"]),
            bytes_moved=float(raw["bytes_moved"]),
            gpu_us=float(raw["gpu_us"]),
            cpu_us=float(raw["cpu_us"]),
        )


@dataclasses.dataclass(frozen=True)
class ProjectionFactors:
    gpu_time: float
    cpu_time: float
    bandwidth: float

    @staticmethod
    def from_dict(raw: dict) -> "ProjectionFactors":
        return ProjectionFactors(
            gpu_time=float(raw["gpu_time"]),
            cpu_time=float(raw["cpu_time"]),
            bandwidth=float(raw["bandwidth"]),
        )


@dataclasses.dataclass(frozen=True)
class Budget:
    orin_bandwidth_bps: float = ORIN_SHARED_BANDWIDTH_BPS
    bandwidth_share: float = BANDWIDTH_SHARE
    sm_time_share: float = SM_TIME_SHARE
    cpu_core_budget: float = CPU_CORE_BUDGET
    target_fps: float = TARGET_FPS

    @property
    def bandwidth_budget_bps(self) -> float:
        return self.orin_bandwidth_bps * self.bandwidth_share


@dataclasses.dataclass(frozen=True)
class Constraint:
    name: str
    value: float
    limit: float
    unit: str
    within_budget: bool


@dataclasses.dataclass(frozen=True)
class BudgetReport:
    fps: float
    constraints: list[Constraint]

    @property
    def passed(self) -> bool:
        return all(c.within_budget for c in self.constraints)


def evaluate(
    stages: list[StageRecord],
    fps: float,
    projection: ProjectionFactors,
    budget: Budget = Budget(),
) -> BudgetReport:
    frame_time_us = 1.0e6 / fps

    total_bytes = sum(s.bytes_moved for s in stages) * projection.bandwidth
    total_gpu_us = sum(s.gpu_us for s in stages) * projection.gpu_time
    total_cpu_us = sum(s.cpu_us for s in stages) * projection.cpu_time

    avg_bandwidth_bps = total_bytes * fps
    sm_time_fraction = total_gpu_us / frame_time_us
    cpu_cores = total_cpu_us / frame_time_us

    constraints = [
        Constraint(
            name="dram_bandwidth",
            value=avg_bandwidth_bps,
            limit=budget.bandwidth_budget_bps,
            unit="B/s",
            within_budget=avg_bandwidth_bps <= budget.bandwidth_budget_bps,
        ),
        Constraint(
            name="sm_time",
            value=sm_time_fraction,
            limit=budget.sm_time_share,
            unit="fraction",
            within_budget=sm_time_fraction <= budget.sm_time_share,
        ),
        Constraint(
            name="cpu_cores",
            value=cpu_cores,
            limit=budget.cpu_core_budget,
            unit="cores",
            within_budget=cpu_cores <= budget.cpu_core_budget,
        ),
        Constraint(
            name="throughput",
            value=fps,
            limit=budget.target_fps,
            unit="fps",
            within_budget=fps >= budget.target_fps,
        ),
    ]
    return BudgetReport(fps=fps, constraints=constraints)


def format_report(report: BudgetReport) -> str:
    width = max(len(c.name) for c in report.constraints)
    lines = []
    for c in report.constraints:
        status = "PASS" if c.within_budget else "FAIL"
        comparator = ">=" if c.name == "throughput" else "<="
        lines.append(
            f"[{status}] {c.name:<{width}}  {c.value:.4g} {comparator} "
            f"{c.limit:.4g} {c.unit}"
        )
    lines.append(f"overall: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


def load_input(path: pathlib.Path) -> tuple[list[StageRecord], float, ProjectionFactors]:
    raw = json.loads(path.read_text())
    stages = [StageRecord.from_dict(s) for s in raw["stages"]]
    fps = float(raw["fps"])
    projection = ProjectionFactors.from_dict(raw["projection"])
    return stages, fps, projection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="Stage-record JSON file.")
    args = parser.parse_args(argv)

    stages, fps, projection = load_input(args.input)
    report = evaluate(stages, fps, projection)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
