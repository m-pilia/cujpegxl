# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import budget_model as bm


def _stage(name, bytes_moved, gpu_us, cpu_us):
    return bm.StageRecord(name=name, bytes_moved=bytes_moved, gpu_us=gpu_us, cpu_us=cpu_us)


_UNIT_PROJECTION = bm.ProjectionFactors(gpu_time=1.0, cpu_time=1.0, bandwidth=1.0)


class BudgetModelTest(unittest.TestCase):
    def test_within_budget_passes(self):
        stages = [_stage("frontend", 100e6, 5000.0, 2000.0)]
        report = bm.evaluate(stages, fps=15.0, projection=_UNIT_PROJECTION)
        self.assertTrue(report.passed)

    def test_bandwidth_violation_fails(self):
        stages = [_stage("bandwidth_hog", 2.0e9, 1000.0, 1000.0)]
        report = bm.evaluate(stages, fps=15.0, projection=_UNIT_PROJECTION)
        self.assertFalse(report.passed)
        offending = {c.name for c in report.constraints if not c.within_budget}
        self.assertIn("dram_bandwidth", offending)

    def test_sm_time_violation_fails(self):
        stages = [_stage("compute_hog", 10e6, 60000.0, 1000.0)]
        report = bm.evaluate(stages, fps=15.0, projection=_UNIT_PROJECTION)
        self.assertFalse(report.passed)
        offending = {c.name for c in report.constraints if not c.within_budget}
        self.assertIn("sm_time", offending)

    def test_below_target_fps_fails(self):
        stages = [_stage("frontend", 10e6, 1000.0, 1000.0)]
        report = bm.evaluate(stages, fps=10.0, projection=_UNIT_PROJECTION)
        self.assertFalse(report.passed)

    def test_projection_scales_gpu_time(self):
        stages = [_stage("frontend", 10e6, 20000.0, 1000.0)]
        doubled = bm.ProjectionFactors(gpu_time=2.0, cpu_time=1.0, bandwidth=1.0)
        report = bm.evaluate(stages, fps=15.0, projection=doubled)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
