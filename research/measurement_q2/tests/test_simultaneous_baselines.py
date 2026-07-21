from __future__ import annotations

import unittest

import numpy as np

from research.measurement_q2.pcudm.simultaneous_baselines import (
    BonferroniGaussianBand,
    ClassicalMaxTBand,
)


def cases(start: int, count: int, group: str = "known") -> list[dict]:
    output = []
    for index in range(start, start + count):
        error = np.array([0.1 * index, -0.05 * index, 0.02 * index], dtype=float)
        output.append({"group": group, "error": error, "valid": np.ones(3, dtype=bool)})
    return output


class SimultaneousBaselineTests(unittest.TestCase):
    def test_max_t_uses_one_score_per_calibration_trajectory(self):
        band = ClassicalMaxTBand(alpha=0.2)
        state = band.fit(cases(1, 8), cases(20, 8))
        self.assertEqual(state.group_calibration_counts["known"], 8)
        self.assertEqual(state.group_order_statistics["known"], 8)

    def test_max_t_interval_matches_field_topology(self):
        band = ClassicalMaxTBand(alpha=0.2)
        band.fit(cases(1, 8), cases(20, 8))
        lower, upper = band.interval(np.zeros(3), "known")
        self.assertEqual(lower.shape, (3,))
        self.assertTrue(np.all(upper >= lower))

    def test_bonferroni_critical_value_increases_with_family_size(self):
        small = BonferroniGaussianBand(alpha=0.05)
        small.fit(cases(1, 8))
        small_critical = small.diagnostics("known")["critical_value"]

        large_cases = []
        for item in cases(1, 8):
            large_cases.append({
                "group": item["group"],
                "error": np.tile(item["error"], 4),
                "valid": np.ones(12, dtype=bool),
            })
        large = BonferroniGaussianBand(alpha=0.05)
        large.fit(large_cases)
        self.assertGreater(large.diagnostics("known")["critical_value"], small_critical)


if __name__ == "__main__":
    unittest.main()
