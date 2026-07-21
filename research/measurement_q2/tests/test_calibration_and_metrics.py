from __future__ import annotations

import unittest

import numpy as np

from research.measurement_q2.pcudm.calibration import PairwiseSimultaneousCalibrator
from research.measurement_q2.pcudm.metrics import aurc, scan_pair_metrics


class CalibrationAndMetricTests(unittest.TestCase):
    def test_pair_score_uses_maximum_within_each_pair(self):
        error = np.array([0.2, -1.5, 0.4])
        scale = np.array([0.2, 0.5, 0.8])
        score = PairwiseSimultaneousCalibrator.pair_score(error, scale)
        self.assertEqual(score, 3.0)

    def test_finite_sample_order_statistic_is_recorded_per_group(self):
        cases = []
        for index in range(20):
            cases.append({
                "group": "known",
                "error": np.array([float(index + 1)]),
                "scale": np.ones(1),
            })
        state = PairwiseSimultaneousCalibrator(alpha=0.05).fit(cases)
        self.assertEqual(state.group_counts["known"], 20)
        self.assertEqual(state.group_order_statistics["known"], 20)
        self.assertEqual(state.group_quantiles["known"], 20.0)

    def test_unseen_group_requires_explicit_pooled_fallback(self):
        calibrator = PairwiseSimultaneousCalibrator()
        state = calibrator.fit([
            {"group": "a", "error": np.array([1.0]), "scale": np.array([1.0])}
        ])
        self.assertEqual(state.group_quantiles["a"], 1.0)
        with self.assertRaises(KeyError):
            calibrator.quantile("unseen")
        self.assertEqual(calibrator.quantile("unseen", allow_pooled_fallback=True), 1.0)

    def test_scan_pair_metric_does_not_create_point_level_replicates(self):
        truth = np.array([0.0, 1.0, 2.0])
        prediction = np.array([0.0, 1.2, 1.8])
        result = scan_pair_metrics(truth, prediction, truth - 0.3, truth + 0.3)
        self.assertEqual(result["field_locations"], 3)
        self.assertTrue(result["simultaneous_covered"])

    def test_aurc_rewards_correct_failure_ranking(self):
        risk = np.array([0.1, 0.2, 1.0, 2.0])
        good_order = np.array([0.1, 0.2, 1.0, 2.0])
        bad_order = -good_order
        self.assertLess(aurc(risk, good_order), aurc(risk, bad_order))


if __name__ == "__main__":
    unittest.main()
