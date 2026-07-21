from __future__ import annotations

import unittest

import numpy as np

from research.measurement_q2.pcudm import (
    PCUDMFieldEstimator,
    StructuralScaleModel,
    SyntheticDomain,
    generate_case,
)


class StructuralScaleModelTests(unittest.TestCase):
    def test_fit_uses_independent_pairs_and_predicts_positive_scale(self):
        tuning = []
        estimator = PCUDMFieldEstimator(mode="cascade")
        for seed in (41, 42, 43, 44):
            case = generate_case(
                seed,
                SyntheticDomain(name="scale-test"),
                panel_count=2,
                points_x=12,
                points_z=10,
            )
            result = estimator.fit(
                case.reference, case.target, case.panel_ids, case.support_candidates
            )
            tuning.append({
                "case_id": case.case_id,
                "reference": case.reference,
                "panel_ids": case.panel_ids,
                "result": result,
                "error": result.normal_displacement - case.normal_displacement_true,
                "valid": case.valid_field_mask,
            })
        model = StructuralScaleModel(max_locations_per_pair=64, random_state=11)
        diagnostics = model.fit(tuning[:3])
        predicted = model.predict(
            tuning[3]["reference"], tuning[3]["panel_ids"], tuning[3]["result"]
        )
        self.assertEqual(diagnostics.scan_pairs, 3)
        self.assertEqual(diagnostics.training_locations, 3 * 64)
        self.assertEqual(predicted.shape, tuning[3]["error"].shape)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted > 0).all())


if __name__ == "__main__":
    unittest.main()
