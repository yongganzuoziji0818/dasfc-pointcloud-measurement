from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from research.measurement_q2.pcudm import PCUDMFieldEstimator, SyntheticDomain, generate_case


class SyntheticAndEstimatorTests(unittest.TestCase):
    def test_generator_is_deterministic_and_truth_is_corresponding(self):
        domain = SyntheticDomain(name="deterministic", noise_mm=0.0, dropout=0.0,
                                 outlier_fraction=0.0, occlusion_fraction=0.0,
                                 density_jitter=0.0)
        first = generate_case(17, domain, panel_count=2, points_x=12, points_z=10)
        second = generate_case(17, domain, panel_count=2, points_x=12, points_z=10)
        np.testing.assert_array_equal(first.reference, second.reference)
        np.testing.assert_array_equal(first.target, second.target)
        np.testing.assert_array_equal(first.support_candidates, second.support_candidates)
        reconstructed = (first.reference + first.displacement_true) @ first.rigid_rotation_true.T
        reconstructed += first.rigid_translation_true
        np.testing.assert_allclose(reconstructed, first.target_clean_corresponding, atol=1e-10)
        self.assertGreater(np.count_nonzero(first.support_candidates != first.support_true), 0)

    def test_rigid_pose_recovery_on_zero_displacement(self):
        domain = SyntheticDomain(name="rigid", noise_mm=0.0, dropout=0.0,
                                 outlier_fraction=0.0, occlusion_fraction=0.0,
                                 density_jitter=0.0, pose_translation_mm=6.0,
                                 pose_rotation_deg=0.8)
        case = generate_case(23, domain, panel_count=2, points_x=18, points_z=14,
                             amplitude_mm=0.0, joint_slip_mm=0.0)
        result = PCUDMFieldEstimator(mode="cascade", icp_iterations=20).fit(
            case.reference, case.target, case.panel_ids, case.support_candidates
        )
        rotation_error = Rotation.from_matrix(result.rotation @ case.rigid_rotation_true.T).magnitude()
        translation_error = np.linalg.norm(result.translation - case.rigid_translation_true)
        self.assertLess(np.rad2deg(rotation_error), 0.25)
        self.assertLess(translation_error, 1.0)
        self.assertLess(np.mean(np.abs(result.normal_displacement)), 1.0)

    def test_rbf_kink_family_is_deterministic_distinct_and_restrained(self):
        domain = SyntheticDomain(name="family", noise_mm=0.0, dropout=0.0,
                                 outlier_fraction=0.0, occlusion_fraction=0.0,
                                 density_jitter=0.0)
        default = generate_case(29, domain, panel_count=3, points_x=18, points_z=14)
        explicit_default = generate_case(
            29, domain, panel_count=3, points_x=18, points_z=14,
            deformation_family="modal_bulge"
        )
        first = generate_case(
            29, domain, panel_count=3, points_x=18, points_z=14,
            deformation_family="rbf_kink"
        )
        second = generate_case(
            29, domain, panel_count=3, points_x=18, points_z=14,
            deformation_family="rbf_kink"
        )
        np.testing.assert_array_equal(
            default.normal_displacement_true,
            explicit_default.normal_displacement_true,
        )
        np.testing.assert_array_equal(
            first.normal_displacement_true, second.normal_displacement_true
        )
        self.assertFalse(np.allclose(
            first.normal_displacement_true, default.normal_displacement_true
        ))
        exact_boundary = np.zeros(first.reference.shape[0], dtype=bool)
        for panel in np.unique(first.panel_ids):
            index = first.panel_ids == panel
            x = first.reference[index, 0]
            z = first.reference[index, 2]
            exact_boundary[index] = (
                np.isclose(x, x.min())
                | np.isclose(x, x.max())
                | np.isclose(z, z.min())
            )
        self.assertLess(float(np.max(np.abs(
            first.normal_displacement_true[exact_boundary]
        ))), 1e-10)
        self.assertEqual(first.metadata["deformation_family"], "rbf_kink")

    def test_joint_estimator_produces_finite_piecewise_field(self):
        case = generate_case(31, SyntheticDomain(name="nominal"), panel_count=3,
                             points_x=20, points_z=16)
        result = PCUDMFieldEstimator(mode="joint", outer_iterations=4).fit(
            case.reference, case.target, case.panel_ids, case.support_candidates
        )
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.normal_displacement).all())
        self.assertTrue((result.scale > 0).all())
        error = np.abs(result.normal_displacement - case.normal_displacement_true)
        self.assertLess(float(np.mean(error[case.valid_field_mask])), 8.0)
        self.assertEqual(np.unique(case.panel_ids).size, 3)


if __name__ == "__main__":
    unittest.main()
