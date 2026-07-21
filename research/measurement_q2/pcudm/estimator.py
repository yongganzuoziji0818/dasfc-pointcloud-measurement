"""Non-learning PCU-DM-Field estimator and matched-capacity cascade baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EstimationResult:
    rotation: np.ndarray
    translation: np.ndarray
    displacement: np.ndarray
    normal_displacement: np.ndarray
    scale: np.ndarray
    support_probability: np.ndarray
    match_distance: np.ndarray
    valid: np.ndarray
    converged: bool
    iterations: int
    diagnostics: dict


def _weighted_rigid(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if source.shape[0] < 3 or total <= 1e-12:
        raise ValueError("At least three positively weighted correspondences are required")
    weights = weights / total
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    x = source - source_center
    y = target - target_center
    covariance = x.T @ (weights[:, None] * y)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _robust_weights(residual: np.ndarray, floor: float = 0.25) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    center = np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual - center)) + floor
    cutoff = 1.5 * scale
    return np.minimum(1.0, cutoff / np.maximum(np.abs(residual - center), 1e-12))


def _panel_local_coordinates(reference: np.ndarray, panel_ids: np.ndarray):
    local = np.zeros((reference.shape[0], 2), dtype=np.float64)
    for panel in np.unique(panel_ids):
        index = panel_ids == panel
        x = reference[index, 0]
        z = reference[index, 2]
        local[index, 0] = (x - x.min()) / max(np.ptp(x), 1e-12)
        local[index, 1] = (z - z.min()) / max(np.ptp(z), 1e-12)
    return local


def _basis(local: np.ndarray) -> np.ndarray:
    x = local[:, 0]
    z = local[:, 1]
    return np.column_stack(
        (
            np.ones_like(x),
            x,
            z,
            x * x,
            x * z,
            z * z,
            np.sin(np.pi * x) * np.sin(np.pi * z),
            np.sin(np.pi * x) * np.sin(2.0 * np.pi * z),
            np.sin(2.0 * np.pi * x) * np.sin(np.pi * z),
        )
    )


def _fit_piecewise_field(
    reference: np.ndarray,
    panel_ids: np.ndarray,
    observed: np.ndarray,
    base_weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    local = _panel_local_coordinates(reference, panel_ids)
    prediction = np.zeros(reference.shape[0], dtype=np.float64)
    for panel in np.unique(panel_ids):
        index = panel_ids == panel
        design = _basis(local[index])
        response = observed[index]
        weights = np.maximum(base_weights[index], 1e-6)
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(4):
            root_w = np.sqrt(weights)
            lhs = design * root_w[:, None]
            rhs = response * root_w
            penalty = np.eye(design.shape[1]) * ridge
            penalty[0, 0] = ridge * 0.05
            coefficients = np.linalg.solve(lhs.T @ lhs + penalty, lhs.T @ rhs)
            residual = response - design @ coefficients
            weights = np.maximum(base_weights[index] * _robust_weights(residual), 1e-6)
        prediction[index] = design @ coefficients
    return prediction


def _local_scale(
    reference: np.ndarray,
    panel_ids: np.ndarray,
    residual: np.ndarray,
    match_distance: np.ndarray,
    neighbors: int,
    floor_mm: float,
) -> np.ndarray:
    scale = np.empty(reference.shape[0], dtype=np.float64)
    for panel in np.unique(panel_ids):
        index = np.flatnonzero(panel_ids == panel)
        coordinates = reference[index][:, (0, 2)]
        count = min(max(4, neighbors), index.size)
        neighbor_index = cKDTree(coordinates).query(coordinates, k=count)[1]
        if neighbor_index.ndim == 1:
            neighbor_index = neighbor_index[:, None]
        local_abs = np.abs(residual[index][neighbor_index])
        local_mad = 1.4826 * np.median(local_abs, axis=1)
        correspondence = 0.20 * match_distance[index]
        scale[index] = np.sqrt(local_mad**2 + correspondence**2 + floor_mm**2)
    return scale


class PCUDMFieldEstimator:
    """Alternating pose/field estimator with a matched one-pass cascade mode.

    Parameters
    ----------
    mode:
        ``"joint"`` alternates pose, piecewise field, scale, and support weights.
        ``"cascade"`` performs the same steps exactly once and is the direct
        matched-capacity baseline required by the novelty audit.
    """

    def __init__(
        self,
        *,
        mode: str = "joint",
        outer_iterations: int = 5,
        icp_iterations: int = 12,
        trim_fraction: float = 0.82,
        ridge: float = 1e-2,
        scale_neighbors: int = 20,
        scale_floor_mm: float = 0.35,
        convergence_mm: float = 1e-3,
        query_workers: int = -1,
    ):
        if mode not in {"joint", "cascade"}:
            raise ValueError("mode must be 'joint' or 'cascade'")
        if not 0.5 <= trim_fraction < 1.0:
            raise ValueError("trim_fraction must be in [0.5, 1)")
        self.mode = mode
        self.outer_iterations = 1 if mode == "cascade" else outer_iterations
        self.icp_iterations = icp_iterations
        self.trim_fraction = trim_fraction
        self.ridge = ridge
        self.scale_neighbors = scale_neighbors
        self.scale_floor_mm = scale_floor_mm
        self.convergence_mm = convergence_mm
        if query_workers == 0 or query_workers < -1:
            raise ValueError("query_workers must be -1 or a positive integer")
        self.query_workers = query_workers

    def _estimate_pose(
        self,
        source: np.ndarray,
        target: np.ndarray,
        pose_weights: np.ndarray,
        initial_rotation: np.ndarray,
        initial_translation: np.ndarray,
    ):
        rotation = initial_rotation.copy()
        translation = initial_translation.copy()
        tree = cKDTree(target)
        last_change = np.inf
        for _ in range(self.icp_iterations):
            transformed = source @ rotation.T + translation
            distance, match_index = tree.query(transformed, workers=self.query_workers)
            eligible = pose_weights > 1e-5
            gate = np.quantile(distance[eligible], self.trim_fraction)
            keep = eligible & (distance <= gate)
            weights = pose_weights[keep] * _robust_weights(distance[keep])
            new_rotation, new_translation = _weighted_rigid(
                source[keep], target[match_index[keep]], weights
            )
            last_change = float(
                np.linalg.norm(new_translation - translation)
                + np.linalg.norm(new_rotation - rotation, ord="fro")
            )
            rotation, translation = new_rotation, new_translation
            if last_change < 1e-7:
                break
        return rotation, translation, last_change

    def fit(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        panel_ids: np.ndarray,
        support_candidates: np.ndarray,
        *,
        initial_rotation: np.ndarray | None = None,
        initial_translation: np.ndarray | None = None,
        pose_locked: bool = False,
    ) -> EstimationResult:
        reference = np.asarray(reference, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        panel_ids = np.asarray(panel_ids)
        support_candidates = np.asarray(support_candidates, dtype=bool)
        if reference.ndim != 2 or reference.shape[1] != 3 or reference.shape[0] < 20:
            raise ValueError("reference must have shape (N, 3) with N >= 20")
        if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] < 20:
            raise ValueError("target must have shape (M, 3) with M >= 20")
        if panel_ids.shape[0] != reference.shape[0] or support_candidates.shape[0] != reference.shape[0]:
            raise ValueError("panel_ids and support_candidates must match reference")
        if support_candidates.sum() < 6:
            raise ValueError("At least six support candidates are required")

        rotation = (
            np.eye(3, dtype=np.float64)
            if initial_rotation is None
            else np.asarray(initial_rotation, dtype=np.float64).copy()
        )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("initial_rotation must be a finite 3x3 matrix")
        # A median can jump by one lattice spacing on regular panel grids when an
        # occlusion removes one side. The midpoint of robust marginal bounds is
        # less sensitive to both that imbalance and the bounded outlier fraction.
        reference_bounds = np.quantile(reference, (0.10, 0.90), axis=0)
        target_bounds = np.quantile(target, (0.10, 0.90), axis=0)
        if initial_translation is None:
            translation = 0.5 * (target_bounds[0] + target_bounds[1])
            translation -= 0.5 * (reference_bounds[0] + reference_bounds[1])
        else:
            translation = np.asarray(initial_translation, dtype=np.float64).copy()
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError("initial_translation must be a finite 3-vector")
        normal_field = np.zeros(reference.shape[0], dtype=np.float64)
        support_probability = support_candidates.astype(np.float64)
        scale = np.full(reference.shape[0], self.scale_floor_mm, dtype=np.float64)
        last_field_change = np.inf
        last_pose_change = np.inf
        valid = np.ones(reference.shape[0], dtype=bool)

        for outer in range(self.outer_iterations):
            deformed_source = reference.copy()
            deformed_source[:, 1] += normal_field
            if self.mode == "joint" and outer > 0:
                pose_weights = 0.15 + 0.85 * support_probability
            else:
                pose_weights = support_candidates.astype(np.float64)
            if pose_locked:
                last_pose_change = 0.0
            else:
                rotation, translation, last_pose_change = self._estimate_pose(
                    deformed_source,
                    target,
                    pose_weights,
                    rotation,
                    translation,
                )

            target_reference_frame = (target - translation) @ rotation
            tree = cKDTree(target_reference_frame)
            query = reference.copy()
            query[:, 1] += normal_field
            match_distance, match_index = tree.query(query, workers=self.query_workers)
            matched = target_reference_frame[match_index]
            observed_normal = matched[:, 1] - reference[:, 1]
            gate = np.quantile(match_distance, self.trim_fraction)
            valid = match_distance <= gate
            data_weights = valid.astype(np.float64) * _robust_weights(match_distance)
            # Geometry-defined supports are strong but not oracle-fixed: observations
            # at those locations still enter the robust fit and can be downweighted.
            data_weights *= 0.35 + 0.65 * (~support_candidates)
            updated_field = _fit_piecewise_field(
                reference,
                panel_ids,
                observed_normal,
                data_weights,
                ridge=self.ridge,
            )
            residual = observed_normal - updated_field
            updated_scale = _local_scale(
                reference,
                panel_ids,
                residual,
                match_distance,
                neighbors=self.scale_neighbors,
                floor_mm=self.scale_floor_mm,
            )
            standardized_field = np.abs(updated_field) / np.maximum(updated_scale, 1e-9)
            support_probability = support_candidates.astype(np.float64) * np.exp(
                -0.5 * standardized_field**2
            )
            support_probability *= np.exp(-match_distance / max(gate, 1e-9))
            last_field_change = float(np.max(np.abs(updated_field - normal_field)))
            normal_field, scale = updated_field, updated_scale
            if self.mode == "joint" and max(last_field_change, last_pose_change) < self.convergence_mm:
                break

        displacement = np.zeros_like(reference)
        displacement[:, 1] = normal_field
        converged = bool(np.isfinite(normal_field).all() and np.isfinite(scale).all())
        return EstimationResult(
            rotation=rotation,
            translation=translation,
            displacement=displacement,
            normal_displacement=normal_field,
            scale=scale,
            support_probability=support_probability,
            match_distance=match_distance,
            valid=valid,
            converged=converged,
            iterations=outer + 1,
            diagnostics={
                "mode": self.mode,
                "pose_locked": pose_locked,
                "last_field_change_mm": last_field_change,
                "last_pose_change": last_pose_change,
                "valid_fraction": float(valid.mean()),
                "median_match_distance_mm": float(np.median(match_distance)),
            },
        )
