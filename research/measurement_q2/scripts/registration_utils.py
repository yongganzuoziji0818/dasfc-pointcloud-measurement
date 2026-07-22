"""Shared registration helpers used by public-data benchmarks."""

from __future__ import annotations

import numpy as np

from research.measurement_q2.pcudm import PCUDMFieldEstimator
from research.measurement_q2.pcudm.registration_frontends import register_multiscale


def transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform to row-vector point coordinates."""
    return points @ rotation.T + translation


def frontend_transform(
    source: np.ndarray,
    target: np.ndarray,
    frontend: str,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate a rigid transform with one frozen registration front end."""
    if frontend == "published_coordinates":
        return np.eye(3), np.zeros(3), {}
    if frontend == "cascade_strong":
        result = PCUDMFieldEstimator(
            mode="cascade", icp_iterations=14, query_workers=workers
        ).fit(
            source,
            target,
            np.zeros(source.shape[0], dtype=np.int32),
            np.ones(source.shape[0], dtype=bool),
        )
        return result.rotation, result.translation, {
            "converged": result.converged,
            "valid_fraction": float(result.valid.mean()),
        }
    result = register_multiscale(
        source,
        target,
        frontend,
        thresholds=(0.20, 0.08, 0.03),
    )
    return result.rotation, result.translation, {
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
    }
