"""Frozen non-learning rigid front ends used by P1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class RegistrationFrontendResult:
    rotation: np.ndarray
    translation: np.ndarray
    fitness: float
    inlier_rmse: float
    frontend: str
    thresholds: tuple[float, ...]


def robust_initial_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_bounds = np.quantile(source, (0.10, 0.90), axis=0)
    target_bounds = np.quantile(target, (0.10, 0.90), axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (
        0.5 * (target_bounds[0] + target_bounds[1])
        - 0.5 * (source_bounds[0] + source_bounds[1])
    )
    return transform


def point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def _normal_radius(source: np.ndarray, target: np.ndarray) -> float:
    combined_min = np.minimum(source.min(axis=0), target.min(axis=0))
    combined_max = np.maximum(source.max(axis=0), target.max(axis=0))
    return max(float(np.linalg.norm(combined_max - combined_min)) * 0.035, 1e-3)


def register_multiscale(
    source: np.ndarray,
    target: np.ndarray,
    frontend: str,
    thresholds: tuple[float, ...] = (100.0, 30.0, 10.0),
) -> RegistrationFrontendResult:
    if frontend not in {"multiscale_trimmed_ptp", "robust_ptpl"}:
        raise ValueError(f"unsupported P1 frontend: {frontend}")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1:] != (3,) or target.shape[1:] != (3,):
        raise ValueError("source and target must have shape (N, 3)")
    source_cloud = point_cloud(source)
    target_cloud = point_cloud(target)
    if frontend == "robust_ptpl":
        radius = _normal_radius(source, target)
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
        source_cloud.estimate_normals(search)
        target_cloud.estimate_normals(search)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    transform = robust_initial_transform(source, target)
    result = None
    for threshold in thresholds:
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            float(threshold),
            transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=24),
        )
        transform = result.transformation
    assert result is not None
    return RegistrationFrontendResult(
        rotation=np.asarray(transform[:3, :3], dtype=np.float64),
        translation=np.asarray(transform[:3, 3], dtype=np.float64),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        frontend=frontend,
        thresholds=tuple(float(value) for value in thresholds),
    )
