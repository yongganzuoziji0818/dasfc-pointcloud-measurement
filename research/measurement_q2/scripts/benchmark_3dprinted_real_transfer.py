"""Run frozen rigid front ends and scale-only risk ranking on real scans."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from SoundBarrierSystem.core.datasets import ThreeDPrintedShapesDataset
from research.measurement_q2.pcudm import PCUDMFieldEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--max-specimens", type=int, default=38)
    parser.add_argument("--voxel-size-mm", type=float, default=2.0)
    parser.add_argument("--mesh-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    return parser.parse_args()


def load_mesh(path: Path, count: int) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"empty mesh: {path}")
    cloud = mesh.sample_points_uniformly(number_of_points=count)
    return cloud, np.asarray(cloud.points, dtype=np.float64)


def load_scan(path: Path, voxel: float) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"empty scan: {path}")
    cloud = cloud.voxel_down_sample(voxel)
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.shape[0] < 20 or not np.isfinite(points).all():
        raise ValueError(f"invalid scan: {path}")
    return cloud, points


def robust_initial_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_bounds = np.quantile(source, (0.10, 0.90), axis=0)
    target_bounds = np.quantile(target, (0.10, 0.90), axis=0)
    source_centre = 0.5 * (source_bounds[0] + source_bounds[1])
    target_centre = 0.5 * (target_bounds[0] + target_bounds[1])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = target_centre - source_centre
    return transform


def multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    transform = initial.copy()
    result = None
    for threshold in (100.0, 30.0, 10.0):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        transform = result.transformation
    assert result is not None
    return transform, float(result.fitness), float(result.inlier_rmse)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def symmetric_metrics(source: np.ndarray, target: np.ndarray) -> dict:
    source_to_target = cKDTree(target).query(source, workers=-1)[0]
    target_to_source = cKDTree(source).query(target, workers=-1)[0]
    return {
        "symmetric_mean_mm": float(
            0.5 * (source_to_target.mean() + target_to_source.mean())
        ),
        "source_to_target_mean_mm": float(source_to_target.mean()),
        "target_to_source_mean_mm": float(target_to_source.mean()),
        "source_to_target_p95_mm": float(np.quantile(source_to_target, 0.95)),
        "target_to_source_p95_mm": float(np.quantile(target_to_source, 0.95)),
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)
    dataset = ThreeDPrintedShapesDataset(args.dataset_root, args.metadata_csv)
    records = list(dataset.records())[: args.max_specimens]
    total = len(records) * 3
    rows = []
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    with args.progress.open("w", encoding="utf-8", buffering=1) as progress:
        completed = 0
        for record in records:
            mesh_cloud, mesh_points = load_mesh(record.mesh_path, args.mesh_samples)
            panel_ids = np.zeros(mesh_points.shape[0], dtype=np.int32)
            support = np.ones(mesh_points.shape[0], dtype=bool)
            for scanner, variants in record.scan_paths.items():
                started = time.perf_counter()
                scan_cloud, scan_points = load_scan(variants["processed"], args.voxel_size_mm)
                direct = symmetric_metrics(mesh_points, scan_points)

                initial = robust_initial_transform(mesh_points, scan_points)
                icp_transform, icp_fitness, icp_rmse = multiscale_icp(
                    mesh_cloud, scan_cloud, initial
                )
                icp_metrics = symmetric_metrics(
                    transform_points(mesh_points, icp_transform), scan_points
                )

                cascade = PCUDMFieldEstimator(mode="cascade", icp_iterations=14).fit(
                    mesh_points, scan_points, panel_ids, support
                )
                cascade_transform = np.eye(4, dtype=np.float64)
                cascade_transform[:3, :3] = cascade.rotation
                cascade_transform[:3, 3] = cascade.translation
                cascade_metrics = symmetric_metrics(
                    transform_points(mesh_points, cascade_transform), scan_points
                )
                valid_scale = cascade.scale[cascade.valid]
                valid_match = cascade.match_distance[cascade.valid]
                rows.append({
                    "specimen_id": record.specimen_id,
                    "scanner": scanner,
                    "scan_points_downsampled": int(scan_points.shape[0]),
                    "published_coordinates": direct,
                    "multiscale_p2p_icp": {
                        **icp_metrics,
                        "fitness_10mm": icp_fitness,
                        "inlier_rmse_10mm": icp_rmse,
                        "translation_norm_mm": float(np.linalg.norm(icp_transform[:3, 3])),
                        "rotation_angle_deg": rotation_angle_deg(icp_transform[:3, :3]),
                    },
                    "cascade_strong_unmodified": {
                        **cascade_metrics,
                        "converged": cascade.converged,
                        "translation_norm_mm": float(np.linalg.norm(cascade.translation)),
                        "rotation_angle_deg": rotation_angle_deg(cascade.rotation),
                        "scale_mean_mm": float(np.mean(valid_scale)),
                        "scale_median_mm": float(np.median(valid_scale)),
                        "scale_q95_mm": float(np.quantile(valid_scale, 0.95)),
                        "match_q95_mm": float(np.quantile(valid_match, 0.95)),
                        "valid_fraction": float(cascade.valid.mean()),
                    },
                    "duration_seconds": float(time.perf_counter() - started),
                })
                completed += 1
                progress.write(json.dumps({
                    "completed": completed,
                    "total": total,
                    "specimen_id": record.specimen_id,
                    "scanner": scanner,
                }) + "\n")

    report = {
        "schema_version": "1.0",
        "status": "REAL_TRANSFER_EXECUTION_COMPLETE_UNANALYZED",
        "independent_cluster": "physical printed specimen",
        "specimens": len(records),
        "device_acquisitions": len(rows),
        "seed": args.seed,
        "voxel_size_mm": args.voxel_size_mm,
        "mesh_samples": args.mesh_samples,
        "fixed_icp_thresholds_mm": [100.0, 30.0, 10.0],
        "rows": rows,
        "claim_boundary": "single-state mesh/scan surface agreement, not displacement truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "specimens": report["specimens"],
        "device_acquisitions": report["device_acquisitions"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
