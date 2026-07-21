"""Measure direct digital-mesh to processed-scan agreement for audited specimens.

The processed point clouds are published after ground removal, position adjustment,
and denoising. This script intentionally does not learn from the data and does not
apply a non-rigid warp. It reports the published alignment first; optional rigid ICP
can be added later as an explicit baseline rather than hidden preprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from SoundBarrierSystem.core.datasets import ThreeDPrintedShapesDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--max-specimens", type=int, default=3)
    parser.add_argument("--voxel-size-mm", type=float, default=2.0)
    parser.add_argument("--mesh-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize_distances(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Distances must be a non-empty finite vector")
    return {
        "count": int(values.size),
        "mean_mm": float(values.mean()),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.quantile(values, 0.90)),
        "p95_mm": float(np.quantile(values, 0.95)),
        "p99_mm": float(np.quantile(values, 0.99)),
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "max_mm": float(values.max()),
    }


def load_mesh_samples(path: Path, count: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {path}")
    cloud = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(cloud.points, dtype=np.float64)


def load_processed_scan(path: Path, voxel_size_mm: float) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    down = cloud.voxel_down_sample(voxel_size=voxel_size_mm)
    points = np.asarray(down.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Voxel downsampling removed all points: {path}")
    return points


def main() -> int:
    args = parse_args()
    if args.voxel_size_mm <= 0 or args.mesh_samples < 1:
        raise ValueError("voxel size and mesh sample count must be positive")

    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)
    dataset = ThreeDPrintedShapesDataset(args.dataset_root, args.metadata_csv)
    records = list(dataset.records())[: max(0, args.max_specimens)]
    rows = []

    for record in records:
        mesh_points = load_mesh_samples(record.mesh_path, args.mesh_samples)
        mesh_tree = cKDTree(mesh_points)
        for scanner, variants in record.scan_paths.items():
            scan_path = variants["processed"]
            scan_points = load_processed_scan(scan_path, args.voxel_size_mm)
            scan_tree = cKDTree(scan_points)
            scan_to_mesh = mesh_tree.query(scan_points, workers=-1)[0]
            mesh_to_scan = scan_tree.query(mesh_points, workers=-1)[0]
            rows.append(
                {
                    "specimen_id": record.specimen_id,
                    "scanner": scanner,
                    "mesh_path": str(record.mesh_path),
                    "scan_path": str(scan_path),
                    "voxel_size_mm": args.voxel_size_mm,
                    "mesh_samples": int(mesh_points.shape[0]),
                    "scan_points_downsampled": int(scan_points.shape[0]),
                    "scan_to_mesh": summarize_distances(scan_to_mesh),
                    "mesh_to_scan": summarize_distances(mesh_to_scan),
                    "symmetric_mean_mm": float(
                        0.5 * (scan_to_mesh.mean() + mesh_to_scan.mean())
                    ),
                }
            )

    report = {
        "schema_version": "1.0",
        "purpose": "direct processed-scan to printing-mesh agreement audit",
        "registration": "none; published processed coordinates",
        "seed": args.seed,
        "specimens": len(records),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
