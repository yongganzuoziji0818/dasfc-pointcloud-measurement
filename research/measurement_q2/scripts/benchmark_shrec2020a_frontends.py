"""Exploratory front-end benchmark on one real deformable SHREC object."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from research.measurement_q2.pcudm import PCUDMFieldEstimator
from research.measurement_q2.pcudm.metrics import aurc
from research.measurement_q2.pcudm.registration_frontends import register_multiscale


def transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def frontend_transform(
    source: np.ndarray,
    target: np.ndarray,
    frontend: str,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--frontends",
        nargs="+",
        choices=(
            "published_coordinates",
            "cascade_strong",
            "multiscale_trimmed_ptp",
            "robust_ptpl",
        ),
    )
    args = parser.parse_args()
    frontends = tuple(args.frontends) if args.frontends else (
        "published_coordinates",
        "cascade_strong",
        "multiscale_trimmed_ptp",
        "robust_ptpl",
    )
    pair_paths = sorted(args.pairs_dir.glob("scan*_scan00.npz"))
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise ValueError("--max-pairs must be positive")
        pair_paths = pair_paths[: args.max_pairs]
    if not pair_paths:
        raise FileNotFoundError(f"no converted scan pairs under {args.pairs_dir}")
    rows = []
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    total = len(pair_paths) * len(frontends)
    completed = 0
    with args.progress.open("w", encoding="utf-8", buffering=1) as progress:
        for pair_path in pair_paths:
            with np.load(pair_path) as data:
                source = np.asarray(data["source"], dtype=np.float64)
                target = np.asarray(data["target"], dtype=np.float64)
                truth = np.asarray(data["ground_truth_target"], dtype=np.float64)
                valid_truth = np.asarray(data["ground_truth_valid"], dtype=bool)
            for frontend in frontends:
                started = time.perf_counter()
                rotation, translation, diagnostics = frontend_transform(
                    source, target, frontend, args.workers
                )
                aligned = transform(source, rotation, translation)
                match_distance, match_index = cKDTree(target).query(
                    aligned, workers=args.workers
                )
                predicted_correspondence = target[match_index]
                error = np.linalg.norm(predicted_correspondence - truth, axis=1)
                valid = valid_truth & np.isfinite(error) & np.isfinite(match_distance)
                correlation = stats.spearmanr(match_distance[valid], error[valid])
                rows.append({
                    "pair_id": pair_path.stem,
                    "frontend": frontend,
                    "valid_correspondences": int(valid.sum()),
                    "mean_correspondence_error_bbox": float(error[valid].mean()),
                    "median_correspondence_error_bbox": float(np.median(error[valid])),
                    "p95_correspondence_error_bbox": float(np.quantile(error[valid], 0.95)),
                    "mean_match_distance_bbox": float(match_distance[valid].mean()),
                    "match_error_spearman": float(correlation.statistic),
                    "match_distance_aurc": float(aurc(error[valid], match_distance[valid])),
                    "duration_seconds": float(time.perf_counter() - started),
                    "diagnostics": diagnostics,
                })
                completed += 1
                progress.write(json.dumps({
                    "completed": completed,
                    "total": total,
                    "pair_id": pair_path.stem,
                    "frontend": frontend,
                }) + "\n")
    aggregates = []
    for frontend in frontends:
        subset = [row for row in rows if row["frontend"] == frontend]
        aggregates.append({
            "frontend": frontend,
            "pair_count": len(subset),
            "mean_of_pair_mean_error_bbox": float(np.mean([
                row["mean_correspondence_error_bbox"] for row in subset
            ])),
            "median_pair_spearman": float(np.median([
                row["match_error_spearman"] for row in subset
            ])),
            "mean_pair_aurc": float(np.mean([
                row["match_distance_aurc"] for row in subset
            ])),
            "failure_count": sum(
                not row["diagnostics"].get("converged", True) for row in subset
            ),
        })
    report = {
        "schema_version": "1.0",
        "status": "SHREC2020A_EXPLORATORY_FRONTEND_BENCHMARK_COMPLETE",
        "physical_object_clusters": 1,
        "scan_pairs": len(pair_paths),
        "coordinate_unit": "full-scan bounding-box diagonal",
        "confirmatory_coverage_inference_allowed": False,
        "engineering_subset": args.max_pairs is not None or args.frontends is not None,
        "claim_boundary": (
            "Released texture-marker correspondence and failure analysis on repeated "
            "scans of one real deformable object; not dense truth and not "
            "independent-structure validation."
        ),
        "aggregates": aggregates,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "scan_pairs": report["scan_pairs"],
        "aggregates": aggregates,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
