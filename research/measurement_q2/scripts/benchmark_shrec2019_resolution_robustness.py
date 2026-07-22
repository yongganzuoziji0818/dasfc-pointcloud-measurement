"""Run the frozen SHREC'19 unlabelled resolution-robustness benchmark."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from research.measurement_q2.scripts.registration_utils import (
    frontend_transform,
    transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    return parser.parse_args()


def load_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 20:
        raise ValueError(f"invalid or empty OBJ vertices: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"non-finite OBJ vertices: {path}")
    return points


def parse_test_sets(dataset_root: Path) -> tuple[list[dict], list[str]]:
    resolutions = ("hires", "lores")
    definitions = {}
    duplicate_warnings = []
    for resolution in resolutions:
        rows = []
        for test_set in range(4):
            path = dataset_root / f"SHREC19_{resolution}" / "test-sets" / f"test-set{test_set}.txt"
            seen = set()
            for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                source, target = [field.strip() for field in raw_line.split(",")]
                key = (test_set, source, target)
                if key in seen:
                    duplicate_warnings.append(
                        f"{resolution}:test_set_{test_set}:{source},{target}:line_{line_number}"
                    )
                    continue
                seen.add(key)
                rows.append({"test_set": test_set, "source": source, "target": target})
        definitions[resolution] = rows
    if definitions["hires"] != definitions["lores"]:
        raise ValueError("high- and low-resolution test definitions differ")
    return definitions["hires"], duplicate_warnings


def common_normalization(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    combined = np.vstack((source, target))
    bounds = np.quantile(combined, (0.10, 0.90), axis=0)
    center = 0.5 * (bounds[0] + bounds[1])
    scale = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("invalid pair normalization scale")
    return center, scale


def surface_fit_metrics(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    workers: int,
) -> dict:
    aligned = transform(source, rotation, translation)
    source_to_target = cKDTree(target).query(aligned, workers=workers)[0]
    target_to_source = cKDTree(aligned).query(target, workers=workers)[0]
    symmetric = np.concatenate((source_to_target, target_to_source))
    return {
        "source_to_target_mean": float(source_to_target.mean()),
        "source_to_target_p95": float(np.quantile(source_to_target, 0.95)),
        "target_to_source_mean": float(target_to_source.mean()),
        "symmetric_mean": float(0.5 * (source_to_target.mean() + target_to_source.mean())),
        "symmetric_median": float(np.median(symmetric)),
        "symmetric_p95": float(np.quantile(symmetric, 0.95)),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_root = args.dataset_root.resolve()
    pairs, duplicate_warnings = parse_test_sets(dataset_root)
    expected_pairs = int(config["dataset"]["expected_unique_pairs"])
    if len(pairs) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} unique pairs, observed {len(pairs)}")
    if args.mode == "smoke":
        selected = set(config["design"]["smoke_pairs"])
        pairs = [
            pair for pair in pairs
            if f"{pair['test_set']}:{pair['source']}:{pair['target']}" in selected
        ]
        if len(pairs) != len(selected):
            raise ValueError("smoke-pair selection did not resolve exactly")

    frontends = tuple(config["design"]["frontends"])
    resolutions = tuple(config["design"]["resolutions"])
    directions = tuple(config["design"]["directions"])
    workers = int(config["design"]["workers"])
    expected_rows = len(pairs) * len(frontends) * len(resolutions) * len(directions)
    rows = []
    cache: dict[tuple[str, str], np.ndarray] = {}

    def vertices(resolution: str, scan_id: str) -> np.ndarray:
        key = (resolution, scan_id)
        if key not in cache:
            cache[key] = load_vertices(
                dataset_root / f"SHREC19_{resolution}" / "models" / f"scan_{scan_id}.obj"
            )
        return cache[key]

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.progress.open("w", encoding="utf-8", buffering=1) as progress:
        for pair in pairs:
            pair_key = f"{pair['test_set']}:{pair['source']}:{pair['target']}"
            hires_source = vertices("hires", pair["source"])
            hires_target = vertices("hires", pair["target"])
            center, scale = common_normalization(hires_source, hires_target)
            for resolution in resolutions:
                normalized = {
                    pair["source"]: (vertices(resolution, pair["source"]) - center) / scale,
                    pair["target"]: (vertices(resolution, pair["target"]) - center) / scale,
                }
                for direction in directions:
                    source_id, target_id = (
                        (pair["source"], pair["target"])
                        if direction == "forward"
                        else (pair["target"], pair["source"])
                    )
                    source = normalized[source_id]
                    target = normalized[target_id]
                    for frontend in frontends:
                        started = time.perf_counter()
                        row = {
                            "pair_key": pair_key,
                            "test_set": pair["test_set"],
                            "source_scan": pair["source"],
                            "target_scan": pair["target"],
                            "resolution": resolution,
                            "direction": direction,
                            "frontend": frontend,
                            "source_vertices": int(source.shape[0]),
                            "target_vertices": int(target.shape[0]),
                            "normalization_scale_source_units": scale,
                        }
                        try:
                            rotation, translation, diagnostics = frontend_transform(
                                source, target, frontend, workers
                            )
                            metrics = surface_fit_metrics(
                                source, target, rotation, translation, workers
                            )
                            row.update({
                                "status": "completed",
                                "rotation": rotation.tolist(),
                                "translation": translation.tolist(),
                                "fit": metrics,
                                "diagnostics": diagnostics,
                            })
                        except Exception as exc:  # failure is retained as an outcome
                            row.update({
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            })
                        row["duration_seconds"] = float(time.perf_counter() - started)
                        rows.append(row)
                        completed += 1
                        progress.write(json.dumps({
                            "completed": completed,
                            "total": expected_rows,
                            "pair_key": pair_key,
                            "resolution": resolution,
                            "direction": direction,
                            "frontend": frontend,
                            "status": row["status"],
                        }) + "\n")

    completed_rows = sum(row["status"] == "completed" for row in rows)
    report = {
        "schema_version": "1.0",
        "protocol_id": config["protocol_id"],
        "status": "SHREC2019_RESOLUTION_ROBUSTNESS_EXECUTION_COMPLETE",
        "mode": args.mode,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "coordinate_unit": "dimensionless common pair normalization",
        "pair_count": len(pairs),
        "expected_rows": expected_rows,
        "attempted_rows": len(rows),
        "completed_rows": completed_rows,
        "failed_rows": len(rows) - completed_rows,
        "duplicate_warnings": duplicate_warnings,
        "claim_boundary": (
            "Observable fit and high/low representation robustness on unlabelled real meshes; "
            "not correspondence truth, displacement truth, or conformal coverage."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "mode": args.mode,
        "pairs": len(pairs),
        "attempted_rows": len(rows),
        "completed_rows": completed_rows,
        "failed_rows": len(rows) - completed_rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
