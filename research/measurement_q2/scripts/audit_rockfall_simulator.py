"""Audit the public ETH Rockfall Simulator release without fitting a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import open3d as o3d


EXPECTED_EPOCHS = ("E0", "E1", "E2", "E3")
EXPECTED_TARGETS = ("T0", "T1", "T2", "T3", "T4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_ts_file(root: Path) -> Path:
    matches = list(root.rglob("rockfall_sim.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rockfall_sim.txt, found {len(matches)}")
    return matches[0]


def parse_ts(path: Path) -> dict[str, dict[str, np.ndarray]]:
    epochs: dict[str, dict[str, np.ndarray]] = {epoch: {} for epoch in EXPECTED_EPOCHS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            point_id = row["Point ID"].strip()
            if "_" not in point_id:
                continue
            epoch, target = point_id.split("_", 1)
            if epoch in epochs and target in EXPECTED_TARGETS:
                epochs[epoch][target] = np.array(
                    [float(row["Easting"]), float(row["Northing"]), float(row["Height"])],
                    dtype=np.float64,
                )
    return epochs


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    residual = target - (source @ rotation.T + translation)
    return rotation, translation, residual


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def audit_point_cloud(path: Path) -> dict:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise RuntimeError(f"Unreadable or empty point cloud: {path}")
    finite = bool(np.isfinite(points).all())
    result = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "point_count": int(points.shape[0]),
        "finite": finite,
        "minimum": points.min(axis=0).tolist(),
        "maximum": points.max(axis=0).tolist(),
        "extent": np.ptp(points, axis=0).tolist(),
        "centroid": points.mean(axis=0).tolist(),
        "has_colors": bool(cloud.has_colors()),
    }
    del cloud, points
    return result


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
    ts_path = find_ts_file(root)
    ts = parse_ts(ts_path)
    missing = [f"{epoch}_{target}" for epoch in EXPECTED_EPOCHS for target in EXPECTED_TARGETS if target not in ts[epoch]]

    t0_stack = np.stack([ts[epoch]["T0"] for epoch in EXPECTED_EPOCHS]) if not missing else np.empty((0, 3))
    t0_range_mm = float(np.max(np.linalg.norm(t0_stack[:, None] - t0_stack[None, :], axis=2)) * 1000.0) if len(t0_stack) else None

    pair_rows = []
    for source_epoch, target_epoch in combinations(EXPECTED_EPOCHS, 2):
        source = np.stack([ts[source_epoch][target] for target in EXPECTED_TARGETS[1:]])
        target = np.stack([ts[target_epoch][target] for target in EXPECTED_TARGETS[1:]])
        rotation, translation, residual = rigid_fit(source, target)
        displacements = target - source
        pair_rows.append(
            {
                "source_epoch": source_epoch,
                "target_epoch": target_epoch,
                "rotation_angle_deg": rotation_angle_deg(rotation),
                "translation_vector_ts_m": translation.tolist(),
                "target_displacement_norms_mm": (np.linalg.norm(displacements, axis=1) * 1000.0).tolist(),
                "rigid_fit_rms_mm": float(np.sqrt(np.mean(np.sum(residual**2, axis=1))) * 1000.0),
                "rigid_fit_max_mm": float(np.max(np.linalg.norm(residual, axis=1)) * 1000.0),
                "rotation_ts": rotation.tolist(),
            }
        )

    cloud_candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ply", ".pcd"} and "raw_pcd" in path.parts
    )
    cloud_rows = [audit_point_cloud(path) for path in cloud_candidates]
    expected_revision = protocol["dataset"]["revision"]
    actual_revision = git_revision(root)
    reference_gate = protocol["reference_gate"]
    consecutive = {("E0", "E1"), ("E1", "E2"), ("E2", "E3")}
    consecutive_rows = [row for row in pair_rows if (row["source_epoch"], row["target_epoch"]) in consecutive]

    gates = {
        "D0_INTEGRITY": actual_revision == expected_revision and len(files) > 0,
        "D1_LAYOUT": not missing and len(cloud_rows) == 4 and all(row["finite"] for row in cloud_rows),
        "D2_REFERENCE": (
            t0_range_mm is not None
            and t0_range_mm <= reference_gate["stable_t0_max_range_mm"]
            and all(row["rigid_fit_rms_mm"] <= reference_gate["moving_rigid_fit_max_rms_mm"] for row in consecutive_rows)
        ),
        "D3_FRAME": False,
    }
    report = {
        "schema_version": "1.0",
        "dataset_id": "eth-rockfall-simulator",
        "dataset_revision_expected": expected_revision,
        "dataset_revision_actual": actual_revision,
        "protocol_id": protocol["protocol_id"],
        "file_count_excluding_git": len(files),
        "total_bytes_excluding_git": sum(path.stat().st_size for path in files),
        "documentation_files": [path.relative_to(root).as_posix() for path in files if path.suffix.lower() == ".md"],
        "ts_file": ts_path.relative_to(root).as_posix(),
        "ts_missing_expected_ids": missing,
        "t0_max_epoch_range_mm": t0_range_mm,
        "ts_pair_rigid_fits": pair_rows,
        "tls_point_clouds": cloud_rows,
        "gates": gates,
        "D3_FRAME_reason": "Fail closed until a documented or independent TS-to-TLS frame/target mapping is verified.",
        "claim_boundary": {
            "real_physical_dual_epoch_tls": bool(gates["D0_INTEGRITY"] and gates["D1_LAYOUT"]),
            "independent_sparse_ts_reference": bool(gates["D2_REFERENCE"]),
            "vector_error_or_interval_coverage_at_ts_targets": False,
            "dense_real_nonrigid_truth": False,
            "nominal_95pct_real_simultaneous_coverage": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if gates["D0_INTEGRITY"] and gates["D1_LAYOUT"] and gates["D2_REFERENCE"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

