"""Run frozen DAS-FC front ends on mapped ETH Rockfall physical events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from research.measurement_q2.pcudm import PCUDMFieldEstimator
from research.measurement_q2.pcudm.registration_frontends import register_multiscale

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPOCHS = ("E0", "E1", "E2", "E3")
TARGETS = ("T0", "T1", "T2", "T3", "T4")
MOVING_TARGETS = ("T1", "T2", "T3", "T4")
PLY_NAMES = {
    "E0": "epoch_1_raw.ply",
    "E1": "epoch_2_raw.ply",
    "E2": "epoch_3_raw.ply",
    "E3": "epoch_4_raw.ply",
}
PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--mapping-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().decode("ascii").strip() != "ply":
            raise ValueError(f"Not PLY: {path}")
        fmt = None
        count = None
        in_vertex = False
        properties: list[tuple[str, str]] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            parts = line.split()
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
                in_vertex = True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif parts[:1] == ["property"] and in_vertex:
                if parts[1] == "list" or parts[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported vertex property: {line}")
                properties.append((parts[2], PLY_TYPES[parts[1]]))
            elif line == "end_header":
                offset = handle.tell()
                break
    if fmt != "binary_little_endian" or count is None:
        raise ValueError(f"Unexpected PLY format: {path}")
    table = np.memmap(path, mode="r", dtype=np.dtype(properties), offset=offset, shape=(count,))
    xyz = np.column_stack([table["x"], table["y"], table["z"]]).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite XYZ: {path}")
    return xyz


def load_ts(path: Path) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, dict[str, np.ndarray]] = {epoch: {} for epoch in EPOCHS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 4 or "_" not in row[0]:
                continue
            epoch, target = row[0].strip().split("_", 1)
            if epoch in values and target in TARGETS:
                values[epoch][target] = np.asarray(row[1:4], dtype=np.float64)
    return values


def load_markers(path: Path) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, dict[str, np.ndarray]] = {epoch: {} for epoch in EPOCHS}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["epoch"]][row["target"]] = np.asarray(
                [row["tls_marker_x_m"], row["tls_marker_y_m"], row["tls_marker_z_m"]],
                dtype=np.float64,
            )
    missing = [f"{e}_{t}" for e in EPOCHS for t in TARGETS if t not in values[e]]
    if missing:
        raise ValueError(f"Missing mapped markers: {missing}")
    return values


def canonical_frame(markers_e0: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    origin = np.mean([markers_e0[t] for t in MOVING_TARGETS], axis=0)
    left = 0.5 * (markers_e0["T1"] + markers_e0["T4"])
    right = 0.5 * (markers_e0["T2"] + markers_e0["T3"])
    bottom = 0.5 * (markers_e0["T3"] + markers_e0["T4"])
    top = 0.5 * (markers_e0["T1"] + markers_e0["T2"])
    x_axis = right - left
    x_axis /= np.linalg.norm(x_axis)
    z_axis = top - bottom
    z_axis -= x_axis * np.dot(z_axis, x_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(basis) < 0.999:
        raise ValueError("Canonical frame is not right-handed orthonormal")
    return origin, basis


def voxel_first(points: np.ndarray, size_mm: float) -> np.ndarray:
    keys = np.floor(points / size_mm).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def prepare_epoch(
    xyz_m: np.ndarray,
    origin_m: np.ndarray,
    basis: np.ndarray,
    moving_bounds: np.ndarray,
    prep: dict,
) -> np.ndarray:
    local_mm = ((xyz_m - origin_m) @ basis) * 1000.0
    context = float(prep["context_margin_mm"])
    normal_half = float(prep["normal_halfwidth_mm"])
    mask = (
        (local_mm[:, 0] >= moving_bounds[0, 0] - context)
        & (local_mm[:, 0] <= moving_bounds[1, 0] + context)
        & (local_mm[:, 2] >= moving_bounds[0, 1] - context)
        & (local_mm[:, 2] <= moving_bounds[1, 1] + context)
        & (np.abs(local_mm[:, 1]) <= normal_half)
    )
    cropped = local_mm[mask]
    if len(cropped) < 1000:
        raise ValueError("Context crop is unexpectedly small")
    return voxel_first(cropped, float(prep["voxel_size_mm"]))


def target_truth_vectors(
    ts: dict[str, dict[str, np.ndarray]],
    source_epoch: str,
    target_epoch: str,
    ts_to_tls_rotation: np.ndarray,
    canonical_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = {}
    for target in TARGETS:
        delta_ts = ts[target_epoch][target] - ts[source_epoch][target]
        delta_tls = delta_ts @ ts_to_tls_rotation.T
        truth[target] = delta_tls @ canonical_basis * 1000.0
    return truth


def interval_score(truth: float, lower: float, upper: float, alpha: float) -> float:
    score = upper - lower
    if truth < lower:
        score += (2.0 / alpha) * (lower - truth)
    elif truth > upper:
        score += (2.0 / alpha) * (truth - upper)
    return float(score)


def run_frontend(
    reference: np.ndarray,
    target: np.ndarray,
    panel_ids: np.ndarray,
    support: np.ndarray,
    frontend: str,
    estimator_cfg: dict,
) -> tuple[object, dict]:
    initial = None
    frontend_diagnostics = {}
    if frontend != "cascade_strong":
        initial = register_multiscale(
            reference,
            target,
            frontend,
            thresholds=tuple(estimator_cfg["multiscale_thresholds_mm"]),
        )
        frontend_diagnostics = {
            "registration_fitness": initial.fitness,
            "registration_inlier_rmse_mm": initial.inlier_rmse,
        }
    result = PCUDMFieldEstimator(
        mode=estimator_cfg["mode"],
        icp_iterations=int(estimator_cfg["icp_iterations"]),
        trim_fraction=float(estimator_cfg["trim_fraction"]),
        ridge=float(estimator_cfg["ridge"]),
        scale_neighbors=int(estimator_cfg["scale_neighbors"]),
        scale_floor_mm=float(estimator_cfg["scale_floor_mm"]),
        query_workers=int(estimator_cfg["query_workers"]),
    ).fit(
        reference,
        target,
        panel_ids,
        support,
        initial_rotation=None if initial is None else initial.rotation,
        initial_translation=None if initial is None else initial.translation,
        pose_locked=initial is not None,
    )
    return result, frontend_diagnostics


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mapping = json.loads((args.mapping_dir / "frame_mapping.json").read_text(encoding="utf-8"))
    if not mapping["gates"]["D3_FRAME_V1"]:
        raise ValueError("D3 frame mapping did not pass")
    markers = load_markers(args.mapping_dir / "frame_residuals.csv")
    ts = load_ts(args.dataset_root / "02_ExportedData" / "01_TS" / "rockfall_sim.txt")
    origin_m, basis = canonical_frame(markers["E0"])
    marker_local_e0 = {
        target: (markers["E0"][target] - origin_m) @ basis * 1000.0
        for target in TARGETS
    }
    moving_xz = np.asarray([[marker_local_e0[t][0], marker_local_e0[t][2]] for t in MOVING_TARGETS])
    moving_bounds = np.stack([moving_xz.min(axis=0), moving_xz.max(axis=0)])

    reports = {}
    quantiles = {}
    for frontend, spec in config["fallback"]["reports"].items():
        report_path = args.repo_root / spec["path"]
        observed_hash = sha256(report_path)
        if observed_hash != spec["sha256"]:
            raise ValueError(f"Frozen report hash mismatch for {frontend}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        observed_q = float(report["calibration"]["homoscedastic"]["pooled_quantile"])
        if not np.isclose(observed_q, spec["homoscedastic_pooled_quantile"], rtol=0, atol=1e-12):
            raise ValueError(f"Frozen fallback quantile mismatch for {frontend}")
        reports[frontend] = {"sha256": observed_hash, "path": spec["path"]}
        quantiles[frontend] = observed_q

    pairs = [tuple(pair) for pair in config["pairs"]]
    frontends = list(config["frontends"])
    run_mode = "formal_evidence"
    if args.smoke:
        pairs = [pairs[0]]
        frontends = ["cascade_strong"]
        run_mode = "engineering_smoke_not_evidence"

    required_epochs = sorted(set(epoch for pair in pairs for epoch in pair))
    prepared = {}
    prep = config["preparation"]
    for epoch in required_epochs:
        path = args.dataset_root / "02_ExportedData" / "02_TLS" / "raw_pcd" / PLY_NAMES[epoch]
        prepared[epoch] = prepare_epoch(read_ply_xyz(path), origin_m, basis, moving_bounds, prep)

    moving_margin = float(prep["moving_margin_mm"])
    rows = []
    event_rows = []
    field_panels = []
    for source_epoch, target_epoch in pairs:
        reference = prepared[source_epoch]
        target_cloud = prepared[target_epoch]
        moving = (
            (reference[:, 0] >= moving_bounds[0, 0] - moving_margin)
            & (reference[:, 0] <= moving_bounds[1, 0] + moving_margin)
            & (reference[:, 2] >= moving_bounds[0, 1] - moving_margin)
            & (reference[:, 2] <= moving_bounds[1, 1] + moving_margin)
        )
        support = ~moving
        if moving.sum() < prep["minimum_moving_reference_points"]:
            raise ValueError("Insufficient moving-region reference points")
        if support.sum() < prep["minimum_support_reference_points"]:
            raise ValueError("Insufficient support reference points")
        panel_ids = moving.astype(np.int32)
        truth = target_truth_vectors(
            ts,
            source_epoch,
            target_epoch,
            np.asarray(mapping["rotation_ts_to_tls"], dtype=np.float64),
            basis,
        )
        marker_source_local = {
            key: (markers[source_epoch][key] - origin_m) @ basis * 1000.0
            for key in TARGETS
        }
        moving_indices = np.flatnonzero(moving)
        interpolation_tree = cKDTree(reference[moving][:, (0, 2)])
        for frontend in frontends:
            started = time.perf_counter()
            result, frontend_diag = run_frontend(
                reference, target_cloud, panel_ids, support, frontend, config["estimator"]
            )
            elapsed = time.perf_counter() - started
            if not result.converged:
                raise RuntimeError(f"Estimator did not converge: {source_epoch}->{target_epoch}/{frontend}")
            scale_value = float(np.median(result.scale[result.valid]))
            radius = quantiles[frontend] * scale_value
            target_coverages = []
            target_scores = []
            vector_errors = []
            normal_errors = []
            for target_name in MOVING_TARGETS:
                location = marker_source_local[target_name]
                distances, neighbours = interpolation_tree.query(
                    location[[0, 2]], k=int(prep["interpolation_neighbors"])
                )
                distances = np.atleast_1d(distances)
                neighbours = np.atleast_1d(neighbours)
                weights = 1.0 / np.maximum(distances, 1e-6)
                weights /= weights.sum()
                global_indices = moving_indices[neighbours]
                normal_field = float(np.sum(weights * result.normal_displacement[global_indices]))
                deformed = location.copy()
                deformed[1] += normal_field
                predicted_location = deformed @ result.rotation.T + result.translation
                predicted_vector = predicted_location - location
                reference_vector = truth[target_name]
                error_vector = predicted_vector - reference_vector
                vector_error = float(np.linalg.norm(error_vector))
                normal_error = float(predicted_vector[1] - reference_vector[1])
                lower = float(predicted_vector[1] - radius)
                upper = float(predicted_vector[1] + radius)
                covered = bool(lower <= reference_vector[1] <= upper)
                score = interval_score(float(reference_vector[1]), lower, upper, config["alpha"])
                target_coverages.append(covered)
                target_scores.append(score)
                vector_errors.append(vector_error)
                normal_errors.append(abs(normal_error))
                rows.append({
                    "event": f"{source_epoch}->{target_epoch}",
                    "frontend": frontend,
                    "target": target_name,
                    "reference_dx_mm": float(reference_vector[0]),
                    "reference_dy_normal_mm": float(reference_vector[1]),
                    "reference_dz_mm": float(reference_vector[2]),
                    "predicted_dx_mm": float(predicted_vector[0]),
                    "predicted_dy_normal_mm": float(predicted_vector[1]),
                    "predicted_dz_mm": float(predicted_vector[2]),
                    "vector_error_mm": vector_error,
                    "normal_error_mm": normal_error,
                    "interval_lower_normal_mm": lower,
                    "interval_upper_normal_mm": upper,
                    "interval_radius_mm": float(radius),
                    "covered_normal": covered,
                    "interval_score_mm": score,
                })
            event_rows.append({
                "event": f"{source_epoch}->{target_epoch}",
                "frontend": frontend,
                "reference_points": int(len(reference)),
                "target_points": int(len(target_cloud)),
                "moving_reference_points": int(moving.sum()),
                "support_reference_points": int(support.sum()),
                "converged": bool(result.converged),
                "valid_fraction": float(result.valid.mean()),
                "median_raw_scale_mm": scale_value,
                "fallback_quantile": quantiles[frontend],
                "interval_radius_mm": float(radius),
                "simultaneous_covered_normal_four_targets": bool(all(target_coverages)),
                "covered_targets": int(sum(target_coverages)),
                "max_vector_error_mm": float(max(vector_errors)),
                "mean_vector_error_mm": float(np.mean(vector_errors)),
                "normal_mae_mm": float(np.mean(normal_errors)),
                "mean_interval_width_mm": float(2.0 * radius),
                "mean_interval_score_mm": float(np.mean(target_scores)),
                "duration_seconds": float(elapsed),
                "fallback_reason": config["fallback"]["reason"],
                **frontend_diag,
            })
            field_panels.append({
                "event": f"{source_epoch}->{target_epoch}",
                "frontend": frontend,
                "x": reference[moving, 0],
                "z": reference[moving, 2],
                "normal": result.normal_displacement[moving],
            })

    for name, payload in (("target_results.csv", rows), ("event_results.csv", event_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in payload for key in row)))
            writer.writeheader()
            writer.writerows(payload)

    columns = len(field_panels)
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.0), squeeze=False, constrained_layout=True)
    for axis, payload in zip(axes.flat, field_panels):
        scatter = axis.scatter(payload["x"], payload["z"], c=payload["normal"], s=2, cmap="coolwarm")
        axis.set_title(f"{payload['event']}\n{payload['frontend']}")
        axis.set_xlabel("canonical x (mm)")
        axis.set_ylabel("canonical z (mm)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=axis, label="predicted normal displacement (mm)")
    fig.savefig(args.output_dir / "predicted_normal_fields.png", dpi=220)
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "run_mode": run_mode,
        "dataset_revision": config["dataset_revision"],
        "algorithm_outputs_accessed": True,
        "calibration_or_tuning_on_rockfall": False,
        "frame_basis_tls_to_canonical": basis.tolist(),
        "frame_origin_tls_m": origin_m.tolist(),
        "moving_bounds_xz_mm": moving_bounds.tolist(),
        "frozen_fallback_reports": reports,
        "events": event_rows,
        "target_rows": rows,
        "claim_boundary": {
            "three_dimensional_point_error": True,
            "normal_component_interval": True,
            "calibrated_three_dimensional_vector_region": False,
            "population_nominal_coverage": False,
            "independent_apparatus_count": 1,
        },
    }
    (args.output_dir / "physical_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
