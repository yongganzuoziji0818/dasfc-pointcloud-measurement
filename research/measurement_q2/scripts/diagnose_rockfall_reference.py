"""Diagnose the ETH Rockfall TS/TLS reference chain without algorithm output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.cluster import DBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPOCHS = ("E0", "E1", "E2", "E3")
TARGETS = ("T1", "T2", "T3", "T4")
PAIRS = (("E0", "E1"), ("E1", "E2"), ("E2", "E3"))
PLY_NAMES = {
    "E0": "epoch_1_raw.ply",
    "E1": "epoch_2_raw.ply",
    "E2": "epoch_3_raw.ply",
    "E3": "epoch_4_raw.ply",
}
PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_ts(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, dict[str, np.ndarray]] = {epoch: {} for epoch in EPOCHS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 4 or "_" not in row[0]:
                continue
            epoch, target = row[0].strip().split("_", 1)
            if epoch in rows and target in ("T0", *TARGETS):
                rows[epoch][target] = np.asarray(row[1:4], dtype=np.float64)
    missing = [f"{e}_{t}" for e in EPOCHS for t in ("T0", *TARGETS) if t not in rows[e]]
    if missing:
        raise ValueError(f"Missing TS coordinates: {missing}")
    return rows


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def ts_diagnostics(ts: dict[str, dict[str, np.ndarray]]) -> tuple[list[dict], list[dict], dict]:
    distance_rows: list[dict] = []
    loo_rows: list[dict] = []
    pair_summary: dict[str, dict] = {}
    for source_epoch, target_epoch in PAIRS:
        source = np.stack([ts[source_epoch][t] for t in TARGETS])
        target = np.stack([ts[target_epoch][t] for t in TARGETS])
        rotation, translation = kabsch(source, target)
        predicted = (rotation @ source.T).T + translation
        residual_mm = np.linalg.norm(predicted - target, axis=1) * 1000.0
        label = f"{source_epoch}->{target_epoch}"
        changes = []
        for left, right in combinations(range(len(TARGETS)), 2):
            before_mm = float(np.linalg.norm(source[left] - source[right]) * 1000.0)
            after_mm = float(np.linalg.norm(target[left] - target[right]) * 1000.0)
            change_mm = after_mm - before_mm
            changes.append(change_mm)
            distance_rows.append(
                {
                    "event": label,
                    "target_pair": f"{TARGETS[left]}-{TARGETS[right]}",
                    "source_distance_mm": before_mm,
                    "target_distance_mm": after_mm,
                    "change_mm": change_mm,
                    "abs_change_mm": abs(change_mm),
                }
            )
        loo_errors = []
        for omitted in range(len(TARGETS)):
            keep = [idx for idx in range(len(TARGETS)) if idx != omitted]
            loo_rotation, loo_translation = kabsch(source[keep], target[keep])
            loo_prediction = loo_rotation @ source[omitted] + loo_translation
            error_vector_mm = (loo_prediction - target[omitted]) * 1000.0
            error_mm = float(np.linalg.norm(error_vector_mm))
            loo_errors.append(error_mm)
            loo_rows.append(
                {
                    "event": label,
                    "omitted_target": TARGETS[omitted],
                    "prediction_error_mm": error_mm,
                    "error_x_mm": float(error_vector_mm[0]),
                    "error_y_mm": float(error_vector_mm[1]),
                    "error_z_mm": float(error_vector_mm[2]),
                }
            )
        trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        pair_summary[label] = {
            "rotation_angle_deg": math.degrees(math.acos(trace)),
            "rigid_fit_residuals_mm": dict(zip(TARGETS, residual_mm.tolist())),
            "rigid_fit_rms_mm": float(np.sqrt(np.mean(residual_mm**2))),
            "rigid_fit_max_mm": float(residual_mm.max()),
            "pairwise_distance_changes_mm": changes,
            "pairwise_abs_change_max_mm": float(np.max(np.abs(changes))),
            "loo_prediction_errors_mm": dict(zip(TARGETS, loo_errors)),
            "loo_prediction_max_mm": float(max(loo_errors)),
            "ideal_rigid_within_1mm": bool(
                np.max(np.abs(changes)) <= 1.0 and max(loo_errors) <= 1.0
            ),
        }
    t0 = np.stack([ts[epoch]["T0"] for epoch in EPOCHS])
    t0_ranges = [float(np.ptp(t0[:, axis]) * 1000.0) for axis in range(3)]
    reference_summary = {
        "t0_axis_ranges_mm": t0_ranges,
        "t0_max_pairwise_range_mm": float(
            max(np.linalg.norm(a - b) for a, b in combinations(t0, 2)) * 1000.0
        ),
        "consecutive_events": pair_summary,
    }
    return distance_rows, loo_rows, reference_summary


def read_ply_vertex_table(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        vertex_count = None
        in_vertex = False
        properties: list[tuple[str, str]] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Truncated PLY header: {path}")
            line = raw.decode("ascii").strip()
            parts = line.split()
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif parts[:1] == ["property"] and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List properties are unsupported in vertex records")
                if parts[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY type: {parts[1]}")
                properties.append((parts[2], PLY_TYPES[parts[1]]))
            elif line == "end_header":
                offset = handle.tell()
                break
    if fmt != "binary_little_endian" or vertex_count is None:
        raise ValueError(f"Expected binary_little_endian vertices: {path}")
    dtype = np.dtype(properties)
    expected = offset + vertex_count * dtype.itemsize
    if path.stat().st_size < expected:
        raise ValueError(f"PLY payload shorter than header declares: {path}")
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(vertex_count,))


def clusters_for(points: np.ndarray, mask: np.ndarray, channel: str) -> list[dict]:
    selected = points[mask]
    if len(selected) < 8:
        return []
    labels = DBSCAN(eps=0.006, min_samples=8, n_jobs=1).fit_predict(selected)
    rows = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        cluster = selected[labels == label]
        if not 8 <= len(cluster) <= 20_000:
            continue
        center = np.median(cluster, axis=0)
        rows.append(
            {
                "channel": channel,
                "cluster_id": int(label),
                "point_count": int(len(cluster)),
                "center_x_m": float(center[0]),
                "center_y_m": float(center[1]),
                "center_z_m": float(center[2]),
                "extent_x_m": float(np.ptp(cluster[:, 0])),
                "extent_y_m": float(np.ptp(cluster[:, 1])),
                "extent_z_m": float(np.ptp(cluster[:, 2])),
            }
        )
    return rows


def tls_diagnostics(dataset_root: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    tls_root = dataset_root / "02_ExportedData" / "02_TLS" / "raw_pcd"
    epoch_rows: list[dict] = []
    cluster_rows: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 11), constrained_layout=True)
    for axis, epoch in zip(axes.flat, EPOCHS):
        table = read_ply_vertex_table(tls_root / PLY_NAMES[epoch])
        required = {"x", "y", "z", "red", "green", "blue", "scalar_Intensity"}
        if not required.issubset(table.dtype.names or ()):
            raise ValueError(f"Missing required PLY properties in {PLY_NAMES[epoch]}")
        xyz = np.column_stack([table["x"], table["y"], table["z"]]).astype(np.float64)
        rgb = np.column_stack([table["red"], table["green"], table["blue"]]).astype(np.uint8)
        intensity = np.asarray(table["scalar_Intensity"], dtype=np.float64)
        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
        if not finite.all():
            raise ValueError(f"Non-finite TLS values in {PLY_NAMES[epoch]}")
        quantile_levels = np.asarray([0.0, 0.5, 0.9, 0.99, 0.999, 1.0])
        quantiles = np.quantile(intensity, quantile_levels)
        intensity_mask = intensity >= quantiles[-2]
        blue_mask = (
            (rgb[:, 2].astype(np.int16) >= 50)
            & (rgb[:, 2].astype(np.int16) - rgb[:, 0].astype(np.int16) >= 15)
            & (rgb[:, 2].astype(np.int16) - rgb[:, 1].astype(np.int16) >= 10)
        )
        rows = clusters_for(xyz, intensity_mask, "intensity_q999")
        rows.extend(clusters_for(xyz, blue_mask, "blue_dominant"))
        for row in rows:
            row["epoch"] = epoch
        cluster_rows.extend(rows)
        epoch_rows.append(
            {
                "epoch": epoch,
                "point_count": int(len(table)),
                "xyz_min_m": xyz.min(axis=0).tolist(),
                "xyz_max_m": xyz.max(axis=0).tolist(),
                "rgb_min": rgb.min(axis=0).astype(int).tolist(),
                "rgb_max": rgb.max(axis=0).astype(int).tolist(),
                "intensity_quantile_levels": quantile_levels.tolist(),
                "intensity_quantiles": quantiles.tolist(),
                "intensity_candidate_count": int(intensity_mask.sum()),
                "blue_candidate_count": int(blue_mask.sum()),
                "retained_cluster_count": len(rows),
            }
        )
        stride = max(1, len(table) // 200_000)
        sample_xyz = xyz[::stride][:200_000]
        sample_rgb = rgb[::stride][:200_000].astype(np.float64) / 255.0
        axis.scatter(sample_xyz[:, 0], sample_xyz[:, 2], c=sample_rgb, s=0.15, linewidths=0)
        axis.set_title(f"{epoch} / {PLY_NAMES[epoch]}")
        axis.set_xlabel("TLS x (m)")
        axis.set_ylabel("TLS z (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(False)
        del table, xyz, rgb, intensity
    fig.suptitle("Raw TLS X-Z projections (diagnostic only)")
    fig.savefig(output_dir / "tls_raw_xz_projection.png", dpi=220)
    plt.close(fig)
    return epoch_rows, cluster_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ts = load_ts(args.dataset_root / "02_ExportedData" / "01_TS" / "rockfall_sim.txt")
    distance_rows, loo_rows, reference_summary = ts_diagnostics(ts)
    tls_rows, cluster_rows = tls_diagnostics(args.dataset_root, args.output_dir)
    write_csv(args.output_dir / "ts_pairwise_distance_changes.csv", distance_rows)
    write_csv(args.output_dir / "ts_leave_one_out_errors.csv", loo_rows)
    write_csv(args.output_dir / "tls_candidate_clusters.csv", cluster_rows)
    report = {
        "schema_version": "1.0",
        "diagnostic_id": "ROCKFALL-REFERENCE-DIAGNOSTIC-V1",
        "dataset_revision": "42a3947d960c8163157c915dea847cda96904a3d",
        "algorithm_outputs_accessed": False,
        "parent_d2_failure_preserved": True,
        "reference_summary": reference_summary,
        "tls_epoch_summaries": tls_rows,
        "candidate_cluster_count": len(cluster_rows),
        "interpretation": {
            "process_success_is_scientific_gate_pass": False,
            "candidate_clusters_establish_target_identity": False,
            "mapping_requires_separate_frozen_protocol": True,
        },
    }
    (args.output_dir / "reference_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
