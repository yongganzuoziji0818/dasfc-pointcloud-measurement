"""Map the ETH Rockfall TS frame to TLS using marker data only."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPOCHS = ("E0", "E1", "E2", "E3")
TARGETS = ("T0", "T1", "T2", "T3", "T4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_ts(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, dict[str, np.ndarray]] = {epoch: {} for epoch in EPOCHS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 4 or "_" not in row[0]:
                continue
            epoch, target = row[0].strip().split("_", 1)
            if epoch in values and target in TARGETS:
                values[epoch][target] = np.asarray(row[1:4], dtype=np.float64)
    missing = [f"{e}_{t}" for e in EPOCHS for t in TARGETS if t not in values[e]]
    if missing:
        raise ValueError(f"Missing TS values: {missing}")
    return {epoch: np.stack([values[epoch][target] for target in TARGETS]) for epoch in EPOCHS}


def load_candidates(path: Path, channel: str, min_points: int) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {epoch: [] for epoch in EPOCHS}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["channel"] != channel or int(row["point_count"]) < min_points:
                continue
            epoch = row["epoch"]
            rows[epoch].append(
                np.asarray(
                    [row["center_x_m"], row["center_y_m"], row["center_z_m"]],
                    dtype=np.float64,
                )
            )
    counts = {epoch: len(rows[epoch]) for epoch in EPOCHS}
    if any(count != 5 for count in counts.values()):
        raise ValueError(f"Expected exactly five candidate clusters per epoch: {counts}")
    return {epoch: np.stack(rows[epoch]) for epoch in EPOCHS}


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


def residuals(source: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    predicted = (rotation @ source.T).T + translation
    return np.linalg.norm(predicted - target, axis=1) * 1000.0


def map_e0(ts_e0: np.ndarray, candidates: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    ranked = []
    for order in permutations(range(5)):
        ordered = candidates[list(order)]
        rotation, translation = kabsch(ts_e0, ordered)
        errors = residuals(ts_e0, ordered, rotation, translation)
        ranked.append((float(np.sqrt(np.mean(errors**2))), float(errors.max()), order, rotation, translation, errors))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    best, second = ranked[0], ranked[1]
    summary = {
        "best_rms_mm": best[0],
        "best_max_residual_mm": best[1],
        "second_best_rms_mm": second[0],
        "second_to_best_rms_ratio": second[0] / best[0] if best[0] > 0 else float("inf"),
        "candidate_index_by_target": dict(zip(TARGETS, best[2])),
        "residual_mm_by_target": dict(zip(TARGETS, best[5].tolist())),
    }
    ordered = candidates[list(best[2])]
    return summary, best[3], best[4], ordered


def track_tls(candidates: dict[str, np.ndarray], labelled_e0: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict]]:
    labelled = {"E0": labelled_e0}
    rows: list[dict] = []
    previous = labelled_e0
    for epoch in EPOCHS[1:]:
        current = candidates[epoch]
        costs = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
        source_indices, candidate_indices = linear_sum_assignment(costs)
        if not np.array_equal(source_indices, np.arange(5)):
            raise RuntimeError("Unexpected assignment order")
        ordered = current[candidate_indices]
        for index, target in enumerate(TARGETS):
            rows.append(
                {
                    "epoch": epoch,
                    "target": target,
                    "candidate_index": int(candidate_indices[index]),
                    "tracking_step_mm": float(costs[index, candidate_indices[index]] * 1000.0),
                }
            )
        labelled[epoch] = ordered
        previous = ordered
    return labelled, rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    limits = config["D3_FRAME"]
    ts = load_ts(args.dataset_root / "02_ExportedData" / "01_TS" / "rockfall_sim.txt")
    candidates = load_candidates(
        args.diagnostic_dir / "tls_candidate_clusters.csv",
        limits["candidate_channel"],
        int(limits["candidate_min_points"]),
    )
    e0_summary, rotation, translation, labelled_e0 = map_e0(ts["E0"], candidates["E0"])
    labelled, tracking_rows = track_tls(candidates, labelled_e0)

    residual_rows: list[dict] = []
    epoch_summary: dict[str, dict] = {}
    for epoch in EPOCHS:
        predicted = (rotation @ ts[epoch].T).T + translation
        vectors_mm = (predicted - labelled[epoch]) * 1000.0
        norms_mm = np.linalg.norm(vectors_mm, axis=1)
        epoch_summary[epoch] = {
            "rms_mm": float(np.sqrt(np.mean(norms_mm**2))),
            "max_mm": float(norms_mm.max()),
            "residual_mm_by_target": dict(zip(TARGETS, norms_mm.tolist())),
        }
        for index, target in enumerate(TARGETS):
            residual_rows.append(
                {
                    "epoch": epoch,
                    "target": target,
                    "ts_predicted_tls_x_m": float(predicted[index, 0]),
                    "ts_predicted_tls_y_m": float(predicted[index, 1]),
                    "ts_predicted_tls_z_m": float(predicted[index, 2]),
                    "tls_marker_x_m": float(labelled[epoch][index, 0]),
                    "tls_marker_y_m": float(labelled[epoch][index, 1]),
                    "tls_marker_z_m": float(labelled[epoch][index, 2]),
                    "residual_x_mm": float(vectors_mm[index, 0]),
                    "residual_y_mm": float(vectors_mm[index, 1]),
                    "residual_z_mm": float(vectors_mm[index, 2]),
                    "residual_norm_mm": float(norms_mm[index]),
                }
            )

    displacement_rows: list[dict] = []
    for source_epoch, target_epoch in zip(EPOCHS[:-1], EPOCHS[1:]):
        ts_vectors_tls_mm = (rotation @ (ts[target_epoch] - ts[source_epoch]).T).T * 1000.0
        marker_vectors_mm = (labelled[target_epoch] - labelled[source_epoch]) * 1000.0
        for index, target in enumerate(TARGETS):
            displacement_rows.append(
                {
                    "event": f"{source_epoch}->{target_epoch}",
                    "target": target,
                    "ts_dx_tls_mm": float(ts_vectors_tls_mm[index, 0]),
                    "ts_dy_tls_mm": float(ts_vectors_tls_mm[index, 1]),
                    "ts_dz_tls_mm": float(ts_vectors_tls_mm[index, 2]),
                    "ts_norm_mm": float(np.linalg.norm(ts_vectors_tls_mm[index])),
                    "marker_dx_mm": float(marker_vectors_mm[index, 0]),
                    "marker_dy_mm": float(marker_vectors_mm[index, 1]),
                    "marker_dz_mm": float(marker_vectors_mm[index, 2]),
                    "marker_norm_mm": float(np.linalg.norm(marker_vectors_mm[index])),
                    "vector_difference_mm": float(np.linalg.norm(ts_vectors_tls_mm[index] - marker_vectors_mm[index])),
                }
            )

    t0_stack = np.stack([labelled[epoch][0] for epoch in EPOCHS])
    t0_drift_mm = float(max(np.linalg.norm(a - b) for a, b in permutations(t0_stack, 2)) * 1000.0)
    max_tracking_step_mm = max(row["tracking_step_mm"] for row in tracking_rows)
    e0_pass = (
        e0_summary["best_rms_mm"] <= limits["e0_fit_rms_max_mm"]
        and e0_summary["best_max_residual_mm"] <= limits["e0_fit_max_residual_mm"]
        and e0_summary["second_to_best_rms_ratio"] >= limits["e0_second_best_rms_ratio_min"]
    )
    holdout_pass = all(
        epoch_summary[epoch]["rms_mm"] <= limits["holdout_epoch_rms_max_mm"]
        and epoch_summary[epoch]["max_mm"] <= limits["holdout_epoch_max_residual_mm"]
        for epoch in EPOCHS[1:]
    )
    tracking_pass = max_tracking_step_mm <= limits["tls_tracking_step_max_mm"]
    t0_pass = t0_drift_mm <= limits["t0_tls_max_drift_mm"]
    d3_pass = bool(e0_pass and holdout_pass and tracking_pass and t0_pass)

    with (args.output_dir / "frame_residuals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(residual_rows[0]))
        writer.writeheader()
        writer.writerows(residual_rows)
    with (args.output_dir / "tls_tracking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracking_rows[0]))
        writer.writeheader()
        writer.writerows(tracking_rows)
    with (args.output_dir / "mapped_reference_vectors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(displacement_rows[0]))
        writer.writeheader()
        writer.writerows(displacement_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for target_index, target in enumerate(TARGETS):
        trajectory = np.stack([labelled[epoch][target_index] for epoch in EPOCHS])
        axes[0].plot(trajectory[:, 0], trajectory[:, 2], "o-", label=target)
    axes[0].set_xlabel("TLS x (m)")
    axes[0].set_ylabel("TLS z (m)")
    axes[0].set_title("TLS-only marker tracking")
    axes[0].axis("equal")
    axes[0].legend(ncol=2)
    holdout_values = [[epoch_summary[e]["residual_mm_by_target"][t] for t in TARGETS] for e in EPOCHS]
    image = axes[1].imshow(holdout_values, aspect="auto", cmap="viridis")
    axes[1].set_xticks(range(5), TARGETS)
    axes[1].set_yticks(range(4), EPOCHS)
    axes[1].set_title("Fixed-frame mapping residual (mm)")
    fig.colorbar(image, ax=axes[1], label="mm")
    fig.savefig(args.output_dir / "frame_mapping_validation.png", dpi=220)
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "mapping_id": "ROCKFALL-FRAME-MAPPING-V1",
        "algorithm_outputs_accessed": False,
        "future_ts_used_for_tls_tracking": False,
        "candidate_counts": {epoch: int(len(candidates[epoch])) for epoch in EPOCHS},
        "e0_mapping": e0_summary,
        "rotation_ts_to_tls": rotation.tolist(),
        "translation_ts_to_tls_m": translation.tolist(),
        "epoch_residual_summary": epoch_summary,
        "max_tls_tracking_step_mm": float(max_tracking_step_mm),
        "t0_tls_max_drift_mm": t0_drift_mm,
        "gates": {
            "E0_MAPPING": bool(e0_pass),
            "TLS_ONLY_TRACKING": bool(tracking_pass),
            "HOLDOUT_EPOCHS": bool(holdout_pass),
            "TLS_T0_STABILITY": bool(t0_pass),
            "D3_FRAME_V1": d3_pass,
        },
    }
    (args.output_dir / "frame_mapping.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0 if d3_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

