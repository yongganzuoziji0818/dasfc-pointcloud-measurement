"""Run the frozen P4 post-confirmatory submission-strengthening analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from research.measurement_q2.pcudm import StructuralScaleModel
from research.measurement_q2.pcudm.metrics import scan_pair_metrics
from research.measurement_q2.scripts.run_das_fc_formal import (
    base_scales,
    case_seed,
    das_scale,
    domain_from_config,
    estimate,
)
from research.measurement_q2.scripts.run_rockfall_physical_validation import (
    MOVING_TARGETS,
    PLY_NAMES,
    canonical_frame,
    load_markers,
    prepare_epoch,
    read_ply_xyz,
    run_frontend,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURE_NAMES = (
    "scale_median_over_diagonal",
    "scale_q95_over_diagonal",
    "match_median_over_diagonal",
    "match_q95_over_diagonal",
    "support_probability_mean",
    "valid_fraction",
    "abs_normal_displacement_q95_over_diagonal",
)
EVENT_EPOCHS = {
    "E0->E1": ("E0", "E1"),
    "E1->E2": ("E1", "E2"),
    "E2->E3": ("E2", "E3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input(project_root: Path, repo_root: Path, relative: str) -> Path:
    base = repo_root if relative.startswith("research/") else project_root
    return base / relative


def verify_inputs(config: dict, project_root: Path, repo_root: Path) -> dict[str, str]:
    observed = {}
    for relative, expected in config["input_sha256"].items():
        path = resolve_input(project_root, repo_root, relative)
        value = sha256(path)
        if value != expected:
            raise ValueError(f"input SHA-256 mismatch: {relative}: {value} != {expected}")
        observed[relative] = value
    return observed


def robust_diagonal(reference: np.ndarray, target: np.ndarray) -> float:
    combined = np.vstack((reference, target))
    bounds = np.quantile(combined, (0.10, 0.90), axis=0)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 1e-12:
        raise ValueError("invalid robust pair diagonal")
    return diagonal


def normalized_features(result: object, mask: np.ndarray, diagonal: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != result.scale.shape or not mask.any():
        raise ValueError("invalid declared feature mask")
    return np.asarray(
        [
            np.median(result.scale[mask]) / diagonal,
            np.quantile(result.scale[mask], 0.95) / diagonal,
            np.median(result.match_distance[mask]) / diagonal,
            np.quantile(result.match_distance[mask], 0.95) / diagonal,
            np.mean(result.support_probability[mask]),
            np.mean(result.valid[mask]),
            np.quantile(np.abs(result.normal_displacement[mask]), 0.95) / diagonal,
        ],
        dtype=np.float64,
    )


def train_frozen_scale_and_features(formal_config: dict) -> tuple[StructuralScaleModel, list[dict]]:
    known = {
        name: domain_from_config(name, spec)
        for name, spec in formal_config["known_domains"].items()
    }
    grid = formal_config["grid"]
    tuning_count = int(formal_config["counts_per_known_domain"]["tuning"])
    training_cases = []
    feature_rows = []
    for domain_index, (domain_name, (domain, geometry)) in enumerate(known.items()):
        for offset in range(tuning_count):
            seed = case_seed(formal_config, "tuning", domain_name, domain_index, offset)
            estimated = estimate(seed, domain, geometry, grid, formal_config.get("estimator"))
            case = estimated["case"]
            result = estimated["result"]
            training_cases.append(
                {
                    "case_id": case.case_id,
                    "group": domain_name,
                    "reference": case.reference,
                    "panel_ids": case.panel_ids,
                    "result": result,
                    "error": estimated["error"],
                    "valid": case.valid_field_mask,
                }
            )
            diagonal = robust_diagonal(case.reference, case.target)
            values = normalized_features(result, case.valid_field_mask, diagonal)
            feature_rows.append(
                {
                    "case_id": case.case_id,
                    "domain": domain_name,
                    "seed": seed,
                    "robust_diagonal": diagonal,
                    **{name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)},
                }
            )
    model = StructuralScaleModel(**formal_config["scale_model"])
    model.fit(training_cases)
    return model, feature_rows


def load_rockfall_features(
    project_root: Path,
    repo_root: Path,
    config: dict,
) -> tuple[list[dict], np.ndarray]:
    rockfall_config = json.loads((repo_root / config["rockfall_config"]).read_text(encoding="utf-8"))
    dataset_root = project_root / config["rockfall_dataset_root"]
    mapping_dir = project_root / config["rockfall_mapping_dir"]
    markers = load_markers(mapping_dir / "frame_residuals.csv")
    origin, basis = canonical_frame(markers["E0"])
    marker_local = {
        target: (markers["E0"][target] - origin) @ basis * 1000.0
        for target in MOVING_TARGETS
    }
    moving_xz = np.asarray([[marker_local[t][0], marker_local[t][2]] for t in MOVING_TARGETS])
    moving_bounds = np.stack([moving_xz.min(axis=0), moving_xz.max(axis=0)])
    prep = rockfall_config["preparation"]
    required_epochs = sorted(set(epoch for pair in rockfall_config["pairs"] for epoch in pair))
    prepared = {}
    for epoch in required_epochs:
        path = dataset_root / "02_ExportedData" / "02_TLS" / "raw_pcd" / PLY_NAMES[epoch]
        prepared[epoch] = prepare_epoch(read_ply_xyz(path), origin, basis, moving_bounds, prep)

    rows = []
    moving_margin = float(prep["moving_margin_mm"])
    for source_epoch, target_epoch in rockfall_config["pairs"]:
        reference = prepared[source_epoch]
        target = prepared[target_epoch]
        moving = (
            (reference[:, 0] >= moving_bounds[0, 0] - moving_margin)
            & (reference[:, 0] <= moving_bounds[1, 0] + moving_margin)
            & (reference[:, 2] >= moving_bounds[0, 1] - moving_margin)
            & (reference[:, 2] <= moving_bounds[1, 1] + moving_margin)
        )
        support = ~moving
        panel_ids = moving.astype(np.int32)
        result, _ = run_frontend(
            reference,
            target,
            panel_ids,
            support,
            "cascade_strong",
            rockfall_config["estimator"],
        )
        if not result.converged:
            raise RuntimeError(f"Rockfall diagnostic estimator failed: {source_epoch}->{target_epoch}")
        diagonal = robust_diagonal(reference, target)
        values = normalized_features(result, moving, diagonal)
        rows.append(
            {
                "event": f"{source_epoch}->{target_epoch}",
                "robust_diagonal": diagonal,
                **{name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)},
            }
        )
    return rows, basis


def audit_domain_alignment(synthetic_rows: list[dict], rockfall_rows: list[dict], spec: dict) -> dict:
    matrix = np.asarray([[row[name] for name in FEATURE_NAMES] for row in synthetic_rows])
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    sample_sd = matrix.std(axis=0, ddof=1)
    scale = float(spec["mad_multiplier"]) * mad
    fallback = np.maximum(
        float(spec["std_fallback_multiplier"]) * sample_sd,
        float(spec["scale_floor"]),
    )
    scale = np.where(scale > float(spec["scale_floor"]), scale, fallback)
    lower = np.quantile(matrix, float(spec["envelope_quantiles"][0]), axis=0)
    upper = np.quantile(matrix, float(spec["envelope_quantiles"][1]), axis=0)
    synthetic_z = (matrix - median) / scale
    synthetic_rms = np.sqrt(np.mean(synthetic_z**2, axis=1))
    audited = []
    for row in rockfall_rows:
        values = np.asarray([row[name] for name in FEATURE_NAMES])
        z = (values - median) / scale
        distances = np.sqrt(np.mean((synthetic_z - z) ** 2, axis=1))
        percentiles = [float(np.mean(matrix[:, index] <= values[index])) for index in range(len(FEATURE_NAMES))]
        audited.append(
            {
                **row,
                "robust_rms_z": float(np.sqrt(np.mean(z**2))),
                "nearest_synthetic_robust_distance": float(np.min(distances)),
                "outside_1_99_feature_count": int(np.sum((values < lower) | (values > upper))),
                "feature_empirical_percentiles": dict(zip(FEATURE_NAMES, percentiles, strict=True)),
            }
        )
    return {
        "feature_names": list(FEATURE_NAMES),
        "synthetic_pairs": len(synthetic_rows),
        "rockfall_events": len(rockfall_rows),
        "synthetic_center": dict(zip(FEATURE_NAMES, median.tolist(), strict=True)),
        "synthetic_robust_scale": dict(zip(FEATURE_NAMES, scale.tolist(), strict=True)),
        "synthetic_envelope_1_99": {
            name: [float(lo), float(hi)]
            for name, lo, hi in zip(FEATURE_NAMES, lower, upper, strict=True)
        },
        "synthetic_robust_rms_z": synthetic_rms.tolist(),
        "synthetic_robust_rms_z_q95": float(np.quantile(synthetic_rms, 0.95)),
        "synthetic_robust_rms_z_q99": float(np.quantile(synthetic_rms, 0.99)),
        "events": audited,
        "claim_boundary": "truth-free descriptive applicability audit; no restored coverage guarantee",
    }


def read_mapping_residuals(path: Path) -> dict[tuple[str, str], np.ndarray]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["epoch"], row["target"])] = np.asarray(
                [row["residual_x_mm"], row["residual_y_mm"], row["residual_z_mm"]],
                dtype=np.float64,
            )
    return rows


def reference_sensitivity(
    target_results: Path,
    mapping_json: Path,
    residual_csv: Path,
    basis: np.ndarray,
) -> dict:
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    residuals = read_mapping_residuals(residual_csv)
    target_rows = list(csv.DictReader(target_results.open("r", encoding="utf-8", newline="")))
    rows = []
    for row in target_rows:
        event = row["event"]
        source, target_epoch = EVENT_EPOCHS[event]
        target_name = row["target"]
        source_normal = float(residuals[(source, target_name)] @ basis[:, 1])
        target_normal = float(residuals[(target_epoch, target_name)] @ basis[:, 1])
        projected_tau = abs(source_normal) + abs(target_normal)
        source_rms = float(mapping["epoch_residual_summary"][source]["rms_mm"])
        target_rms = float(mapping["epoch_residual_summary"][target_epoch]["rms_mm"])
        rss_tau = math.sqrt(source_rms**2 + target_rms**2)
        reference = float(row["reference_dy_normal_mm"])
        prediction = float(row["predicted_dy_normal_mm"])
        lower = float(row["interval_lower_normal_mm"])
        upper = float(row["interval_upper_normal_mm"])
        output = {
            "event": event,
            "frontend": row["frontend"],
            "target": target_name,
            "reference_normal_mm": reference,
            "predicted_normal_mm": prediction,
            "interval_lower_mm": lower,
            "interval_upper_mm": upper,
            "central_reference_covered": lower <= reference <= upper,
        }
        for mode, tau in (("target_projected_sum", projected_tau), ("event_rms_rss", rss_tau)):
            output[f"{mode}_tolerance_mm"] = float(tau)
            output[f"{mode}_intersects"] = bool(lower <= reference + tau and upper >= reference - tau)
            output[f"{mode}_fully_contains"] = bool(lower <= reference - tau and upper >= reference + tau)
            output[f"{mode}_abs_error_lower_mm"] = float(max(abs(prediction - reference) - tau, 0.0))
            output[f"{mode}_abs_error_upper_mm"] = float(abs(prediction - reference) + tau)
        rows.append(output)

    event_summaries = []
    for (event, frontend), group in sorted(_group(rows, "event", "frontend").items()):
        summary = {
            "event": event,
            "frontend": frontend,
            "targets": len(group),
            "central_reference_all_four": all(row["central_reference_covered"] for row in group),
        }
        for mode in ("target_projected_sum", "event_rms_rss"):
            summary[f"{mode}_all_four_intersect"] = all(row[f"{mode}_intersects"] for row in group)
            summary[f"{mode}_all_four_fully_contained"] = all(row[f"{mode}_fully_contains"] for row in group)
        event_summaries.append(summary)

    frontend_summaries = []
    for frontend, group in sorted(_group(event_summaries, "frontend").items()):
        summary = {
            "frontend": frontend,
            "events": len(group),
            "central_reference_events": sum(row["central_reference_all_four"] for row in group),
        }
        for mode in ("target_projected_sum", "event_rms_rss"):
            summary[f"{mode}_intersect_events"] = sum(row[f"{mode}_all_four_intersect"] for row in group)
            summary[f"{mode}_fully_contained_events"] = sum(
                row[f"{mode}_all_four_fully_contained"] for row in group
            )
        frontend_summaries.append(summary)
    return {
        "target_rows": rows,
        "event_summaries": event_summaries,
        "frontend_summaries": frontend_summaries,
        "claim_boundary": (
            "post-confirmatory frame-mapping tolerance sensitivity; not a GUM budget, "
            "traceability statement, or population-coverage estimate"
        ),
    }


def _group(rows: list[dict], *keys: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups[key[0] if len(key) == 1 else key].append(row)
    return groups


def field_example(
    formal_config: dict,
    formal_report: dict,
    scale_model: StructuralScaleModel,
    spec: dict,
) -> tuple[dict, list[dict]]:
    by_case = defaultdict(dict)
    for row in formal_report["rows"]:
        if row["domain_status"] == "known_calibration_group":
            by_case[row["case_id"]][row["method"]] = row
    eligible = []
    for case_id, methods in by_case.items():
        pointwise = methods.get("das_pointwise")
        das = methods.get("das_grouped")
        if not pointwise or not das:
            continue
        if (
            float(pointwise["point_coverage"]) >= float(spec["minimum_pointwise_point_coverage"])
            and not bool(pointwise["simultaneous_covered"])
            and bool(das["simultaneous_covered"])
        ):
            eligible.append(case_id)
    if not eligible:
        raise ValueError("no field example satisfies the frozen selection rule")
    case_id = sorted(eligible)[0]
    stored_pointwise = by_case[case_id]["das_pointwise"]
    stored_das = by_case[case_id]["das_grouped"]
    domain_name = stored_das["domain"]
    domain_index = list(formal_config["known_domains"]).index(domain_name)
    domain, geometry = domain_from_config(domain_name, formal_config["known_domains"][domain_name])
    seed = int(stored_das["seed"])
    expected_seed = case_seed(
        formal_config,
        "test_known",
        domain_name,
        domain_index,
        int(seed - formal_config["seed_bases"]["test_known"] - domain_index * 10_000),
    )
    if seed != expected_seed:
        raise ValueError("selected-case seed does not match frozen seed rule")
    estimated = estimate(seed, domain, geometry, formal_config["grid"], formal_config.get("estimator"))
    case = estimated["case"]
    result = estimated["result"]
    scales = base_scales(estimated, scale_model)
    blend = float(formal_report["blend_by_domain"][domain_name])
    adaptive = das_scale(scales, blend)
    q_point = float(formal_report["pointwise_calibration"]["quantiles"][domain_name])
    q_das = float(formal_report["calibration"]["das"]["group_quantiles"][domain_name])
    valid = case.valid_field_mask
    truth = case.normal_displacement_true
    prediction = result.normal_displacement
    radii = {"pointwise": q_point * adaptive, "das": q_das * adaptive}
    reconstructed = {}
    for name, radius in radii.items():
        reconstructed[name] = scan_pair_metrics(
            truth,
            prediction,
            prediction - radius,
            prediction + radius,
            valid=valid,
            alpha=float(formal_config["alpha"]),
        )
    tolerance = float(spec["reconstruction_absolute_tolerance"])
    comparisons = (
        (reconstructed["pointwise"], stored_pointwise),
        (reconstructed["das"], stored_das),
    )
    for observed, stored in comparisons:
        for key in ("point_coverage", "mean_interval_width_mm"):
            if not np.isclose(float(observed[key]), float(stored[key]), rtol=0, atol=tolerance):
                raise ValueError(f"field reconstruction mismatch for {key}")
        if bool(observed["simultaneous_covered"]) != bool(stored["simultaneous_covered"]):
            raise ValueError("field reconstruction simultaneous-coverage mismatch")

    error = prediction - truth
    rows = []
    for index in np.flatnonzero(valid):
        rows.append(
            {
                "case_id": case_id,
                "domain": domain_name,
                "seed": seed,
                "location_index": int(index),
                "x_mm": float(case.reference[index, 0]),
                "z_mm": float(case.reference[index, 2]),
                "error_mm": float(error[index]),
                "pointwise_radius_mm": float(radii["pointwise"][index]),
                "das_radius_mm": float(radii["das"][index]),
                "pointwise_standardized_abs_error": float(abs(error[index]) / radii["pointwise"][index]),
                "das_standardized_abs_error": float(abs(error[index]) / radii["das"][index]),
                "pointwise_covered": bool(abs(error[index]) <= radii["pointwise"][index]),
                "das_covered": bool(abs(error[index]) <= radii["das"][index]),
            }
        )
    summary = {
        "selection_rule": (
            "lexicographically first known-domain case with pointwise point coverage >= 0.90, "
            "pointwise simultaneous failure, and DAS-FC simultaneous success"
        ),
        "eligible_cases": len(eligible),
        "selected_case_id": case_id,
        "domain": domain_name,
        "seed": seed,
        "field_locations": len(rows),
        "pointwise": reconstructed["pointwise"],
        "das": reconstructed["das"],
        "reconstruction_gate": True,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    output_dir: Path,
    domain_audit: dict,
    sensitivity: dict,
    field_rows: list[dict],
    figure_spec: dict,
) -> None:
    palette = figure_spec["colorblind_palette"]
    plt.rcParams.update(
        {
            "font.family": figure_spec["font_family"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), constrained_layout=True)

    ax = axes[0, 0]
    synthetic = np.asarray(domain_audit["synthetic_robust_rms_z"])
    ax.boxplot(
        [synthetic],
        positions=[0],
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#BDBDBD", "edgecolor": "#444444"},
        medianprops={"color": "#000000"},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
        flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "#777777"},
    )
    for index, event in enumerate(domain_audit["events"]):
        ax.scatter(1, event["robust_rms_z"], s=35, color=palette[index], marker=("o", "s", "^")[index], label=event["event"], zorder=3)
    ax.axhline(domain_audit["synthetic_robust_rms_z_q99"], color="#666666", linestyle="--", linewidth=1, label="synthetic 99th percentile")
    ax.set_xticks([0, 1], ["Synthetic tuning\n(n=240)", "Rockfall\n(n=3)"])
    ax.set_ylabel("Robust RMS diagnostic distance")
    ax.set_title("Observable domain diagnostic")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    frontends = [row["frontend"] for row in sensitivity["frontend_summaries"]]
    x = np.arange(len(frontends))
    width = 0.25
    values = [
        [row["central_reference_events"] for row in sensitivity["frontend_summaries"]],
        [row["target_projected_sum_fully_contained_events"] for row in sensitivity["frontend_summaries"]],
        [row["event_rms_rss_fully_contained_events"] for row in sensitivity["frontend_summaries"]],
    ]
    labels = ["central value", "projected tolerance", "RMS-RSS tolerance"]
    colors = [palette[0], palette[2], palette[1]]
    hatches = ["", "//", "xx"]
    for index, (vals, label, color, hatch) in enumerate(zip(values, labels, colors, hatches, strict=True)):
        ax.bar(x + (index - 1) * width, vals, width, label=label, color=color, edgecolor="black", linewidth=0.5, hatch=hatch)
    frontend_labels = {
        "cascade_strong": "Cascade\nStrong",
        "multiscale_trimmed_ptp": "Trimmed\nPtP",
        "robust_ptpl": "Robust\nPtPL",
    }
    ax.set_xticks(x, [frontend_labels[name] for name in frontends])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(0, 3.35)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Events containing all 4 targets")
    ax.set_title("Reference-tolerance sensitivity")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2, handlelength=1.5)

    common_field_vmax = max(
        1.5,
        float(
            np.quantile(
                [
                    row[key]
                    for row in field_rows
                    for key in ("pointwise_standardized_abs_error", "das_standardized_abs_error")
                ],
                0.99,
            )
        ),
    )
    for panel, key, title in (
        (axes[1, 0], "pointwise_standardized_abs_error", "Pointwise interval"),
        (axes[1, 1], "das_standardized_abs_error", "DAS-FC simultaneous interval"),
    ):
        values = np.asarray([row[key] for row in field_rows])
        scatter = panel.scatter(
            [row["x_mm"] for row in field_rows],
            [row["z_mm"] for row in field_rows],
            c=values,
            cmap="cividis",
            vmin=0,
            vmax=common_field_vmax,
            s=18,
            marker="s",
            linewidths=0,
        )
        failed = values > 1.0
        panel.scatter(
            np.asarray([row["x_mm"] for row in field_rows])[failed],
            np.asarray([row["z_mm"] for row in field_rows])[failed],
            facecolors="none",
            edgecolors=palette[3],
            s=32,
            linewidths=0.8,
        )
        panel.set_xlabel("Field x (mm)")
        panel.set_ylabel("Field z (mm)")
        panel.set_title(title)
        panel.set_aspect("equal", adjustable="box")
        colorbar = fig.colorbar(scatter, ax=panel, fraction=0.046, pad=0.03)
        colorbar.set_label("|error| / interval radius")
        colorbar.ax.axhline(1.0, color=palette[3], linewidth=0.8)

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = output_dir / "figure8_p4_submission_strengthening"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=int(figure_spec["dpi"]), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observed_inputs = verify_inputs(config, args.project_root, args.repo_root)
    formal_config = json.loads((args.repo_root / config["formal_config"]).read_text(encoding="utf-8"))
    formal_report = json.loads((args.repo_root / config["formal_report"]).read_text(encoding="utf-8"))

    scale_model, synthetic_rows = train_frozen_scale_and_features(formal_config)
    rockfall_rows, basis = load_rockfall_features(args.project_root, args.repo_root, config)
    physical_report = json.loads(
        (args.project_root / config["rockfall_physical_report"]).read_text(encoding="utf-8")
    )
    if not np.allclose(
        basis,
        np.asarray(physical_report["frame_basis_tls_to_canonical"], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("reconstructed Rockfall canonical basis differs from frozen D5 report")
    domain_audit = audit_domain_alignment(synthetic_rows, rockfall_rows, config["domain_audit"])
    sensitivity = reference_sensitivity(
        args.project_root / config["rockfall_target_results"],
        args.project_root / config["rockfall_mapping_dir"] / "frame_mapping.json",
        args.project_root / config["rockfall_mapping_dir"] / "frame_residuals.csv",
        basis,
    )
    example_summary, example_rows = field_example(
        formal_config,
        formal_report,
        scale_model,
        config["field_example"],
    )

    write_csv(args.output_dir / "synthetic_tuning_diagnostics.csv", synthetic_rows)
    write_csv(args.output_dir / "rockfall_diagnostics.csv", rockfall_rows)
    write_csv(args.output_dir / "rockfall_reference_sensitivity_targets.csv", sensitivity["target_rows"])
    write_csv(args.output_dir / "field_failure_example.csv", example_rows)
    make_figure(args.output_dir, domain_audit, sensitivity, example_rows, config["figure"])

    report = {
        "schema_version": "1.0",
        "protocol_id": config["protocol_id"],
        "status": "P4_SUBMISSION_STRENGTHENING_COMPLETE",
        "input_sha256": observed_inputs,
        "domain_alignment": domain_audit,
        "reference_sensitivity": sensitivity,
        "field_example": example_summary,
        "claim_boundary": (
            "post-confirmatory sensitivity, applicability diagnostics, and deterministic illustration; "
            "not new dense physical truth or population coverage"
        ),
    }
    (args.output_dir / "p4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "synthetic_pairs": domain_audit["synthetic_pairs"],
                "rockfall_events": domain_audit["rockfall_events"],
                "selected_case": example_summary["selected_case_id"],
                "reference_sensitivity": sensitivity["frontend_summaries"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
