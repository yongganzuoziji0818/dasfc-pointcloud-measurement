"""Pilot the cross-fitted structural scale + scan-pair conformal algorithm."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.measurement_q2.pcudm import (
    PCUDMFieldEstimator,
    PairwiseSimultaneousCalibrator,
    StructuralScaleModel,
    SyntheticDomain,
    generate_case,
)
from research.measurement_q2.pcudm.metrics import scan_pair_metrics


DOMAINS = {
    "nominal": SyntheticDomain(
        name="nominal", noise_mm=0.55, heteroscedasticity=0.8, dropout=0.05,
        outlier_fraction=0.01, occlusion_fraction=0.05, pose_translation_mm=7.0,
        pose_rotation_deg=0.8, density_jitter=0.05,
        support_candidate_contamination=0.15, support_candidate_miss=0.08,
    ),
    "noisy_sparse": SyntheticDomain(
        name="noisy_sparse", noise_mm=1.25, heteroscedasticity=1.7, dropout=0.18,
        outlier_fraction=0.04, occlusion_fraction=0.18, pose_translation_mm=12.0,
        pose_rotation_deg=1.6, density_jitter=0.18,
        support_candidate_contamination=0.25, support_candidate_miss=0.15,
    ),
    "unseen_compound": SyntheticDomain(
        name="unseen_compound", noise_mm=2.0, heteroscedasticity=2.4, dropout=0.28,
        outlier_fraction=0.07, occlusion_fraction=0.28, pose_translation_mm=16.0,
        pose_rotation_deg=2.4, density_jitter=0.28,
        support_candidate_contamination=0.35, support_candidate_miss=0.20,
    ),
}

SCALE_METHODS = (
    "homoscedastic",
    "raw_local",
    "learned_structural",
    "adaptive_structural",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-per-known-domain", type=int, default=20)
    parser.add_argument("--validation-per-known-domain", type=int, default=15)
    parser.add_argument("--calibration-per-known-domain", type=int, default=40)
    parser.add_argument("--test-per-domain", type=int, default=15)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def estimate(seed: int, domain: SyntheticDomain) -> dict:
    case = generate_case(seed, domain, points_x=18, points_z=14)
    started = time.perf_counter()
    result = PCUDMFieldEstimator(mode="cascade", icp_iterations=14).fit(
        case.reference, case.target, case.panel_ids, case.support_candidates
    )
    return {
        "case": case,
        "result": result,
        "error": result.normal_displacement - case.normal_displacement_true,
        "duration_seconds": time.perf_counter() - started,
    }


def scale_for(
    method: str,
    estimated: dict,
    model: StructuralScaleModel,
    domain_name: str,
    blend_by_domain: dict[str, float],
) -> np.ndarray:
    case = estimated["case"]
    result = estimated["result"]
    if method == "homoscedastic":
        value = float(np.median(result.scale[case.valid_field_mask]))
        return np.full(result.scale.shape, max(value, 1e-6), dtype=np.float64)
    if method == "raw_local":
        return result.scale
    if method == "learned_structural":
        return model.predict(case.reference, case.panel_ids, result)
    if method == "adaptive_structural":
        constant = np.full(
            result.scale.shape,
            max(float(np.median(result.scale[case.valid_field_mask])), 1e-6),
            dtype=np.float64,
        )
        learned = model.predict(case.reference, case.panel_ids, result)
        blend = blend_by_domain.get(domain_name, 0.0)
        return np.exp((1.0 - blend) * np.log(constant) + blend * np.log(learned))
    raise KeyError(method)


def main() -> int:
    args = parse_args()
    counts = (
        args.tuning_per_known_domain,
        args.validation_per_known_domain,
        args.calibration_per_known_domain,
        args.test_per_domain,
    )
    if min(counts) < 1:
        raise ValueError("all split counts must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    known_domains = ("nominal", "noisy_sparse")

    tuning = []
    for domain_index, domain_name in enumerate(known_domains):
        for offset in range(args.tuning_per_known_domain):
            estimated = estimate(30_000 + domain_index * 1_000 + offset, DOMAINS[domain_name])
            case = estimated["case"]
            tuning.append({
                "case_id": case.case_id,
                "reference": case.reference,
                "panel_ids": case.panel_ids,
                "result": estimated["result"],
                "error": estimated["error"],
                "valid": case.valid_field_mask,
            })

    scale_model = StructuralScaleModel(
        target_quantile=0.80,
        max_locations_per_pair=256,
        random_state=20260719,
    )
    scale_diagnostics = scale_model.fit(tuning)

    blend_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    blend_by_domain = {}
    blend_validation = {}
    for domain_index, domain_name in enumerate(known_domains):
        validation = [
            estimate(35_000 + domain_index * 1_000 + offset, DOMAINS[domain_name])
            for offset in range(args.validation_per_known_domain)
        ]
        candidates = []
        for blend in blend_grid:
            calibration_cases = []
            mean_scales = []
            for estimated in validation:
                case = estimated["case"]
                result = estimated["result"]
                constant = np.full(
                    result.scale.shape,
                    max(float(np.median(result.scale[case.valid_field_mask])), 1e-6),
                    dtype=np.float64,
                )
                learned = scale_model.predict(case.reference, case.panel_ids, result)
                blended = np.exp(
                    (1.0 - blend) * np.log(constant) + blend * np.log(learned)
                )
                calibration_cases.append({
                    "group": domain_name,
                    "error": estimated["error"],
                    "scale": blended,
                    "valid": case.valid_field_mask,
                })
                mean_scales.append(float(np.mean(blended[case.valid_field_mask])))
            state = PairwiseSimultaneousCalibrator(alpha=0.05).fit(calibration_cases)
            quantile = state.group_quantiles[domain_name]
            candidates.append({
                "blend": blend,
                "validation_quantile": quantile,
                "validation_mean_width_mm": float(2.0 * quantile * np.mean(mean_scales)),
                "validation_scan_pairs": len(validation),
            })
        selected = min(candidates, key=lambda item: (item["validation_mean_width_mm"], item["blend"]))
        blend_by_domain[domain_name] = selected["blend"]
        blend_validation[domain_name] = {
            "selected_blend": selected["blend"],
            "candidates": candidates,
        }
    # A group absent from tuning/validation has no basis for heteroscedastic
    # adaptation. The explicit conservative fallback is equivalent to blend=0.
    blend_by_domain["unseen_compound"] = 0.0

    calibration_payload = {method: [] for method in SCALE_METHODS}
    for domain_index, domain_name in enumerate(known_domains):
        for offset in range(args.calibration_per_known_domain):
            estimated = estimate(40_000 + domain_index * 1_000 + offset, DOMAINS[domain_name])
            case = estimated["case"]
            for method in SCALE_METHODS:
                calibration_payload[method].append({
                    "group": domain_name,
                    "error": estimated["error"],
                    "scale": scale_for(
                        method, estimated, scale_model, domain_name, blend_by_domain
                    ),
                    "valid": case.valid_field_mask,
                })

    calibrators = {}
    calibration_states = {}
    for method in SCALE_METHODS:
        calibrator = PairwiseSimultaneousCalibrator(alpha=0.05)
        state = calibrator.fit(calibration_payload[method])
        calibrators[method] = calibrator
        calibration_states[method] = {
            "group_quantiles": state.group_quantiles,
            "group_counts": state.group_counts,
            "group_order_statistics": state.group_order_statistics,
            "pooled_quantile": state.pooled_quantile,
            "pooled_count": state.pooled_count,
            "pooled_order_statistic": state.pooled_order_statistic,
        }

    rows = []
    for domain_index, domain_name in enumerate((*known_domains, "unseen_compound")):
        is_known = domain_name in known_domains
        for offset in range(args.test_per_domain):
            seed = 50_000 + domain_index * 1_000 + offset
            estimated = estimate(seed, DOMAINS[domain_name])
            case = estimated["case"]
            result = estimated["result"]
            for method in SCALE_METHODS:
                scale = scale_for(
                    method, estimated, scale_model, domain_name, blend_by_domain
                )
                interval_calibrator = (
                    calibrators["homoscedastic"]
                    if method == "adaptive_structural" and not is_known
                    else calibrators[method]
                )
                lower, upper = interval_calibrator.interval(
                    result.normal_displacement,
                    scale,
                    domain_name,
                    allow_pooled_fallback=not is_known,
                )
                metrics = scan_pair_metrics(
                    case.normal_displacement_true,
                    result.normal_displacement,
                    lower,
                    upper,
                    valid=case.valid_field_mask,
                )
                rows.append({
                    "case_id": case.case_id,
                    "seed": seed,
                    "domain": domain_name,
                    "domain_status": "known_calibration_group" if is_known else "unseen_empirical",
                    "scale_method": method,
                    "converged": result.converged,
                    "duration_seconds": estimated["duration_seconds"],
                    "mean_scale_mm": float(np.mean(scale[case.valid_field_mask])),
                    **metrics,
                })

    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["scale_method"], row["domain"], row["domain_status"])].append(row)
    summaries = []
    for (method, domain, status), group_rows in sorted(buckets.items()):
        summaries.append({
            "scale_method": method,
            "domain": domain,
            "domain_status": status,
            "scan_pairs": len(group_rows),
            "simultaneous_coverage": float(np.mean([r["simultaneous_covered"] for r in group_rows])),
            "normal_mae_mm": float(np.mean([r["normal_mae_mm"] for r in group_rows])),
            "mean_interval_width_mm": float(np.mean([r["mean_interval_width_mm"] for r in group_rows])),
            "interval_score_mm": float(np.mean([r["interval_score_mm"] for r in group_rows])),
            "mean_scale_mm": float(np.mean([r["mean_scale_mm"] for r in group_rows])),
        })

    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["scale_method"]] = row
    paired_rows = []
    for case_id, method_rows in sorted(by_case.items()):
        learned = method_rows["adaptive_structural"]
        for baseline in ("homoscedastic", "raw_local", "learned_structural"):
            base = method_rows[baseline]
            paired_rows.append({
                "case_id": case_id,
                "domain": learned["domain"],
                "domain_status": learned["domain_status"],
                "baseline": baseline,
                "width_difference_learned_minus_baseline_mm": (
                    learned["mean_interval_width_mm"] - base["mean_interval_width_mm"]
                ),
                "interval_score_difference_learned_minus_baseline_mm": (
                    learned["interval_score_mm"] - base["interval_score_mm"]
                ),
            })

    known_pairs = [row for row in paired_rows if row["domain_status"] == "known_calibration_group"]
    decisions = {}
    for baseline in ("homoscedastic", "raw_local", "learned_structural"):
        selected = [row for row in known_pairs if row["baseline"] == baseline]
        decisions[baseline] = {
            "mean_width_difference_mm": float(
                np.mean([row["width_difference_learned_minus_baseline_mm"] for row in selected])
            ),
            "mean_interval_score_difference_mm": float(
                np.mean([row["interval_score_difference_learned_minus_baseline_mm"] for row in selected])
            ),
        }
    learned_known = [
        row for row in rows
        if row["scale_method"] == "adaptive_structural"
        and row["domain_status"] == "known_calibration_group"
    ]
    decision = {
        "comparisons": decisions,
        "adaptive_known_domain_simultaneous_coverage": float(
            np.mean([row["simultaneous_covered"] for row in learned_known])
        ),
        "candidate_signal_present": bool(
            decisions["homoscedastic"]["mean_interval_score_difference_mm"] < 0
            and decisions["raw_local"]["mean_interval_score_difference_mm"] < 0
            and float(np.mean([row["simultaneous_covered"] for row in learned_known])) >= 0.90
        ),
        "note": "Development pilot only; tuning, calibration, and test seeds are disjoint.",
    }

    report = {
        "schema_version": "1.0",
        "status": "DEVELOPMENT_SCALE_CALIBRATION_PILOT",
        "independent_unit": "complete synthetic scan pair",
        "split_seed_ranges": {
            "tuning": "30000-series",
            "validation": "35000-series",
            "calibration": "40000-series",
            "test": "50000-series",
        },
        "counts": {
            "tuning_per_known_domain": args.tuning_per_known_domain,
            "validation_per_known_domain": args.validation_per_known_domain,
            "calibration_per_known_domain": args.calibration_per_known_domain,
            "test_per_domain": args.test_per_domain,
        },
        "scale_model": {
            "scan_pairs": scale_diagnostics.scan_pairs,
            "training_locations": scale_diagnostics.training_locations,
            "target_quantile": scale_diagnostics.target_quantile,
            "feature_names": scale_diagnostics.feature_names,
        },
        "adaptive_blend": {
            "selection_split": "validation only",
            "blend_by_domain": blend_by_domain,
            "validation_details": blend_validation,
            "unseen_group_policy": "blend=0 conservative homoscedastic fallback",
        },
        "calibration": calibration_states,
        "summaries": summaries,
        "paired_decision": decision,
        "rows": rows,
        "paired_rows": paired_rows,
    }
    (output_dir / "scale_pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "scale_pilot_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summaries": summaries, "paired_decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
