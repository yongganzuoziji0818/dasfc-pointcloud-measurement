"""Run the pre-freeze PCU-DM mechanism pilot on independent synthetic scan pairs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from research.measurement_q2.pcudm import (
    PCUDMFieldEstimator,
    PairwiseSimultaneousCalibrator,
    SyntheticDomain,
    generate_case,
)
from research.measurement_q2.pcudm.metrics import aurc, scan_pair_metrics


DOMAINS = {
    "nominal": SyntheticDomain(
        name="nominal", noise_mm=0.55, heteroscedasticity=0.8, dropout=0.05,
        outlier_fraction=0.01, occlusion_fraction=0.05, pose_translation_mm=7.0,
        pose_rotation_deg=0.8, density_jitter=0.05,
    ),
    "noisy_sparse": SyntheticDomain(
        name="noisy_sparse", noise_mm=1.25, heteroscedasticity=1.7, dropout=0.18,
        outlier_fraction=0.04, occlusion_fraction=0.18, pose_translation_mm=12.0,
        pose_rotation_deg=1.6, density_jitter=0.18,
    ),
    "unseen_compound": SyntheticDomain(
        name="unseen_compound", noise_mm=2.0, heteroscedasticity=2.4, dropout=0.28,
        outlier_fraction=0.07, occlusion_fraction=0.28, pose_translation_mm=16.0,
        pose_rotation_deg=2.4, density_jitter=0.28,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-per-known-domain", type=int, default=20)
    parser.add_argument("--test-per-domain", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def estimator(method: str) -> PCUDMFieldEstimator:
    if method == "cascade_strong":
        return PCUDMFieldEstimator(mode="cascade", icp_iterations=14)
    if method == "pcudm_joint":
        return PCUDMFieldEstimator(mode="joint", outer_iterations=5, icp_iterations=10)
    raise KeyError(method)


def estimate_case(method: str, case) -> dict:
    started = time.perf_counter()
    result = estimator(method).fit(
        case.reference, case.target, case.panel_ids, case.support_candidates
    )
    duration = time.perf_counter() - started
    error = result.normal_displacement - case.normal_displacement_true
    rotation_error = Rotation.from_matrix(
        result.rotation @ case.rigid_rotation_true.T
    ).magnitude()
    return {
        "result": result,
        "error": error,
        "duration_seconds": duration,
        "rotation_error_deg": float(np.rad2deg(rotation_error)),
        "translation_error_mm": float(
            np.linalg.norm(result.translation - case.rigid_translation_true)
        ),
    }


def summarize(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["method"], row["domain"], row["domain_status"])].append(row)
    summaries = []
    for (method, domain, status), group_rows in sorted(buckets.items()):
        summaries.append(
            {
                "method": method,
                "domain": domain,
                "domain_status": status,
                "scan_pairs": len(group_rows),
                "simultaneous_coverage": float(
                    np.mean([row["simultaneous_covered"] for row in group_rows])
                ),
                "normal_mae_mm": float(np.mean([row["normal_mae_mm"] for row in group_rows])),
                "normal_rmse_mm": float(np.mean([row["normal_rmse_mm"] for row in group_rows])),
                "interval_score_mm": float(
                    np.mean([row["interval_score_mm"] for row in group_rows])
                ),
                "mean_interval_width_mm": float(
                    np.mean([row["mean_interval_width_mm"] for row in group_rows])
                ),
                "translation_error_mm": float(
                    np.mean([row["translation_error_mm"] for row in group_rows])
                ),
                "rotation_error_deg": float(
                    np.mean([row["rotation_error_deg"] for row in group_rows])
                ),
                "failure_count": int(sum(not row["converged"] for row in group_rows)),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    if args.calibration_per_known_domain < 1 or args.test_per_domain < 1:
        raise ValueError("pilot counts must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ("cascade_strong", "pcudm_joint")
    known_domains = ("nominal", "noisy_sparse")

    calibration_payload = {method: [] for method in methods}
    calibration_runtime = []
    for domain_index, domain_name in enumerate(known_domains):
        domain = DOMAINS[domain_name]
        for offset in range(args.calibration_per_known_domain):
            seed = 10_000 + domain_index * 1_000 + offset
            case = generate_case(seed, domain, points_x=18, points_z=14)
            for method in methods:
                estimate = estimate_case(method, case)
                calibration_payload[method].append(
                    {
                        "group": domain_name,
                        "error": estimate["error"],
                        "scale": estimate["result"].scale,
                        "valid": case.valid_field_mask,
                    }
                )
                calibration_runtime.append(
                    {
                        "split": "calibration",
                        "case_id": case.case_id,
                        "domain": domain_name,
                        "method": method,
                        "duration_seconds": estimate["duration_seconds"],
                        "converged": estimate["result"].converged,
                    }
                )

    calibrators = {}
    calibration_states = {}
    for method in methods:
        calibrator = PairwiseSimultaneousCalibrator(alpha=0.05)
        state = calibrator.fit(calibration_payload[method])
        calibrators[method] = calibrator
        calibration_states[method] = {
            "alpha": state.alpha,
            "group_quantiles": state.group_quantiles,
            "group_counts": state.group_counts,
            "group_order_statistics": state.group_order_statistics,
            "pooled_quantile": state.pooled_quantile,
            "pooled_count": state.pooled_count,
            "pooled_order_statistic": state.pooled_order_statistic,
        }

    rows = []
    for domain_index, domain_name in enumerate((*known_domains, "unseen_compound")):
        domain = DOMAINS[domain_name]
        is_known = domain_name in known_domains
        for offset in range(args.test_per_domain):
            seed = 20_000 + domain_index * 1_000 + offset
            case = generate_case(seed, domain, points_x=18, points_z=14)
            for method in methods:
                estimate = estimate_case(method, case)
                result = estimate["result"]
                lower, upper = calibrators[method].interval(
                    result.normal_displacement,
                    result.scale,
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
                radius = 0.5 * (upper - lower)
                rows.append(
                    {
                        "split": "test",
                        "case_id": case.case_id,
                        "seed": seed,
                        "domain": domain_name,
                        "domain_status": "known_calibration_group" if is_known else "unseen_empirical",
                        "method": method,
                        "converged": result.converged,
                        "duration_seconds": estimate["duration_seconds"],
                        "translation_error_mm": estimate["translation_error_mm"],
                        "rotation_error_deg": estimate["rotation_error_deg"],
                        "reject_score_max_radius_mm": float(
                            np.max(radius[case.valid_field_mask])
                        ),
                        **metrics,
                    }
                )

    summaries = summarize(rows)
    paired = []
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["method"]] = row
    for case_id, method_rows in sorted(by_case.items()):
        if set(method_rows) != set(methods):
            continue
        cascade = method_rows["cascade_strong"]
        joint = method_rows["pcudm_joint"]
        paired.append(
            {
                "case_id": case_id,
                "domain": joint["domain"],
                "domain_status": joint["domain_status"],
                "mae_difference_joint_minus_cascade_mm": (
                    joint["normal_mae_mm"] - cascade["normal_mae_mm"]
                ),
                "interval_score_difference_joint_minus_cascade_mm": (
                    joint["interval_score_mm"] - cascade["interval_score_mm"]
                ),
                "width_difference_joint_minus_cascade_mm": (
                    joint["mean_interval_width_mm"] - cascade["mean_interval_width_mm"]
                ),
            }
        )

    aurc_rows = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        aurc_rows.append(
            {
                "method": method,
                "scan_pairs": len(selected),
                "aurc": aurc(
                    np.asarray([row["normal_mae_mm"] for row in selected]),
                    np.asarray([row["reject_score_max_radius_mm"] for row in selected]),
                ),
            }
        )

    paired_known = [row for row in paired if row["domain_status"] == "known_calibration_group"]
    decision = {
        "known_domain_mean_mae_difference_joint_minus_cascade_mm": float(
            np.mean([row["mae_difference_joint_minus_cascade_mm"] for row in paired_known])
        ),
        "known_domain_mean_interval_score_difference_joint_minus_cascade_mm": float(
            np.mean([row["interval_score_difference_joint_minus_cascade_mm"] for row in paired_known])
        ),
        "mechanism_signal_present": bool(
            np.mean([row["interval_score_difference_joint_minus_cascade_mm"] for row in paired_known]) < 0
        ),
        "note": "Development pilot only; no confirmatory inference or paper claim.",
    }
    report = {
        "schema_version": "1.0",
        "status": "DEVELOPMENT_PILOT",
        "independent_unit": "complete synthetic scan pair",
        "counts": {
            "calibration_per_known_domain": args.calibration_per_known_domain,
            "test_per_domain": args.test_per_domain,
            "known_domains": len(known_domains),
            "unseen_domains": 1,
        },
        "domains": {key: value.__dict__ for key, value in DOMAINS.items()},
        "calibration": calibration_states,
        "summaries": summaries,
        "aurc": aurc_rows,
        "paired_decision": decision,
        "rows": rows,
        "paired_rows": paired,
        "calibration_runtime": calibration_runtime,
    }
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "pilot_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "summaries": summaries,
        "aurc": aurc_rows,
        "paired_decision": decision,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
