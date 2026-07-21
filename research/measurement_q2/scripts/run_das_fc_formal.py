"""Execute the frozen multi-domain DAS-FC confirmatory synthetic experiment."""

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
from research.measurement_q2.pcudm.metrics import aurc, scan_pair_metrics
from research.measurement_q2.pcudm.simultaneous_baselines import (
    BonferroniGaussianBand,
    ClassicalMaxTBand,
)
from research.measurement_q2.pcudm.registration_frontends import register_multiscale


PAIR_METHODS = (
    "homoscedastic_grouped",
    "raw_local_grouped",
    "learned_grouped",
    "das_grouped",
    "das_pooled",
    "das_pointwise",
    "das_no_fallback",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def domain_from_config(name: str, spec: dict) -> tuple[SyntheticDomain, dict]:
    return SyntheticDomain(name=name, **spec["sensor"]), dict(spec.get("geometry", {}))


def case_seed(
    config: dict,
    stage: str,
    domain_name: str,
    domain_index: int,
    offset: int,
) -> int:
    """Resolve an optional frozen seed list without changing formal-v1 defaults."""
    explicit = config.get("explicit_seeds", {}).get(stage, {}).get(domain_name)
    if explicit is not None:
        if offset >= len(explicit):
            raise ValueError(
                f"explicit seed list too short for {stage}/{domain_name}: "
                f"need index {offset}, have {len(explicit)}"
            )
        return int(explicit[offset])
    return int(config["seed_bases"][stage] + domain_index * 10_000 + offset)


def estimate(
    seed: int,
    domain: SyntheticDomain,
    geometry: dict,
    grid: dict,
    estimator_config: dict | None = None,
) -> dict:
    case = generate_case(seed, domain, **grid, **geometry)
    started = time.perf_counter()
    estimator_config = estimator_config or {}
    frontend = str(estimator_config.get("frontend", "cascade_strong"))
    if frontend == "cascade_strong":
        initial = None
    elif frontend in {"multiscale_trimmed_ptp", "robust_ptpl"}:
        initial = register_multiscale(case.reference, case.target, frontend)
    else:
        raise ValueError(f"unknown estimator frontend: {frontend}")
    result = PCUDMFieldEstimator(
        mode="cascade",
        icp_iterations=14,
        query_workers=int(estimator_config.get("query_workers", -1)),
    ).fit(
        case.reference,
        case.target,
        case.panel_ids,
        case.support_candidates,
        initial_rotation=None if initial is None else initial.rotation,
        initial_translation=None if initial is None else initial.translation,
        pose_locked=initial is not None,
    )
    return {
        "case": case,
        "result": result,
        "error": result.normal_displacement - case.normal_displacement_true,
        "duration_seconds": time.perf_counter() - started,
        "frontend": frontend,
    }


def scan_features(estimated: dict) -> np.ndarray:
    result = estimated["result"]
    valid = estimated["case"].valid_field_mask
    return np.asarray(
        [
            np.median(result.scale[valid]),
            np.quantile(result.scale[valid], 0.95),
            np.median(result.match_distance[valid]),
            np.quantile(result.match_distance[valid], 0.95),
            np.mean(result.support_probability[valid]),
            np.mean(result.valid[valid]),
            np.quantile(np.abs(result.normal_displacement[valid]), 0.95),
        ],
        dtype=np.float64,
    )


def base_scales(estimated: dict, model: StructuralScaleModel) -> dict[str, np.ndarray]:
    case = estimated["case"]
    result = estimated["result"]
    constant_value = max(float(np.median(result.scale[case.valid_field_mask])), 1e-6)
    return {
        "homoscedastic": np.full(result.scale.shape, constant_value, dtype=np.float64),
        "raw_local": result.scale,
        "learned": model.predict(case.reference, case.panel_ids, result),
    }


def das_scale(base: dict[str, np.ndarray], blend: float) -> np.ndarray:
    return np.exp(
        (1.0 - blend) * np.log(base["homoscedastic"])
        + blend * np.log(base["learned"])
    )


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    return median, max(scale, 1e-6)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config["experiment_id"] = config["experiment_id"] + "-smoke"
        config["run_mode"] = "engineering_smoke_not_evidence"
        config["counts_per_known_domain"] = {
            "tuning": 4,
            "validation": 3,
            "calibration": 20 if config.get("classical_baselines", False) else 5,
            "test": 3,
        }
        config["counts_per_unseen_domain"] = {
            "test": 0
            if config.get("classical_baselines", False) or "p2_metadata" in config
            else 3
        }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    alpha = float(config["alpha"])
    classical_enabled = bool(config.get("classical_baselines", False))
    if classical_enabled and config["counts_per_unseen_domain"]["test"] != 0:
        raise ValueError(
            "classical P3 baselines are frozen for known groups only; unseen test count must be zero"
        )
    grid = config["grid"]
    known = {
        name: domain_from_config(name, spec)
        for name, spec in config["known_domains"].items()
    }
    unseen = {
        name: domain_from_config(name, spec)
        for name, spec in config["unseen_domains"].items()
    }

    progress_stream = progress_path.open("w", encoding="utf-8")

    def log_progress(stage: str, completed: int, total: int, detail: str = ""):
        progress_stream.write(json.dumps({
            "time": time.time(), "stage": stage, "completed": completed,
            "total": total, "detail": detail,
        }) + "\n")
        progress_stream.flush()

    tuning_cases = []
    tuning_features = []
    tuning_count = config["counts_per_known_domain"]["tuning"]
    total = tuning_count * len(known)
    completed = 0
    for domain_index, (domain_name, (domain, geometry)) in enumerate(known.items()):
        for offset in range(tuning_count):
            seed = case_seed(config, "tuning", domain_name, domain_index, offset)
            estimated = estimate(seed, domain, geometry, grid, config.get("estimator"))
            case = estimated["case"]
            tuning_cases.append({
                "case_id": case.case_id,
                "group": domain_name,
                "reference": case.reference,
                "panel_ids": case.panel_ids,
                "result": estimated["result"],
                "error": estimated["error"],
                "valid": case.valid_field_mask,
            })
            tuning_features.append(scan_features(estimated))
            completed += 1
            if completed % 20 == 0 or completed == total:
                log_progress("tuning_estimation", completed, total, domain_name)

    scale_model = StructuralScaleModel(**config["scale_model"])
    scale_diagnostics = scale_model.fit(tuning_cases)
    classical_tuning_cases = None
    bonferroni_band = None
    if classical_enabled:
        classical_tuning_cases = [
            {
                "group": item["group"],
                "error": item["error"],
                "valid": item["valid"],
            }
            for item in tuning_cases
        ]
        bonferroni_band = BonferroniGaussianBand(alpha=alpha)
        bonferroni_band.fit(classical_tuning_cases)
    tuning_feature_matrix = np.vstack(tuning_features)
    ood_center = tuning_feature_matrix.mean(axis=0)
    ood_scale = np.maximum(tuning_feature_matrix.std(axis=0, ddof=1), 1e-6)
    del tuning_cases

    validation_count = config["counts_per_known_domain"]["validation"]
    validation_by_domain = {}
    blend_by_domain = {}
    blend_validation = {}
    for domain_index, (domain_name, (domain, geometry)) in enumerate(known.items()):
        validation = []
        for offset in range(validation_count):
            seed = case_seed(config, "validation", domain_name, domain_index, offset)
            validation.append(
                estimate(seed, domain, geometry, grid, config.get("estimator"))
            )
        validation_by_domain[domain_name] = validation
        candidates = []
        for blend in config["blend_grid"]:
            cases = []
            mean_scales = []
            for estimated in validation:
                case = estimated["case"]
                scales = base_scales(estimated, scale_model)
                scale = das_scale(scales, float(blend))
                cases.append({
                    "group": domain_name, "error": estimated["error"],
                    "scale": scale, "valid": case.valid_field_mask,
                })
                mean_scales.append(float(np.mean(scale[case.valid_field_mask])))
            state = PairwiseSimultaneousCalibrator(alpha=alpha).fit(cases)
            quantile = state.group_quantiles[domain_name]
            candidates.append({
                "blend": float(blend),
                "validation_quantile": quantile,
                "validation_mean_width_mm": float(2 * quantile * np.mean(mean_scales)),
            })
        selected = min(candidates, key=lambda item: (item["validation_mean_width_mm"], item["blend"]))
        blend_by_domain[domain_name] = selected["blend"]
        blend_validation[domain_name] = {"selected": selected, "candidates": candidates}
        log_progress("validation_blend", domain_index + 1, len(known), domain_name)

    # Freeze rejection-score normalization entirely from known validation pairs.
    validation_score_components = {"scale": [], "residual": [], "ood": []}
    for domain_name, validation in validation_by_domain.items():
        for estimated in validation:
            case = estimated["case"]
            scales = base_scales(estimated, scale_model)
            adaptive = das_scale(scales, blend_by_domain[domain_name])
            validation_score_components["scale"].append(
                float(np.mean(adaptive[case.valid_field_mask]))
            )
            validation_score_components["residual"].append(
                float(np.median(estimated["result"].match_distance[case.valid_field_mask]))
            )
            feature = scan_features(estimated)
            validation_score_components["ood"].append(
                float(np.sqrt(np.mean(((feature - ood_center) / ood_scale) ** 2)))
            )
    score_norm = {
        key: robust_location_scale(np.asarray(values))
        for key, values in validation_score_components.items()
    }
    validation_combined = []
    for index in range(len(validation_score_components["scale"])):
        components = []
        for key in ("scale", "residual", "ood"):
            median, spread = score_norm[key]
            components.append((validation_score_components[key][index] - median) / spread)
        validation_combined.append(max(components))
    reject_threshold = float(np.quantile(validation_combined, 0.95, method="higher"))
    del validation_by_domain

    calibration_payload = {
        "homoscedastic": [], "raw_local": [], "learned": [], "das": []
    }
    pointwise_scores = defaultdict(list)
    classical_calibration_cases = []
    calibration_count = config["counts_per_known_domain"]["calibration"]
    total = calibration_count * len(known)
    completed = 0
    for domain_index, (domain_name, (domain, geometry)) in enumerate(known.items()):
        for offset in range(calibration_count):
            seed = case_seed(config, "calibration", domain_name, domain_index, offset)
            estimated = estimate(seed, domain, geometry, grid, config.get("estimator"))
            case = estimated["case"]
            scales = base_scales(estimated, scale_model)
            adaptive = das_scale(scales, blend_by_domain[domain_name])
            method_scales = {
                "homoscedastic": scales["homoscedastic"],
                "raw_local": scales["raw_local"],
                "learned": scales["learned"],
                "das": adaptive,
            }
            for method, scale in method_scales.items():
                calibration_payload[method].append({
                    "group": domain_name,
                    "error": estimated["error"],
                    "scale": scale,
                    "valid": case.valid_field_mask,
                })
            valid = case.valid_field_mask
            if classical_enabled:
                classical_calibration_cases.append({
                    "group": domain_name,
                    "error": estimated["error"],
                    "valid": valid,
                })
            standardized = np.abs(estimated["error"][valid]) / adaptive[valid]
            pointwise_scores[domain_name].extend(standardized.tolist())
            pointwise_scores["__pooled__"].extend(standardized.tolist())
            completed += 1
            if completed % 50 == 0 or completed == total:
                log_progress("calibration_estimation", completed, total, domain_name)

    calibrators = {}
    calibration_states = {}
    for method, cases in calibration_payload.items():
        calibrator = PairwiseSimultaneousCalibrator(alpha=alpha)
        state = calibrator.fit(cases)
        calibrators[method] = calibrator
        calibration_states[method] = {
            "group_quantiles": state.group_quantiles,
            "group_counts": state.group_counts,
            "group_order_statistics": state.group_order_statistics,
            "pooled_quantile": state.pooled_quantile,
            "pooled_count": state.pooled_count,
            "pooled_order_statistic": state.pooled_order_statistic,
        }
    pointwise_quantiles = {
        group: float(np.quantile(values, 1.0 - alpha, method="higher"))
        for group, values in pointwise_scores.items()
    }
    pointwise_counts = {group: len(values) for group, values in pointwise_scores.items()}
    max_t_band = None
    if classical_enabled:
        assert classical_tuning_cases is not None and bonferroni_band is not None
        max_t_band = ClassicalMaxTBand(alpha=alpha)
        max_t_state = max_t_band.fit(
            classical_tuning_cases, classical_calibration_cases
        )
        calibration_states["classical_max_t"] = {
            "group_quantiles": max_t_state.group_quantiles,
            "group_counts": max_t_state.group_calibration_counts,
            "group_order_statistics": max_t_state.group_order_statistics,
        }
        calibration_states["bonferroni_gaussian"] = {
            group: bonferroni_band.diagnostics(group) for group in known
        }
    del calibration_payload, pointwise_scores

    rows = []
    selection_rows = []
    test_specs = [
        ("known_calibration_group", "test_known", known,
         config["counts_per_known_domain"]["test"]),
        ("unseen_empirical", "test_unseen", unseen,
         config["counts_per_unseen_domain"]["test"]),
    ]
    total = sum(len(domains) * count for _, _, domains, count in test_specs)
    completed = 0
    for status, seed_stage, domains, test_count in test_specs:
        for domain_index, (domain_name, (domain, geometry)) in enumerate(domains.items()):
            is_known = status == "known_calibration_group"
            for offset in range(test_count):
                seed = case_seed(config, seed_stage, domain_name, domain_index, offset)
                estimated = estimate(
                    seed, domain, geometry, grid, config.get("estimator")
                )
                case = estimated["case"]
                result = estimated["result"]
                scales = base_scales(estimated, scale_model)
                blend = blend_by_domain[domain_name] if is_known else 0.0
                adaptive = das_scale(scales, blend)
                prediction = result.normal_displacement
                valid = case.valid_field_mask

                interval_specs = {}
                for output_method, calibration_method, scale in (
                    ("homoscedastic_grouped", "homoscedastic", scales["homoscedastic"]),
                    ("raw_local_grouped", "raw_local", scales["raw_local"]),
                    ("learned_grouped", "learned", scales["learned"]),
                ):
                    calibrator = calibrators[calibration_method]
                    q = (
                        calibrator.quantile(domain_name)
                        if is_known
                        else calibrator.state.pooled_quantile
                    )
                    interval_specs[output_method] = (scale, q)

                if is_known:
                    q_das = calibrators["das"].quantile(domain_name)
                    full_scale = adaptive
                else:
                    q_das = calibrators["homoscedastic"].state.pooled_quantile
                    full_scale = scales["homoscedastic"]
                interval_specs["das_grouped"] = (full_scale, q_das)
                interval_specs["das_pooled"] = (
                    adaptive, calibrators["das"].state.pooled_quantile
                )
                interval_specs["das_pointwise"] = (
                    adaptive,
                    pointwise_quantiles[domain_name] if is_known else pointwise_quantiles["__pooled__"],
                )
                interval_specs["das_no_fallback"] = (
                    scales["learned"],
                    calibrators["learned"].quantile(domain_name)
                    if is_known else calibrators["learned"].state.pooled_quantile,
                )
                if classical_enabled:
                    assert max_t_band is not None and bonferroni_band is not None
                    max_t_lower, max_t_upper = max_t_band.interval(
                        prediction, domain_name
                    )
                    bonf_lower, bonf_upper = bonferroni_band.interval(
                        prediction, domain_name
                    )
                    interval_specs["classical_max_t"] = (
                        0.5 * (max_t_upper - max_t_lower),
                        1.0,
                    )
                    interval_specs["bonferroni_gaussian"] = (
                        0.5 * (bonf_upper - bonf_lower),
                        1.0,
                    )

                full_row = None
                for method, (scale, quantile) in interval_specs.items():
                    radius = quantile * scale
                    metrics = scan_pair_metrics(
                        case.normal_displacement_true,
                        prediction,
                        prediction - radius,
                        prediction + radius,
                        valid=valid,
                        alpha=alpha,
                    )
                    row = {
                        "case_id": case.case_id,
                        "seed": seed,
                        "domain": domain_name,
                        "domain_status": status,
                        "method": method,
                        "converged": result.converged,
                        "duration_seconds": estimated["duration_seconds"],
                        "calibration_quantile": float(quantile),
                        "mean_scale_mm": float(np.mean(scale[valid])),
                        **metrics,
                    }
                    rows.append(row)
                    if method == "das_grouped":
                        full_row = row

                feature = scan_features(estimated)
                component_values = {
                    "scale": float(np.mean(full_scale[valid])),
                    "residual": float(np.median(result.match_distance[valid])),
                    "ood": float(np.sqrt(np.mean(((feature - ood_center) / ood_scale) ** 2))),
                }
                standardized = {
                    key: (value - score_norm[key][0]) / score_norm[key][1]
                    for key, value in component_values.items()
                }
                combined = max(standardized.values())
                selection_rows.append({
                    "case_id": case.case_id,
                    "domain": domain_name,
                    "domain_status": status,
                    "risk_normal_mae_mm": full_row["normal_mae_mm"],
                    "score_scale": component_values["scale"],
                    "score_residual": component_values["residual"],
                    "score_ood": component_values["ood"],
                    "score_combined": combined,
                    "rejected": bool(combined > reject_threshold),
                })
                completed += 1
                if completed % 40 == 0 or completed == total:
                    log_progress("test_estimation", completed, total, domain_name)

    progress_stream.close()

    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["method"], row["domain"], row["domain_status"])].append(row)
    summaries = []
    for (method, domain, status), group_rows in sorted(buckets.items()):
        summaries.append({
            "method": method,
            "domain": domain,
            "domain_status": status,
            "scan_pairs": len(group_rows),
            "simultaneous_coverage": float(np.mean([r["simultaneous_covered"] for r in group_rows])),
            "point_coverage": float(np.mean([r["point_coverage"] for r in group_rows])),
            "normal_mae_mm": float(np.mean([r["normal_mae_mm"] for r in group_rows])),
            "mean_interval_width_mm": float(np.mean([r["mean_interval_width_mm"] for r in group_rows])),
            "interval_score_mm": float(np.mean([r["interval_score_mm"] for r in group_rows])),
            "failure_count": int(sum(not r["converged"] for r in group_rows)),
        })

    aurc_results = []
    risk = np.asarray([row["risk_normal_mae_mm"] for row in selection_rows])
    for key in ("score_scale", "score_residual", "score_ood", "score_combined"):
        aurc_results.append({
            "rejector": key,
            "scan_pairs": len(selection_rows),
            "aurc": aurc(risk, np.asarray([row[key] for row in selection_rows])),
        })

    report = {
        "schema_version": "1.0",
        "status": "CONFIRMATORY_EXECUTION_COMPLETE_UNANALYZED",
        "experiment_id": config["experiment_id"],
        "config": config,
        "independent_unit": "complete synthetic scan pair",
        "scale_model": {
            "scan_pairs": scale_diagnostics.scan_pairs,
            "training_locations": scale_diagnostics.training_locations,
            "target_quantile": scale_diagnostics.target_quantile,
            "feature_names": scale_diagnostics.feature_names,
        },
        "blend_by_domain": blend_by_domain,
        "blend_validation": blend_validation,
        "calibration": calibration_states,
        "classical_baselines_enabled": classical_enabled,
        "pointwise_calibration": {
            "quantiles": pointwise_quantiles,
            "location_counts": pointwise_counts,
            "warning": "points are pseudoreplicated; baseline only",
        },
        "rejector": {
            "validation_threshold": reject_threshold,
            "normalization": {
                key: {"median": value[0], "mad_scale": value[1]}
                for key, value in score_norm.items()
            },
            "aurc": aurc_results,
        },
        "summaries": summaries,
        "rows": rows,
        "selection_rows": selection_rows,
    }
    (output_dir / "formal_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "formal_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "selection_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    print(json.dumps({
        "status": report["status"],
        "blend_by_domain": blend_by_domain,
        "aurc": aurc_results,
        "summary_rows": len(summaries),
        "pair_rows": len(rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
