"""Analyze all four predeclared full-field grid-sensitivity variants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs=4, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def analyze(path: Path, repetitions: int, rng: np.random.Generator) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = raw["config"]
    grid = f"{config['grid']['points_x']}x{config['grid']['points_z']}"
    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in raw["rows"]:
        indexed[row["case_id"]][row["method"]] = row
    case_ids = sorted(
        case_id
        for case_id, methods in indexed.items()
        if methods["das_grouped"]["domain_status"] == "known_calibration_group"
    )
    if len(case_ids) != 240:
        raise ValueError(f"{grid}: expected 240 known test pairs")
    differences = np.asarray([
        indexed[case_id]["das_grouped"]["interval_score_mm"]
        - indexed[case_id]["homoscedastic_grouped"]["interval_score_mm"]
        for case_id in case_ids
    ])
    domains = np.asarray(
        [indexed[case_id]["das_grouped"]["domain"] for case_id in case_ids],
        dtype=object,
    )
    domain_indices = [
        np.flatnonzero(domains == domain) for domain in sorted(set(domains))
    ]
    boot = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in domain_indices
        ])
        boot[repetition] = differences[sampled].mean()
    ci = np.quantile(boot, (0.025, 0.975))
    das_rows = [indexed[case_id]["das_grouped"] for case_id in case_ids]
    homoscedastic_rows = [
        indexed[case_id]["homoscedastic_grouped"] for case_id in case_ids
    ]
    unseen = [
        methods["das_grouped"]
        for methods in indexed.values()
        if methods["das_grouped"]["domain_status"] == "unseen_empirical"
    ]
    return {
        "grid": grid,
        "points_x": int(config["grid"]["points_x"]),
        "points_z": int(config["grid"]["points_z"]),
        "known_scan_pairs": len(das_rows),
        "unseen_scan_pairs": len(unseen),
        "mean_field_locations": float(np.mean([row["field_locations"] for row in das_rows])),
        "das_known_simultaneous_coverage": float(
            np.mean([row["simultaneous_covered"] for row in das_rows])
        ),
        "das_known_mean_interval_width_mm": float(
            np.mean([row["mean_interval_width_mm"] for row in das_rows])
        ),
        "das_known_mean_normalized_interval_width": float(
            np.mean([row["normalized_mean_interval_width"] for row in das_rows])
        ),
        "homoscedastic_known_mean_interval_width_mm": float(
            np.mean([row["mean_interval_width_mm"] for row in homoscedastic_rows])
        ),
        "das_minus_homoscedastic_interval_score_mm": float(differences.mean()),
        "domain_stratified_bootstrap_95_ci_mm": [float(ci[0]), float(ci[1])],
        "unseen_empirical_simultaneous_coverage": float(
            np.mean([row["simultaneous_covered"] for row in unseen])
        ),
        "nonconverged_method_rows": sum(not bool(row["converged"]) for row in raw["rows"]),
        "seed_bases": config["seed_bases"],
    }


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    variants = [analyze(path, args.bootstrap, rng) for path in args.inputs]
    variants.sort(key=lambda row: (row["points_x"], row["points_z"]))
    labels = {row["grid"] for row in variants}
    if labels != {"12x9", "18x14", "24x18", "30x22"}:
        raise ValueError(f"unexpected grid set: {sorted(labels)}")
    reference = next(row for row in variants if row["grid"] == "18x14")
    reference_width = reference["das_known_mean_normalized_interval_width"]
    for row in variants:
        row["normalized_width_ratio_to_18x14"] = float(
            row["das_known_mean_normalized_interval_width"] / reference_width
        )
        row["coverage_gate"] = row["das_known_simultaneous_coverage"] >= 0.90
        row["interval_score_gate"] = (
            row["domain_stratified_bootstrap_95_ci_mm"][1] < 0.0
        )
        row["normalized_width_gate"] = (
            0.75 <= row["normalized_width_ratio_to_18x14"] <= 1.25
        )
        row["variant_gate_passed"] = all((
            row["coverage_gate"],
            row["interval_score_gate"],
            row["normalized_width_gate"],
        ))
    report = {
        "schema_version": "1.0",
        "status": "GRID_SENSITIVITY_ANALYSIS_COMPLETE",
        "independent_unit": "complete synthetic scan pair",
        "formal_v1_replacement": False,
        "variants": variants,
        "all_four_grid_gates_passed": all(row["variant_gate_passed"] for row in variants),
        "analysis_parameters": {
            "seed": args.seed,
            "domain_stratified_bootstrap_repetitions": args.bootstrap,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
