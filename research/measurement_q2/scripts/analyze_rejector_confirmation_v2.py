"""Analyze the predeclared scale-only rejector confirmation on fresh cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCORES = ("score_scale", "score_residual", "score_ood", "score_combined")
COMPARATORS = SCORES[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def aurc(risk: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    prefix = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)
    coverage = np.arange(1, risk.size + 1) / risk.size
    return float(np.trapezoid(prefix, coverage))


def risk_at_coverage(risk: np.ndarray, score: np.ndarray, coverage: float) -> float:
    retained = max(1, int(np.ceil(coverage * risk.size)))
    order = np.argsort(score, kind="stable")
    return float(risk[order[:retained]].mean())


def analyze_scope(
    rows: list[dict], repetitions: int, rng: np.random.Generator
) -> dict:
    risk = np.asarray([row["risk_normal_mae_mm"] for row in rows], dtype=np.float64)
    domains = np.asarray([row["domain"] for row in rows], dtype=object)
    scores = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in SCORES
    }
    observed = {key: aurc(risk, scores[key]) for key in SCORES}
    indices_by_domain = [
        np.flatnonzero(domains == domain) for domain in sorted(set(domains))
    ]
    differences = {
        comparator: np.empty(repetitions, dtype=np.float64)
        for comparator in COMPARATORS
    }
    for repetition in range(repetitions):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in indices_by_domain
        ])
        scale_value = aurc(risk[sampled], scores["score_scale"][sampled])
        for comparator in COMPARATORS:
            differences[comparator][repetition] = (
                scale_value - aurc(risk[sampled], scores[comparator][sampled])
            )
    comparisons = {}
    for comparator in COMPARATORS:
        values = differences[comparator]
        ci = np.quantile(values, (0.025, 0.975))
        comparisons[comparator] = {
            "observed_scale_minus_comparator": float(
                observed["score_scale"] - observed[comparator]
            ),
            "domain_stratified_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
            "upper_ci_below_zero": bool(ci[1] < 0.0),
        }
    fixed_coverages = {}
    for key in SCORES:
        fixed_coverages[key] = {
            str(coverage): risk_at_coverage(risk, scores[key], coverage)
            for coverage in (0.50, 0.75, 0.90, 0.95)
        }
    return {
        "n_scan_pairs": len(rows),
        "domains": sorted(set(domains)),
        "aurc": observed,
        "scale_vs_comparators": comparisons,
        "risk_at_fixed_coverage_mm": fixed_coverages,
        "no_abstention_mean_risk_mm": float(risk.mean()),
        "all_comparison_upper_bounds_below_zero": all(
            item["upper_ci_below_zero"] for item in comparisons.values()
        ),
    }


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    rows = raw["selection_rows"]
    if len(rows) != 480 or len({row["case_id"] for row in rows}) != 480:
        raise ValueError("expected 480 unique v2 scan pairs")
    if any(row["case_id"].startswith(("nominal-seed-400", "compound_extreme-seed-500")) for row in rows):
        raise ValueError("v1 test seed family detected in v2")
    rng = np.random.default_rng(args.seed)
    overall = analyze_scope(rows, args.bootstrap, rng)
    known = analyze_scope(
        [row for row in rows if row["domain_status"] == "known_calibration_group"],
        args.bootstrap,
        rng,
    )
    unseen = analyze_scope(
        [row for row in rows if row["domain_status"] == "unseen_empirical"],
        args.bootstrap,
        rng,
    )
    report = {
        "schema_version": "1.0",
        "status": "REJECTOR_CONFIRMATION_V2_ANALYZED",
        "independent_unit": "complete synthetic scan pair",
        "primary_scope": "all eight domains",
        "primary_hypothesis": (
            "scale-only AURC is lower than residual-only, OOD-only, and the v1 combined score"
        ),
        "overall": overall,
        "known_secondary": known,
        "unseen_secondary_empirical": unseen,
        "primary_gate_passed": overall["all_comparison_upper_bounds_below_zero"],
        "nonconverged_method_rows": sum(
            not bool(row["converged"]) for row in raw["rows"]
        ),
        "analysis_parameters": {
            "seed": args.seed,
            "domain_stratified_bootstrap_repetitions": args.bootstrap,
        },
        "claim_boundary": (
            "v2 confirms only selective-risk ranking; unseen-domain coverage remains empirical"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "primary_gate_passed": report["primary_gate_passed"],
        "overall": report["overall"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
