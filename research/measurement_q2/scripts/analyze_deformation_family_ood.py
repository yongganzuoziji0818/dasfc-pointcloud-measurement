"""Analyze the frozen cross-deformation-family stress test at scan-pair level.

This analysis is descriptive and robustness-oriented.  The ``rbf_kink`` cases
were never used for model fitting, shrinkage selection, conformal calibration,
or rejector selection, so no nominal coverage guarantee is attached to them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


OOD_DOMAIN = "deformation_family_ood"
FALLBACK = "das_grouped"
NO_FALLBACK = "das_no_fallback"
SCORES = ("score_scale", "score_residual", "score_ood", "score_combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite statistic: {value}")
    return value


def percentile_ci(values: np.ndarray) -> list[float]:
    return [finite(np.quantile(values, 0.025)), finite(np.quantile(values, 0.975))]


def wilson(successes: int, total: int) -> list[float]:
    z = float(stats.norm.ppf(0.975))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [finite(centre - half), finite(centre + half)]


def aurc(risk: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    prefix_risk = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)
    coverage = np.arange(1, risk.size + 1) / risk.size
    return finite(np.trapezoid(prefix_risk, coverage))


def method_summary(rows: list[dict]) -> dict:
    successes = sum(bool(row["simultaneous_covered"]) for row in rows)
    return {
        "n_scan_pairs": len(rows),
        "normal_mae_mm": finite(np.mean([row["normal_mae_mm"] for row in rows])),
        "simultaneous_successes": successes,
        "simultaneous_coverage": finite(successes / len(rows)),
        "simultaneous_coverage_wilson_95_ci": wilson(successes, len(rows)),
        "mean_point_coverage": finite(np.mean([row["point_coverage"] for row in rows])),
        "mean_interval_width_mm": finite(
            np.mean([row["mean_interval_width_mm"] for row in rows])
        ),
        "mean_interval_score_mm": finite(
            np.mean([row["interval_score_mm"] for row in rows])
        ),
        "failure_count": sum(not bool(row["converged"]) for row in rows),
    }


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [row for row in raw["rows"] if row["domain"] == OOD_DOMAIN]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if FALLBACK not in by_method or NO_FALLBACK not in by_method:
        raise ValueError("required fallback comparison is absent")
    if any(len(group) != 240 for group in by_method.values()):
        raise ValueError("expected 240 complete scan pairs per method")

    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        indexed[row["case_id"]][row["method"]] = row
    case_ids = sorted(indexed)
    if len(case_ids) != 240:
        raise ValueError("expected 240 unique OOD scan pairs")

    fallback = np.asarray(
        [[indexed[c][FALLBACK]["simultaneous_covered"],
          indexed[c][FALLBACK]["mean_interval_width_mm"],
          indexed[c][FALLBACK]["interval_score_mm"]] for c in case_ids],
        dtype=np.float64,
    )
    no_fallback = np.asarray(
        [[indexed[c][NO_FALLBACK]["simultaneous_covered"],
          indexed[c][NO_FALLBACK]["mean_interval_width_mm"],
          indexed[c][NO_FALLBACK]["interval_score_mm"]] for c in case_ids],
        dtype=np.float64,
    )
    paired = fallback - no_fallback
    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, len(case_ids), size=(args.bootstrap, len(case_ids)))
    boot = paired[sampled].mean(axis=1)

    selection = [
        row for row in raw["selection_rows"] if row["domain"] == OOD_DOMAIN
    ]
    if len(selection) != 240:
        raise ValueError("expected 240 OOD selective-risk rows")
    risk = np.asarray([row["risk_normal_mae_mm"] for row in selection], dtype=np.float64)
    score_arrays = {
        key: np.asarray([row[key] for row in selection], dtype=np.float64)
        for key in SCORES
    }
    observed_aurc = {key: aurc(risk, score_arrays[key]) for key in SCORES}
    aurc_boot = {key: np.empty(args.bootstrap, dtype=np.float64) for key in SCORES}
    for repetition, indices in enumerate(sampled):
        for key in SCORES:
            aurc_boot[key][repetition] = aurc(risk[indices], score_arrays[key][indices])
    scale_comparisons = {}
    for key in SCORES[1:]:
        differences = aurc_boot["score_scale"] - aurc_boot[key]
        scale_comparisons[key] = {
            "observed_scale_minus_comparator": finite(
                observed_aurc["score_scale"] - observed_aurc[key]
            ),
            "paired_bootstrap_95_ci": percentile_ci(differences),
        }

    report = {
        "schema_version": "1.0",
        "status": "DEFORMATION_FAMILY_OOD_ANALYZED",
        "independent_unit": "complete synthetic scan pair",
        "scope": "post-formal empirical stress test without a coverage guarantee",
        "deformation_family": "rbf_kink",
        "n_scan_pairs": len(case_ids),
        "methods": {key: method_summary(value) for key, value in sorted(by_method.items())},
        "fallback_minus_no_fallback": {
            "simultaneous_coverage_difference": finite(paired[:, 0].mean()),
            "simultaneous_coverage_difference_bootstrap_95_ci": percentile_ci(boot[:, 0]),
            "mean_width_difference_mm": finite(paired[:, 1].mean()),
            "mean_width_difference_bootstrap_95_ci_mm": percentile_ci(boot[:, 1]),
            "mean_interval_score_difference_mm": finite(paired[:, 2].mean()),
            "mean_interval_score_difference_bootstrap_95_ci_mm": percentile_ci(boot[:, 2]),
            "interpretation": (
                "Fallback raises empirical simultaneous coverage but also widens intervals "
                "and worsens interval score; neither policy dominates on coverage and efficiency."
            ),
        },
        "selective_risk": {
            "aurc": observed_aurc,
            "lower_is_better": True,
            "scale_vs_comparators": scale_comparisons,
            "interpretation": (
                "The rejector was frozen before this deformation-family stress test; results "
                "are an external robustness audit rather than a new selection step."
            ),
        },
        "robustness_verdict": "MIXED_EMPIRICAL_ROBUSTNESS",
        "claim_boundary": (
            "Observed robustness is limited to the tested modal_bulge and rbf_kink generators; "
            "do not claim arbitrary deformation-family or field coverage guarantees."
        ),
        "analysis_parameters": {
            "seed": args.seed,
            "paired_bootstrap_repetitions": args.bootstrap,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fb = report["methods"][FALLBACK]
    no = report["methods"][NO_FALLBACK]
    selective = report["selective_risk"]["aurc"]
    lines = [
        "# Deformation-family OOD statistical audit",
        "",
        f"- Independent unit: complete scan pair; n={len(case_ids)}.",
        f"- Shared point-estimator normal MAE: {fb['normal_mae_mm']:.3f} mm; failures: {fb['failure_count']}.",
        f"- Fallback: coverage {fb['simultaneous_coverage']:.3f}, width {fb['mean_interval_width_mm']:.3f} mm, score {fb['mean_interval_score_mm']:.3f} mm.",
        f"- No fallback: coverage {no['simultaneous_coverage']:.3f}, width {no['mean_interval_width_mm']:.3f} mm, score {no['mean_interval_score_mm']:.3f} mm.",
        f"- Selective-risk AURC (scale/residual/OOD/combined): {selective['score_scale']:.4f}/{selective['score_residual']:.4f}/{selective['score_ood']:.4f}/{selective['score_combined']:.4f}.",
        "",
        "Verdict: **MIXED_EMPIRICAL_ROBUSTNESS**. Fallback recovers coverage at a material efficiency cost; no arbitrary-family or field coverage guarantee is supported.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
