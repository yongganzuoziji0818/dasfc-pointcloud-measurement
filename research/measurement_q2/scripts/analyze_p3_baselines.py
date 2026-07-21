"""Analyze the fresh-seed P3 simultaneous-band benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.measurement_q2.scripts.analyze_das_fc_formal import (
    KNOWN_STATUS,
    aggregate_method_table,
    coverage_table,
    holm_adjust,
    method_rows_by_case,
    paired_contrast,
)


CONTRASTS = (
    "homoscedastic_grouped",
    "classical_max_t",
    "raw_local_grouped",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=50_000)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not raw.get("classical_baselines_enabled"):
        raise ValueError("input was not executed with classical P3 baselines")
    indexed = method_rows_by_case(raw["rows"])
    rng = np.random.default_rng(args.seed)
    contrasts = []
    p_values = []
    for baseline in CONTRASTS:
        result, _ = paired_contrast(
            indexed, baseline, rng, args.bootstrap, args.permutations
        )
        contrasts.append(result)
        p_values.append(result["paired_t"]["p_two_sided"])
    for result, adjusted in zip(contrasts, holm_adjust(p_values), strict=True):
        result["paired_t"]["p_holm_three_predeclared"] = float(adjusted)

    aggregate = aggregate_method_table(raw["rows"])
    coverage = coverage_table(raw["rows"])
    aggregate_known = {
        item["method"]: item
        for item in aggregate
        if item["domain_status"] == KNOWN_STATUS
    }
    required_methods = {
        "das_grouped",
        "homoscedastic_grouped",
        "classical_max_t",
        "bonferroni_gaussian",
        "raw_local_grouped",
    }
    missing = sorted(required_methods - aggregate_known.keys())
    if missing:
        raise ValueError(f"missing P3 methods: {missing}")

    contrast_pass = {
        item["baseline"]: (
            item["stratified_bootstrap_95_ci_mm"][1] < 0.0
            and item["paired_t"]["p_holm_three_predeclared"] < 0.05
        )
        for item in contrasts
    }
    das = aggregate_known["das_grouped"]
    coverage_efficiency_pass = (
        das["simultaneous_coverage"] >= 0.90
        and das["mean_interval_width_mm"]
        < aggregate_known["homoscedastic_grouped"]["mean_interval_width_mm"]
        and das["mean_interval_width_mm"]
        < aggregate_known["classical_max_t"]["mean_interval_width_mm"]
    )
    report = {
        "schema_version": "1.0",
        "status": "P3_CLASSICAL_BASELINE_ANALYSIS_COMPLETE",
        "source_experiment_id": raw["experiment_id"],
        "independent_unit": "complete fresh-seed synthetic scan pair",
        "confirmatory_scope": "known groups in the prospective P3 extension",
        "contrasts": contrasts,
        "contrast_pass": contrast_pass,
        "aggregate_by_method": aggregate,
        "coverage_by_domain": coverage,
        "coverage_efficiency_pass": bool(coverage_efficiency_pass),
        "p3_gate_passed": bool(
            contrast_pass["homoscedastic_grouped"]
            and contrast_pass["classical_max_t"]
            and coverage_efficiency_pass
        ),
        "analysis_parameters": {
            "seed": args.seed,
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "holm_family": list(CONTRASTS),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P3 classical simultaneous-band analysis",
        "",
        "All methods share the same point estimate and fresh-seed test scan pairs.",
        "",
        "| baseline | mean DAS-minus-baseline score (mm) | bootstrap 95% CI | Holm p | pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for item in contrasts:
        interval = item["stratified_bootstrap_95_ci_mm"]
        lines.append(
            f"| {item['baseline']} | {item['mean_difference_mm']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{item['paired_t']['p_holm_three_predeclared']:.4g} | "
            f"{'PASS' if contrast_pass[item['baseline']] else 'FAIL'} |"
        )
    lines.extend([
        "",
        f"Coverage-efficiency gate: {'PASS' if coverage_efficiency_pass else 'FAIL'}.",
        f"Overall P3 gate: {'PASS' if report['p3_gate_passed'] else 'FAIL'}.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "contrast_pass": contrast_pass,
        "coverage_efficiency_pass": coverage_efficiency_pass,
        "p3_gate_passed": report["p3_gate_passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

