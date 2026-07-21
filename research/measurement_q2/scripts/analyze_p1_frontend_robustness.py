"""Aggregate P1 front-end results at the complete scan-pair level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for statistics_path in sorted(args.runs_dir.glob("*/formal_statistics.json")):
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        raw = json.loads(
            (statistics_path.parent / "formal_report.json").read_text(encoding="utf-8")
        )
        frontend = raw["config"]["p1_metadata"]["frontend"]
        primary = next(
            item for item in statistics["paired_interval_score_contrasts"]
            if item["baseline"] == "homoscedastic_grouped"
        )
        das = next(
            item for item in statistics["aggregate_by_method"]
            if item["domain_status"] == "known_calibration_group"
            and item["method"] == "das_grouped"
        )
        rows.append({
            "frontend": frontend,
            "mean_das_minus_homoscedastic_score_mm": primary["mean_difference_mm"],
            "bootstrap_95_ci_mm": primary["stratified_bootstrap_95_ci_mm"],
            "paired_dz": primary["paired_cohen_dz"],
            "known_coverage": das["simultaneous_coverage"],
            "known_mean_width_mm": das["mean_interval_width_mm"],
            "known_mean_interval_score_mm": das["mean_interval_score_mm"],
            "failure_count": das["failure_count"],
            "negative_effect": primary["mean_difference_mm"] < 0.0,
            "ci_excludes_zero_in_expected_direction": (
                primary["stratified_bootstrap_95_ci_mm"][1] < 0.0
            ),
        })
    expected = {"cascade_strong", "multiscale_trimmed_ptp", "robust_ptpl"}
    observed = {row["frontend"] for row in rows}
    if observed != expected:
        raise ValueError(f"P1 front-end set mismatch: expected={expected}, observed={observed}")
    report = {
        "schema_version": "1.0",
        "status": "P1_FRONTEND_ROBUSTNESS_ANALYSIS_COMPLETE",
        "shared_scan_pairs": True,
        "rows": rows,
        "negative_effect_frontends": sum(row["negative_effect"] for row in rows),
        "ci_excluding_zero_frontends": sum(
            row["ci_excludes_zero_in_expected_direction"] for row in rows
        ),
        "p1_gate_passed": (
            all(row["negative_effect"] for row in rows)
            and sum(row["ci_excludes_zero_in_expected_direction"] for row in rows) >= 2
            and all(row["known_coverage"] >= 0.90 for row in rows)
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P1 cross-front-end robustness",
        "",
        "| front end | DAS-homoscedastic score (mm) | bootstrap 95% CI | coverage | width (mm) | failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        interval = row["bootstrap_95_ci_mm"]
        lines.append(
            f"| {row['frontend']} | {row['mean_das_minus_homoscedastic_score_mm']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {row['known_coverage']:.3f} | "
            f"{row['known_mean_width_mm']:.3f} | {row['failure_count']} |"
        )
    lines.extend([
        "",
        f"P1 gate: {'PASS' if report['p1_gate_passed'] else 'FAIL'}.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "negative_effect_frontends": report["negative_effect_frontends"],
        "ci_excluding_zero_frontends": report["ci_excluding_zero_frontends"],
        "p1_gate_passed": report["p1_gate_passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

