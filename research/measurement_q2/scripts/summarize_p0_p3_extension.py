"""Audit and consolidate the frozen P0b-P3 extension evidence package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_FRONTENDS = {
    "cascade_strong",
    "multiscale_trimmed_ptp",
    "robust_ptpl",
}
EXPECTED_CALIBRATION_SIZES = {30, 60, 100, 150}


def load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if any(token in text for token in ("NaN", "Infinity", "-Infinity")):
        raise ValueError(f"non-finite JSON token in {path}")
    return json.loads(text)


def audit_p0b(report: dict) -> dict:
    if report.get("status") != "SHREC2020A_EXPLORATORY_FRONTEND_BENCHMARK_COMPLETE":
        raise ValueError("P0b status mismatch")
    rows = report.get("rows", [])
    combinations = {(row["pair_id"], row["frontend"]) for row in rows}
    frontends = {row["frontend"] for row in rows}
    if len(rows) != 44 or len(combinations) != 44 or len(frontends) != 4:
        raise ValueError("P0b does not contain 44 unique pair/front-end rows")
    if report.get("scan_pairs") != 11 or report.get("physical_object_clusters") != 1:
        raise ValueError("P0b cluster or pair count mismatch")
    if report.get("confirmatory_coverage_inference_allowed") is not False:
        raise ValueError("P0b claim boundary was not retained")
    if report.get("engineering_subset") is not False:
        raise ValueError("P0b report is an engineering subset")
    if min(int(row["valid_correspondences"]) for row in rows) < 300:
        raise ValueError("P0b has too few released marker correspondences")
    return {
        "status": "complete",
        "scan_pairs": 11,
        "physical_object_clusters": 1,
        "rows": 44,
        "frontends": sorted(frontends),
        "failure_count": int(sum(item["failure_count"] for item in report["aggregates"])),
        "confirmatory_gate_applicable": False,
        "claim_boundary": report["claim_boundary"],
    }


def audit_p1(report: dict) -> dict:
    if report.get("status") != "P1_FRONTEND_ROBUSTNESS_ANALYSIS_COMPLETE":
        raise ValueError("P1 status mismatch")
    rows = report.get("rows", [])
    observed = {row["frontend"] for row in rows}
    if len(rows) != 3 or observed != EXPECTED_FRONTENDS:
        raise ValueError(f"P1 front-end mismatch: {observed}")
    return {
        "status": "complete",
        "frontends": sorted(observed),
        "negative_effect_frontends": int(report["negative_effect_frontends"]),
        "ci_excluding_zero_frontends": int(report["ci_excluding_zero_frontends"]),
        "failure_count": int(sum(row["failure_count"] for row in rows)),
        "gate_passed": bool(report["p1_gate_passed"]),
        "rows": rows,
    }


def audit_p2(report: dict) -> dict:
    if report.get("status") != "P2_REPEATED_SPLIT_ANALYSIS_COMPLETE":
        raise ValueError("P2 status mismatch")
    records = report.get("records", [])
    summaries = report.get("summaries", [])
    sizes = {int(item["calibration_size_per_group"]) for item in summaries}
    repetitions = {int(item["repetitions"]) for item in summaries}
    if len(records) != 120 or sizes != EXPECTED_CALIBRATION_SIZES or repetitions != {30}:
        raise ValueError("P2 repeated-split evidence is incomplete")
    if len({item["run"] for item in records}) != 120:
        raise ValueError("P2 contains duplicate run identifiers")
    return {
        "status": "complete",
        "runs": 120,
        "calibration_sizes_per_group": sorted(sizes),
        "repetitions_per_size": 30,
        "failure_count": int(sum(item["failure_count"] for item in records)),
        "preferred_calibration_size_per_group": int(
            report["preferred_calibration_size_per_group"]
        ),
        "gate_passed": bool(report["p2_stability_gate_passed"]),
        "summaries": summaries,
        "independence_note": report["independence_note"],
    }


def audit_p3(report: dict) -> dict:
    if report.get("status") != "P3_CLASSICAL_BASELINE_ANALYSIS_COMPLETE":
        raise ValueError("P3 status mismatch")
    contrasts = report.get("contrasts", [])
    if len(contrasts) != 3 or any(item["n_scan_pairs"] != 240 for item in contrasts):
        raise ValueError("P3 contrast evidence is incomplete")
    das_rows = [
        item for item in report.get("aggregate_by_method", [])
        if item["method"] == "das_grouped"
    ]
    if len(das_rows) != 1:
        raise ValueError("P3 DAS aggregate is missing or duplicated")
    das = das_rows[0]
    return {
        "status": "complete",
        "scan_pairs": 240,
        "failure_count": int(das["failure_count"]),
        "das_coverage": float(das["simultaneous_coverage"]),
        "das_mean_width_mm": float(das["mean_interval_width_mm"]),
        "das_interval_score_mm": float(das["mean_interval_score_mm"]),
        "contrast_pass": dict(report["contrast_pass"]),
        "coverage_efficiency_pass": bool(report["coverage_efficiency_pass"]),
        "gate_passed": bool(report["p3_gate_passed"]),
        "contrasts": contrasts,
    }


def ensure_finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            ensure_finite(child)
    elif isinstance(value, list):
        for child in value:
            ensure_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite consolidated statistic: {value}")


def render_markdown(report: dict) -> str:
    gates = report["scientific_gates"]
    return "\n".join([
        "# P0-P3 extension evidence decision",
        "",
        f"Evidence package complete: **{'yes' if report['evidence_complete'] else 'no'}**.",
        f"Ready to begin evidence-bounded writing: **{'yes' if report['writing_can_begin'] else 'no'}**.",
        "",
        "| package | status | scientific gate | failures |",
        "|---|---|:---:|---:|",
        f"| P0b real-scan supplement | {report['p0b']['status']} | n/a | {report['p0b']['failure_count']} |",
        f"| P1 cross-front-end robustness | {report['p1']['status']} | {'PASS' if gates['p1'] else 'FAIL'} | {report['p1']['failure_count']} |",
        f"| P2 repeated-split stability | {report['p2']['status']} | {'PASS' if gates['p2'] else 'FAIL'} | {report['p2']['failure_count']} |",
        f"| P3 classical baselines | {report['p3']['status']} | {'PASS' if gates['p3'] else 'FAIL'} | {report['p3']['failure_count']} |",
        "",
        (
            "All target extension gates passed; the planned strengthened claim set may "
            "enter drafting subject to manuscript integrity review."
            if report["all_target_gates_passed"] else
            "At least one target extension gate failed. Writing may still begin, but the "
            "failed claim must be reported as a negative result or removed."
        ),
        "",
        "P0b remains exploratory one-object sparse-marker evidence and cannot be used as "
        "confirmatory coverage or real sound-barrier validation.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0b", required=True, type=Path)
    parser.add_argument("--p1", required=True, type=Path)
    parser.add_argument("--p2", required=True, type=Path)
    parser.add_argument("--p3", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    consolidated = {
        "schema_version": "1.0",
        "status": "P0_P3_EXTENSION_EVIDENCE_AUDIT_COMPLETE",
        "p0b": audit_p0b(load_json(args.p0b)),
        "p1": audit_p1(load_json(args.p1)),
        "p2": audit_p2(load_json(args.p2)),
        "p3": audit_p3(load_json(args.p3)),
    }
    consolidated["scientific_gates"] = {
        "p1": consolidated["p1"]["gate_passed"],
        "p2": consolidated["p2"]["gate_passed"],
        "p3": consolidated["p3"]["gate_passed"],
    }
    consolidated["evidence_complete"] = True
    consolidated["writing_can_begin"] = True
    consolidated["all_target_gates_passed"] = all(
        consolidated["scientific_gates"].values()
    )
    consolidated["writing_rule"] = (
        "Evidence completeness permits writing. Scientific gate failures do not block "
        "writing, but they require claim removal, limitation, or explicit negative-result "
        "reporting."
    )
    ensure_finite(consolidated)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(consolidated), encoding="utf-8")
    print(json.dumps({
        "status": consolidated["status"],
        "evidence_complete": consolidated["evidence_complete"],
        "all_target_gates_passed": consolidated["all_target_gates_passed"],
        "writing_can_begin": consolidated["writing_can_begin"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

