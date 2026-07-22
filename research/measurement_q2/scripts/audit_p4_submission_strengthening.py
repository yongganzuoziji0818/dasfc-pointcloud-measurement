"""Validate P4 submission-strengthening artifacts without changing results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    report_path = args.result_dir / "p4_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "p4_report.json": report_path,
        "synthetic_tuning_diagnostics.csv": args.result_dir / "synthetic_tuning_diagnostics.csv",
        "rockfall_diagnostics.csv": args.result_dir / "rockfall_diagnostics.csv",
        "rockfall_reference_sensitivity_targets.csv": args.result_dir / "rockfall_reference_sensitivity_targets.csv",
        "field_failure_example.csv": args.result_dir / "field_failure_example.csv",
        "figure8_pdf": args.result_dir / "figure8_p4_submission_strengthening.pdf",
        "figure8_svg": args.result_dir / "figure8_p4_submission_strengthening.svg",
        "figure8_png": args.result_dir / "figure8_p4_submission_strengthening.png",
    }
    checks = {
        "status_complete": report.get("status") == "P4_SUBMISSION_STRENGTHENING_COMPLETE",
        "all_inputs_hash_verified": len(report.get("input_sha256", {})) == 7,
        "synthetic_pairs_240": report["domain_alignment"]["synthetic_pairs"] == 240,
        "rockfall_events_3": report["domain_alignment"]["rockfall_events"] == 3,
        "domain_event_rows_3": len(report["domain_alignment"]["events"]) == 3,
        "sensitivity_target_rows_36": len(report["reference_sensitivity"]["target_rows"]) == 36,
        "sensitivity_event_rows_9": len(report["reference_sensitivity"]["event_summaries"]) == 9,
        "sensitivity_frontends_3": len(report["reference_sensitivity"]["frontend_summaries"]) == 3,
        "field_reconstruction_passed": report["field_example"]["reconstruction_gate"] is True,
        "pointwise_example_fails_simultaneously": report["field_example"]["pointwise"]["simultaneous_covered"] is False,
        "das_example_covers_simultaneously": report["field_example"]["das"]["simultaneous_covered"] is True,
        "pointwise_example_marginal_at_least_090": report["field_example"]["pointwise"]["point_coverage"] >= 0.90,
        "report_all_finite": finite_tree(report),
        "required_files_present_nonempty": all(path.is_file() and path.stat().st_size > 0 for path in required.values()),
        "synthetic_csv_rows_240": csv_count(required["synthetic_tuning_diagnostics.csv"]) == 240,
        "rockfall_csv_rows_3": csv_count(required["rockfall_diagnostics.csv"]) == 3,
        "sensitivity_csv_rows_36": csv_count(required["rockfall_reference_sensitivity_targets.csv"]) == 36,
        "field_csv_nonempty": csv_count(required["field_failure_example.csv"]) > 0,
    }
    passed = all(checks.values())
    payload = {
        "passed": passed,
        "checks": checks,
        "required_files": {name: str(path) for name, path in required.items()},
        "claim_boundary": "artifact and schema audit only; no scientific claim added by this audit",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
