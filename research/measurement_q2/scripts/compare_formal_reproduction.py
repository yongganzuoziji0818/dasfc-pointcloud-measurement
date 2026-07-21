"""Compare deterministic scientific outputs from two DAS-FC formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--rerun", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonical_scientific_report(path: Path) -> dict:
    report = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    for row in report["rows"]:
        row.pop("duration_seconds", None)
    report["rows"].sort(key=lambda row: (row["case_id"], row["method"]))
    report["selection_rows"].sort(key=lambda row: row["case_id"])
    report["summaries"].sort(
        key=lambda row: (row["domain_status"], row["domain"], row["method"])
    )
    return report


def encoded(report: dict) -> bytes:
    return json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def main() -> int:
    args = parse_args()
    original = canonical_scientific_report(args.original)
    rerun = canonical_scientific_report(args.rerun)
    original_bytes = encoded(original)
    rerun_bytes = encoded(rerun)
    exact = original_bytes == rerun_bytes
    report = {
        "schema_version": "1.0",
        "status": "REPRODUCIBILITY_COMPARISON_COMPLETE",
        "determinism_class": "deterministic scientific metrics; wall-clock durations excluded",
        "original": str(args.original),
        "rerun": str(args.rerun),
        "canonical_sha256_original": hashlib.sha256(original_bytes).hexdigest(),
        "canonical_sha256_rerun": hashlib.sha256(rerun_bytes).hexdigest(),
        "scientific_outputs_exact_match": exact,
        "verdict": "REPRODUCIBLE" if exact else "NOT_REPRODUCIBLE",
        "excluded_fields": ["rows[].duration_seconds"],
        "included_scientific_content": [
            "configuration and seed bases",
            "scale-model diagnostics and selected blends",
            "all calibration quantiles and order statistics",
            "all 3360 method-level scientific result rows",
            "all 480 rejector selection rows",
            "domain summaries and AURC values",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())

