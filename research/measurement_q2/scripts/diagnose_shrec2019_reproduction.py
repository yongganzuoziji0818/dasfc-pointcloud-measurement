"""Diagnose exact-reproduction failure without rerunning SHREC'19 algorithms."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--reproduction-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        entries[Path(filename).name] = digest
    return entries


def numeric_differences(first, second, prefix: str = "") -> list[tuple[str, float]]:
    differences = []
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) & set(second)):
            if key in {"captured_at_utc", "duration_seconds", "dataset_root"}:
                continue
            child = f"{prefix}.{key}" if prefix else key
            differences.extend(numeric_differences(first[key], second[key], child))
    elif isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        for index, (left, right) in enumerate(zip(first, second)):
            differences.extend(numeric_differences(left, right, f"{prefix}[{index}]"))
    elif (
        isinstance(first, (int, float)) and not isinstance(first, bool)
        and isinstance(second, (int, float)) and not isinstance(second, bool)
    ):
        left = float(first)
        right = float(second)
        if math.isfinite(left) and math.isfinite(right):
            differences.append((prefix, abs(left - right)))
    return differences


def row_key(row: dict) -> tuple:
    return (
        row["pair_key"], row["frontend"], row["resolution"], row["direction"]
    )


def main() -> int:
    args = parse_args()
    original = json.loads(
        (args.original_dir / "execution_report.json").read_text(encoding="utf-8")
    )
    reproduction = json.loads(
        (args.reproduction_dir / "execution_report.json").read_text(encoding="utf-8")
    )
    original_rows = {row_key(row): row for row in original["rows"]}
    reproduction_rows = {row_key(row): row for row in reproduction["rows"]}
    keys_match = set(original_rows) == set(reproduction_rows)
    per_frontend = defaultdict(lambda: {
        "row_count": 0,
        "exact_after_ignoring_duration": 0,
        "numeric_value_count": 0,
        "nonzero_numeric_differences": 0,
        "maximum_absolute_difference": 0.0,
        "paths_at_maximum": [],
    })
    global_maximum = 0.0
    global_examples = []
    if keys_match:
        for key in sorted(original_rows):
            first = original_rows[key]
            second = reproduction_rows[key]
            frontend = key[1]
            bucket = per_frontend[frontend]
            bucket["row_count"] += 1
            comparable_first = {k: v for k, v in first.items() if k != "duration_seconds"}
            comparable_second = {k: v for k, v in second.items() if k != "duration_seconds"}
            if comparable_first == comparable_second:
                bucket["exact_after_ignoring_duration"] += 1
            diffs = numeric_differences(comparable_first, comparable_second)
            bucket["numeric_value_count"] += len(diffs)
            bucket["nonzero_numeric_differences"] += sum(value != 0.0 for _, value in diffs)
            if diffs:
                row_max = max(value for _, value in diffs)
                if row_max > bucket["maximum_absolute_difference"]:
                    bucket["maximum_absolute_difference"] = row_max
                    bucket["paths_at_maximum"] = [
                        {"row_key": list(key), "path": path, "difference": value}
                        for path, value in diffs if value == row_max
                    ][:5]
                if row_max > global_maximum:
                    global_maximum = row_max
                    global_examples = [
                        {"row_key": list(key), "path": path, "difference": value}
                        for path, value in diffs if value == row_max
                    ][:10]

    original_analysis = json.loads(
        (args.original_dir / "analysis.json").read_text(encoding="utf-8")
    )
    reproduction_analysis = json.loads(
        (args.reproduction_dir / "analysis.json").read_text(encoding="utf-8")
    )
    analysis_differences = numeric_differences(original_analysis, reproduction_analysis)
    shared_input_original = manifest(args.original_dir / "source_input_manifest.sha256")
    shared_input_reproduction = manifest(
        args.reproduction_dir / "source_input_manifest.sha256"
    )
    common_input_names = sorted(set(shared_input_original) & set(shared_input_reproduction))
    input_mismatches = {
        name: {
            "original": shared_input_original[name],
            "reproduction": shared_input_reproduction[name],
        }
        for name in common_input_names
        if shared_input_original[name] != shared_input_reproduction[name]
    }
    report = {
        "schema_version": "1.0",
        "status": "SHREC2019_REPRODUCTION_DIAGNOSIS_COMPLETE",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_keys_match": keys_match,
        "original_rows": len(original_rows),
        "reproduction_rows": len(reproduction_rows),
        "shared_input_hash_mismatches": input_mismatches,
        "per_frontend": dict(per_frontend),
        "global_maximum_absolute_numeric_difference": global_maximum,
        "global_maximum_examples": global_examples,
        "analysis_numeric_value_count": len(analysis_differences),
        "analysis_nonzero_numeric_differences": sum(
            value != 0.0 for _, value in analysis_differences
        ),
        "analysis_maximum_absolute_difference": max(
            (value for _, value in analysis_differences), default=0.0
        ),
        "diagnostic_boundary": (
            "This quantifies drift only. It does not change the preregistered exact-match "
            "failure into a pass and does not authorize manuscript integration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    structural_errors = not keys_match or bool(input_mismatches)
    return 1 if structural_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
