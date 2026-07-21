"""Compare SHREC'19 formal and reproduction scientific outputs exactly."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--reproduction-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonicalize_execution(payload: dict) -> dict:
    canonical = copy.deepcopy(payload)
    canonical.pop("captured_at_utc", None)
    canonical.pop("dataset_root", None)
    for row in canonical.get("rows", []):
        row.pop("duration_seconds", None)
    return canonical


def canonicalize_analysis(payload: dict) -> dict:
    canonical = copy.deepcopy(payload)
    canonical.pop("captured_at_utc", None)
    return canonical


def digest(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    comparisons = {}
    for name, canonicalizer in (
        ("execution_report.json", canonicalize_execution),
        ("analysis.json", canonicalize_analysis),
    ):
        original = json.loads((args.original_dir / name).read_text(encoding="utf-8"))
        reproduction = json.loads(
            (args.reproduction_dir / name).read_text(encoding="utf-8")
        )
        original_digest = digest(canonicalizer(original))
        reproduction_digest = digest(canonicalizer(reproduction))
        comparisons[name] = {
            "original_canonical_sha256": original_digest,
            "reproduction_canonical_sha256": reproduction_digest,
            "exact_scientific_match": original_digest == reproduction_digest,
        }
    reproducible = all(item["exact_scientific_match"] for item in comparisons.values())
    report = {
        "schema_version": "1.0",
        "status": "SHREC2019_REPRODUCIBLE" if reproducible else "SHREC2019_NOT_REPRODUCIBLE",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Exact canonical JSON match after removing timestamps, runtime durations, "
            "and resolved root paths; no scientific metric tolerance applied."
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if reproducible else 1


if __name__ == "__main__":
    raise SystemExit(main())
