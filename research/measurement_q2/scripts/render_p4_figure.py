"""Render the P4 manuscript figure from immutable audited result files only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from research.measurement_q2.scripts.run_p4_submission_strengthening import make_figure


EXPECTED = {
    "p4_report.json": "fe4cf9953d78bb31c5c1deb76f196a07fa345b475deac85d5e520f2bfdb6e592",
    "field_failure_example.csv": "5603c7c94d0b93d3f2d5addb7aa7bedeb14f436178937e3dcebac9f37c89f457",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    for name, expected in EXPECTED.items():
        observed = digest(args.input_dir / name)
        if observed != expected:
            raise ValueError(f"frozen input hash mismatch for {name}: {observed}")

    report = json.loads((args.input_dir / "p4_report.json").read_text(encoding="utf-8"))
    with (args.input_dir / "field_failure_example.csv").open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in {"case_id", "domain"}:
                    parsed[key] = value
                elif value in {"True", "False"}:
                    parsed[key] = value == "True"
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    make_figure(
        args.output_dir,
        report["domain_alignment"],
        report["reference_sensitivity"],
        rows,
        {
            "dpi": args.dpi,
            "font_family": "DejaVu Sans",
            "colorblind_palette": ["#0072B2", "#E69F00", "#009E73", "#D55E00"],
        },
    )
    print(json.dumps({"status": "P4_FIGURE_RENDERED", "input_sha256": EXPECTED}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
