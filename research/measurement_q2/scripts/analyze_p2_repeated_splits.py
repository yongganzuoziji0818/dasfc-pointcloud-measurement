"""Aggregate P2 repeated-split outcomes without treating repetitions as IID."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY = "das_grouped"
BASELINE = "homoscedastic_grouped"
KNOWN = "known_calibration_group"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--required-sign-repetitions", type=int, default=27)
    return parser.parse_args()


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite P2 statistic: {value}")
    return value


def method_summary(report: dict, method: str) -> dict:
    matches = [
        item for item in report["aggregate_by_method"]
        if item["domain_status"] == KNOWN and item["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one aggregate row for {method}, found {len(matches)}")
    return matches[0]


def summarize(values: np.ndarray) -> dict:
    return {
        "median": finite(np.median(values)),
        "q25": finite(np.quantile(values, 0.25)),
        "q75": finite(np.quantile(values, 0.75)),
        "minimum": finite(values.min()),
        "maximum": finite(values.max()),
        "stability_interval_2p5_97p5": [
            finite(np.quantile(values, 0.025)),
            finite(np.quantile(values, 0.975)),
        ],
    }


def main() -> int:
    args = parse_args()
    records = []
    for statistics_path in sorted(args.runs_dir.glob("*/formal_statistics.json")):
        report = json.loads(statistics_path.read_text(encoding="utf-8"))
        source_dir = statistics_path.parent
        source_report = json.loads((source_dir / "formal_report.json").read_text(encoding="utf-8"))
        metadata = source_report["config"]["p2_metadata"]
        primary = method_summary(report, PRIMARY)
        baseline = method_summary(report, BASELINE)
        contrast = next(
            item for item in report["paired_interval_score_contrasts"]
            if item["baseline"] == BASELINE
        )
        records.append({
            "run": source_dir.name,
            "repetition": int(metadata["repetition"]),
            "split_seed": int(metadata["split_seed"]),
            "calibration_size_per_group": int(
                metadata["calibration_trajectories_per_group"]
            ),
            "das_coverage": finite(primary["simultaneous_coverage"]),
            "das_width_mm": finite(primary["mean_interval_width_mm"]),
            "das_interval_score_mm": finite(primary["mean_interval_score_mm"]),
            "homoscedastic_coverage": finite(baseline["simultaneous_coverage"]),
            "homoscedastic_width_mm": finite(baseline["mean_interval_width_mm"]),
            "das_minus_homoscedastic_interval_score_mm": finite(
                contrast["mean_difference_mm"]
            ),
            "failure_count": int(primary["failure_count"]),
        })
    if not records:
        raise FileNotFoundError(f"no completed P2 runs below {args.runs_dir}")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["calibration_size_per_group"]].append(record)
    summaries = []
    for calibration_size, group in sorted(grouped.items()):
        if len({item["repetition"] for item in group}) != len(group):
            raise ValueError(f"duplicate repetition for calibration size {calibration_size}")
        effects = np.asarray([
            item["das_minus_homoscedastic_interval_score_mm"] for item in group
        ])
        coverages = np.asarray([item["das_coverage"] for item in group])
        widths = np.asarray([item["das_width_mm"] for item in group])
        effect_summary = summarize(effects)
        summaries.append({
            "calibration_size_per_group": calibration_size,
            "repetitions": len(group),
            "expected_negative_sign_count": int(np.sum(effects < 0.0)),
            "effect_mm": effect_summary,
            "coverage": summarize(coverages),
            "width_mm": summarize(widths),
            "stability_gate": (
                int(np.sum(effects < 0.0)) >= args.required_sign_repetitions
                and effect_summary["stability_interval_2p5_97p5"][1] < 0.0
            ),
            "failure_count_total": int(sum(item["failure_count"] for item in group)),
        })

    preferred_size = max(grouped)
    preferred = next(
        item for item in summaries
        if item["calibration_size_per_group"] == preferred_size
    )
    report = {
        "schema_version": "1.0",
        "status": "P2_REPEATED_SPLIT_ANALYSIS_COMPLETE",
        "independence_note": (
            "The 30 repetitions reuse one frozen trajectory pool and are sensitivity "
            "analyses, not independent experiments."
        ),
        "records": records,
        "summaries": summaries,
        "preferred_calibration_size_per_group": preferred_size,
        "p2_stability_gate_passed": bool(preferred["stability_gate"]),
        "required_expected_sign_repetitions": args.required_sign_repetitions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P2 repeated split and calibration-size sensitivity",
        "",
        report["independence_note"],
        "",
        "| calibration n/group | repetitions | negative effects | median effect (mm) | 2.5-97.5% stability interval | median coverage | gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in summaries:
        interval = item["effect_mm"]["stability_interval_2p5_97p5"]
        lines.append(
            f"| {item['calibration_size_per_group']} | {item['repetitions']} | "
            f"{item['expected_negative_sign_count']} | {item['effect_mm']['median']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{item['coverage']['median']:.3f} | "
            f"{'PASS' if item['stability_gate'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        f"Preferred-size P2 gate: {'PASS' if report['p2_stability_gate_passed'] else 'FAIL'}.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "run_count": len(records),
        "p2_stability_gate_passed": report["p2_stability_gate_passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
