"""Analyze the frozen SHREC'19 resolution-robustness benchmark descriptively."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def summary(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "maximum": float(finite.max()),
    }


def rotation_gap_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def cycle_defect(forward: dict, reverse: dict) -> tuple[float, float]:
    rotation_forward = np.asarray(forward["rotation"], dtype=np.float64)
    rotation_reverse = np.asarray(reverse["rotation"], dtype=np.float64)
    translation_forward = np.asarray(forward["translation"], dtype=np.float64)
    translation_reverse = np.asarray(reverse["translation"], dtype=np.float64)
    composed_rotation = rotation_reverse @ rotation_forward
    composed_translation = translation_forward @ rotation_reverse.T + translation_reverse
    return rotation_gap_degrees(composed_rotation, np.eye(3)), float(
        np.linalg.norm(composed_translation)
    )


def spearman(x: list[float], y: list[float]) -> dict:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if valid.sum() < 3 or np.ptp(x_array[valid]) == 0 or np.ptp(y_array[valid]) == 0:
        return {"n": int(valid.sum()), "rho": None, "p_value_not_reported": True}
    result = stats.spearmanr(x_array[valid], y_array[valid])
    return {
        "n": int(valid.sum()),
        "rho": float(result.statistic),
        "p_value_not_reported": True,
        "note": "Finite-benchmark descriptive association; no independent-unit inference.",
    }


def leave_one_scan_out(rows: list[dict], metric: str) -> dict:
    scan_ids = sorted({
        scan_id for row in rows for scan_id in (row["source_scan"], row["target_scan"])
    })
    medians = []
    for scan_id in scan_ids:
        retained = [
            row[metric] for row in rows
            if scan_id not in (row["source_scan"], row["target_scan"])
            and math.isfinite(row[metric])
        ]
        if retained:
            medians.append(float(np.median(retained)))
    return {
        "removed_scan_ids": len(medians),
        "median_min": float(min(medians)) if medians else None,
        "median_max": float(max(medians)) if medians else None,
        "interpretation": "Sensitivity range, not a confidence interval.",
    }


def build_pair_metrics(report: dict, frontends: list[str]) -> tuple[list[dict], list[dict]]:
    indexed = {}
    for row in report["rows"]:
        key = (row["pair_key"], row["frontend"], row["resolution"], row["direction"])
        indexed[key] = row
    pair_info = {
        row["pair_key"]: (row["test_set"], row["source_scan"], row["target_scan"])
        for row in report["rows"]
    }
    metrics = []
    incomplete = []
    for pair_key, (test_set, source_scan, target_scan) in sorted(pair_info.items()):
        for frontend in frontends:
            required = {
                (resolution, direction): indexed.get((pair_key, frontend, resolution, direction))
                for resolution in ("hires", "lores")
                for direction in ("forward", "reverse")
            }
            missing = [
                f"{resolution}:{direction}"
                for (resolution, direction), row in required.items()
                if row is None or row.get("status") != "completed"
            ]
            if missing:
                incomplete.append({
                    "pair_key": pair_key,
                    "frontend": frontend,
                    "missing_or_failed": missing,
                })
                continue
            resolution_fit_gaps = []
            rotation_gaps = []
            translation_gaps = []
            for direction in ("forward", "reverse"):
                hires = required[("hires", direction)]
                lores = required[("lores", direction)]
                resolution_fit_gaps.append(abs(
                    hires["fit"]["symmetric_mean"] - lores["fit"]["symmetric_mean"]
                ))
                rotation_gaps.append(rotation_gap_degrees(
                    np.asarray(hires["rotation"]), np.asarray(lores["rotation"])
                ))
                translation_gaps.append(float(np.linalg.norm(
                    np.asarray(hires["translation"]) - np.asarray(lores["translation"])
                )))
            cycle_rotation = []
            cycle_translation = []
            for resolution in ("hires", "lores"):
                rotation, translation = cycle_defect(
                    required[(resolution, "forward")], required[(resolution, "reverse")]
                )
                cycle_rotation.append(rotation)
                cycle_translation.append(translation)
            fit_values = [
                row["fit"]["symmetric_mean"] for row in required.values()
            ]
            diagnostic_values = [
                row["fit"]["source_to_target_p95"] for row in required.values()
            ]
            metrics.append({
                "pair_key": pair_key,
                "test_set": test_set,
                "source_scan": source_scan,
                "target_scan": target_scan,
                "frontend": frontend,
                "resolution_fit_gap": float(np.mean(resolution_fit_gaps)),
                "rotation_gap_degrees": float(np.mean(rotation_gaps)),
                "translation_gap": float(np.mean(translation_gaps)),
                "cycle_rotation_degrees": float(np.mean(cycle_rotation)),
                "cycle_translation": float(np.mean(cycle_translation)),
                "mean_symmetric_fit": float(np.mean(fit_values)),
                "mean_match_distance_p95": float(np.mean(diagnostic_values)),
            })
    return metrics, incomplete


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frontends = list(config["design"]["frontends"])
    pair_metrics, incomplete = build_pair_metrics(report, frontends)
    by_frontend = defaultdict(list)
    for row in pair_metrics:
        by_frontend[row["frontend"]].append(row)

    metric_names = (
        "resolution_fit_gap",
        "rotation_gap_degrees",
        "translation_gap",
        "cycle_rotation_degrees",
        "cycle_translation",
        "mean_symmetric_fit",
        "mean_match_distance_p95",
    )
    frontend_summaries = {}
    for frontend in frontends:
        rows = by_frontend[frontend]
        frontend_summaries[frontend] = {
            "complete_pairs": len(rows),
            "metrics": {
                metric: summary([row[metric] for row in rows]) for metric in metric_names
            },
            "by_test_set": {
                str(test_set): {
                    metric: summary([
                        row[metric] for row in rows if row["test_set"] == test_set
                    ]) for metric in metric_names
                }
                for test_set in range(4)
            },
            "diagnostic_associations": {
                "match_p95_vs_resolution_fit_gap": spearman(
                    [row["mean_match_distance_p95"] for row in rows],
                    [row["resolution_fit_gap"] for row in rows],
                ),
                "match_p95_vs_translation_gap": spearman(
                    [row["mean_match_distance_p95"] for row in rows],
                    [row["translation_gap"] for row in rows],
                ),
            },
            "leave_one_scan_out": {
                metric: leave_one_scan_out(rows, metric)
                for metric in ("resolution_fit_gap", "translation_gap", "mean_symmetric_fit")
            },
        }

    comparisons = {}
    cascade = {row["pair_key"]: row for row in by_frontend["cascade_strong"]}
    for comparator in ("multiscale_trimmed_ptp", "robust_ptpl"):
        other = {row["pair_key"]: row for row in by_frontend[comparator]}
        common = sorted(set(cascade) & set(other))
        comparison = {}
        for metric in (
            "resolution_fit_gap",
            "rotation_gap_degrees",
            "translation_gap",
            "cycle_rotation_degrees",
            "cycle_translation",
            "mean_symmetric_fit",
        ):
            deltas = [cascade[key][metric] - other[key][metric] for key in common]
            comparison[metric] = {
                "cascade_minus_comparator": summary(deltas),
                "cascade_lower_count": int(sum(value < 0 for value in deltas)),
                "ties": int(sum(value == 0 for value in deltas)),
                "pair_count": len(deltas),
            }
        comparisons[comparator] = comparison

    expected_pairs = 4 if report["mode"] == "smoke" else int(
        config["dataset"]["expected_unique_pairs"]
    )
    structural_errors = []
    if report["pair_count"] != expected_pairs:
        structural_errors.append(
            f"expected_{expected_pairs}_pairs_observed_{report['pair_count']}"
        )
    if report["attempted_rows"] != report["expected_rows"]:
        structural_errors.append("attempted_row_count_does_not_match_expected")

    analysis = {
        "schema_version": "1.0",
        "protocol_id": config["protocol_id"],
        "status": (
            "SHREC2019_SMOKE_ANALYSIS_COMPLETE" if report["mode"] == "smoke"
            else "SHREC2019_EXPLORATORY_ANALYSIS_COMPLETE"
        ),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": report["mode"],
        "pair_count": report["pair_count"],
        "attempted_rows": report["attempted_rows"],
        "completed_rows": report["completed_rows"],
        "failed_rows": report["failed_rows"],
        "complete_pair_frontend_rows": len(pair_metrics),
        "incomplete_pair_frontend_rows": incomplete,
        "structural_errors": structural_errors,
        "frontend_summaries": frontend_summaries,
        "cascade_comparisons": comparisons,
        "pair_metrics": pair_metrics,
        "identity_baseline_caveat": (
            "Published coordinates have identity transforms and therefore zero transform/cycle "
            "gaps by construction; use them only as an observable fit baseline."
        ),
        "claim_boundary": (
            "Descriptive finite-benchmark evidence for resolution robustness and failure "
            "diagnostics on unlabelled real meshes; no truth-referenced accuracy or coverage."
        ),
        "statistical_inference": "No p-values or population confidence intervals computed.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SHREC'19 resolution-robustness descriptive analysis",
        "",
        f"- Status: `{analysis['status']}`",
        f"- Pairs: {analysis['pair_count']}",
        f"- Attempted/completed/failed rows: {analysis['attempted_rows']}/"
        f"{analysis['completed_rows']}/{analysis['failed_rows']}",
        "- Inference: no p-values or population confidence intervals.",
        "",
        "| Front end | Median fit gap | Median translation gap | Median rotation gap (deg) | Median symmetric fit |",
        "|---|---:|---:|---:|---:|",
    ]
    for frontend in frontends:
        metrics = frontend_summaries[frontend]["metrics"]
        lines.append(
            f"| {frontend} | {metrics['resolution_fit_gap'].get('median', float('nan')):.6g} | "
            f"{metrics['translation_gap'].get('median', float('nan')):.6g} | "
            f"{metrics['rotation_gap_degrees'].get('median', float('nan')):.6g} | "
            f"{metrics['mean_symmetric_fit'].get('median', float('nan')):.6g} |"
        )
    lines.extend([
        "",
        "Published coordinates are an identity negative control and are not ranked on transform or cycle gaps.",
        "All quantities are dimensionless except rotation in degrees. These data contain no public correspondence or displacement truth.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": analysis["status"],
        "pair_count": analysis["pair_count"],
        "attempted_rows": analysis["attempted_rows"],
        "completed_rows": analysis["completed_rows"],
        "failed_rows": analysis["failed_rows"],
        "structural_errors": structural_errors,
        "incomplete_pair_frontend_rows": len(incomplete),
    }, ensure_ascii=False, indent=2))
    return 1 if structural_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
