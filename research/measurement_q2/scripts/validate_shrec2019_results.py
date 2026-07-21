"""Validate and compact the frozen SHREC'19 v1.1 descriptive results."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = (
    "resolution_fit_gap",
    "translation_gap",
    "rotation_gap_degrees",
    "cycle_translation",
    "cycle_rotation_degrees",
    "mean_symmetric_fit",
)
ESTIMATORS = ("cascade_strong", "multiscale_trimmed_ptp", "robust_ptpl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--reproduction", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--formal-ok", required=True, type=Path)
    parser.add_argument("--reproduction-ok", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def compact_summary(summary: dict) -> dict:
    return {
        key: summary[key]
        for key in ("n", "median", "q25", "q75", "maximum")
        if key in summary
    }


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def simpson_scan(rows: list[dict]) -> dict:
    indexed = {
        (row["pair_key"], row["frontend"]): row
        for row in rows
    }
    result = {}
    for comparator in ESTIMATORS[1:]:
        overall = []
        strata = {str(index): [] for index in range(4)}
        for (pair_key, frontend), cascade_row in indexed.items():
            if frontend != "cascade_strong":
                continue
            other = indexed.get((pair_key, comparator))
            if other is None:
                continue
            delta = (
                cascade_row["resolution_fit_gap"]
                - other["resolution_fit_gap"]
            )
            overall.append(delta)
            strata[str(cascade_row["test_set"])].append(delta)
        overall_median = median(overall)
        stratum_medians = {
            key: median(values) for key, values in strata.items() if values
        }
        signs = {
            int(np.sign(value)) for value in stratum_medians.values() if value != 0
        }
        reversal = bool(overall_median != 0 and signs and -int(np.sign(overall_median)) in signs)
        result[comparator] = {
            "overall_median_cascade_minus_comparator": overall_median,
            "test_set_medians": stratum_medians,
            "at_least_one_stratum_opposes_overall": reversal,
        }
    return result


def main() -> int:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    reproduction = json.loads(args.reproduction.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))

    structural_checks = {
        "formal_sentinel_present": args.formal_ok.is_file(),
        "reproduction_sentinel_present": args.reproduction_ok.is_file(),
        "pair_count_is_76": analysis.get("pair_count") == 76,
        "attempted_rows_is_1216": analysis.get("attempted_rows") == 1216,
        "completed_rows_is_1216": analysis.get("completed_rows") == 1216,
        "failed_rows_is_zero": analysis.get("failed_rows") == 0,
        "no_structural_errors": not analysis.get("structural_errors"),
        "no_incomplete_pair_frontend_rows": not analysis.get(
            "incomplete_pair_frontend_rows"
        ),
        "exact_reproduction_pass": reproduction.get("status")
        == "SHREC2019_REPRODUCIBLE",
        "protocol_matches": analysis.get("protocol_id") == config.get("protocol_id"),
    }

    frontends = config["design"]["frontends"]
    compact_frontends = {}
    for frontend in frontends:
        source = analysis["frontend_summaries"][frontend]
        compact_frontends[frontend] = {
            "complete_pairs": source["complete_pairs"],
            "metrics": {
                metric: compact_summary(source["metrics"][metric])
                for metric in METRICS
            },
            "by_test_set_resolution_fit_gap": {
                test_set: compact_summary(values["resolution_fit_gap"])
                for test_set, values in source["by_test_set"].items()
            },
            "diagnostic_associations": source["diagnostic_associations"],
            "leave_one_scan_out": source["leave_one_scan_out"],
        }

    comparisons = {}
    for comparator, values in analysis["cascade_comparisons"].items():
        comparisons[comparator] = {
            metric: {
                "cascade_minus_comparator": compact_summary(
                    values[metric]["cascade_minus_comparator"]
                ),
                "cascade_lower_count": values[metric]["cascade_lower_count"],
                "ties": values[metric]["ties"],
                "pair_count": values[metric]["pair_count"],
            }
            for metric in METRICS
        }

    simpson = simpson_scan(analysis["pair_metrics"])
    fallacy_scan = [
        {
            "id": 1,
            "name": "Simpson's paradox",
            "status": "CAUTION" if any(
                item["at_least_one_stratum_opposes_overall"]
                for item in simpson.values()
            ) else "CHECKED_NO_REVERSAL",
            "evidence": simpson,
            "action": "Report exact test-set summaries beside the aggregate result.",
        },
        {
            "id": 2,
            "name": "Ecological fallacy",
            "status": "CHECKED_BOUNDARY_ENFORCED",
            "evidence": "Pair rows are not described as independent specimens or individuals.",
        },
        {
            "id": 3,
            "name": "Berkson's paradox",
            "status": "CAUTION",
            "evidence": "SHREC'19 is a curated benchmark with a published pair graph, not a random deployment sample.",
            "action": "Do not generalize prevalence or population performance.",
        },
        {
            "id": 4,
            "name": "Collider bias",
            "status": "NOT_APPLICABLE",
            "evidence": "No adjusted causal model or conditioned covariate is used.",
        },
        {
            "id": 5,
            "name": "Base-rate neglect",
            "status": "NOT_APPLICABLE",
            "evidence": "No diagnostic classifier, sensitivity, specificity, PPV, or NPV is reported.",
        },
        {
            "id": 6,
            "name": "Regression to the mean",
            "status": "NOT_APPLICABLE",
            "evidence": "No extreme-score recruitment or intervention pre/post claim is made.",
        },
        {
            "id": 7,
            "name": "Survivorship bias",
            "status": "CHECKED_NO_ATTRITION",
            "evidence": "All 1,216 frozen rows completed; failures would have remained outcomes.",
        },
        {
            "id": 8,
            "name": "Look-elsewhere effect",
            "status": "CHECKED_DESCRIPTIVE_ONLY",
            "evidence": "All frozen endpoints are retained and no p-values are computed.",
        },
        {
            "id": 9,
            "name": "Garden of forking paths",
            "status": "NOTE",
            "evidence": "v1.1 changed only thread-control determinism after a preserved v1 exact-reproduction failure; algorithms, pairs, and endpoints were unchanged.",
        },
        {
            "id": 10,
            "name": "Correlation is not causation",
            "status": "CHECKED_BOUNDARY_ENFORCED",
            "evidence": "Spearman coefficients are labelled finite-benchmark diagnostic associations only.",
        },
        {
            "id": 11,
            "name": "Reverse causality",
            "status": "NOT_APPLICABLE",
            "evidence": "No directional causal or temporal claim is made.",
        },
    ]

    output = {
        "schema_version": "1.0",
        "protocol_id": config["protocol_id"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SHREC2019_RESULTS_VALIDATED"
        if all(structural_checks.values())
        else "SHREC2019_RESULTS_VALIDATION_FAILED",
        "structural_checks": structural_checks,
        "reproduction": {
            "status": reproduction.get("status"),
            "canonical_execution_sha256": reproduction.get(
                "canonical_execution_sha256"
            ),
            "canonical_analysis_sha256": reproduction.get(
                "canonical_analysis_sha256"
            ),
        },
        "frontend_summaries": compact_frontends,
        "cascade_comparisons": comparisons,
        "fallacy_scan": fallacy_scan,
        "fallacy_categories_checked": len(fallacy_scan),
        "confidence": "CAUTION",
        "allowed_claim": "Finite-benchmark descriptive evidence about cross-resolution registration robustness and failure diagnostics on unlabelled real meshes.",
        "prohibited_claims": config["interpretation"]["prohibited_claims"],
        "statistical_inference": "No p-values or population confidence intervals; medians, IQRs, counts, exact strata, and leave-one-scan-ID-out sensitivity only.",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SHREC'19 v1.1 results validation",
        "",
        f"- Status: `{output['status']}`",
        "- Statistical confidence: `CAUTION` (finite curated benchmark; physical independence undocumented).",
        f"- Fallacy coverage: {len(fallacy_scan)}/11 checked.",
        f"- Exact reproduction: `{reproduction.get('status')}`.",
        "- Inference: no p-values or population confidence intervals.",
        "",
        "## Primary descriptive endpoints",
        "",
        "| Front end | Fit gap, median [IQR] | Translation gap, median [IQR] | Rotation gap, median [IQR], deg | Symmetric fit, median [IQR] |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "published_coordinates": "Published coordinates (identity control)",
        "cascade_strong": "Cascade-Strong",
        "multiscale_trimmed_ptp": "Multiscale trimmed PTP",
        "robust_ptpl": "Robust PTPL",
    }
    for frontend in frontends:
        metrics = compact_frontends[frontend]["metrics"]
        def cell(name: str) -> str:
            item = metrics[name]
            return f"{item['median']:.6g} [{item['q25']:.6g}, {item['q75']:.6g}]"
        lines.append(
            f"| {labels[frontend]} | {cell('resolution_fit_gap')} | "
            f"{cell('translation_gap')} | {cell('rotation_gap_degrees')} | "
            f"{cell('mean_symmetric_fit')} |"
        )
    lines.extend([
        "",
        "The identity control has zero transform and cycle gaps by construction and is not ranked on those endpoints.",
        "",
        "## Fallacy scan",
        "",
    ])
    lines.extend(
        f"{item['id']}. **{item['name']}** — `{item['status']}`. {item['evidence'] if isinstance(item['evidence'], str) else 'Exact aggregate/stratum signs stored in validation.json.'}"
        for item in fallacy_scan
    )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        output["allowed_claim"],
        "These data do not validate displacement accuracy, correspondence accuracy, simultaneous coverage, millimetre accuracy, or railway deployment.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "checks_passed": sum(structural_checks.values()),
        "checks_total": len(structural_checks),
        "fallacy_categories_checked": len(fallacy_scan),
    }, indent=2))
    return 0 if output["status"] == "SHREC2019_RESULTS_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
