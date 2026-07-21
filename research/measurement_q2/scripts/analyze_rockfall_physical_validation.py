"""Audit and visualize the frozen ETH Rockfall physical validation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FRONTENDS = ("cascade_strong", "multiscale_trimmed_ptp", "robust_ptpl")
EVENTS = ("E0->E1", "E1->E2", "E2->E3")
LABELS = {
    "cascade_strong": "Cascade-Strong",
    "multiscale_trimmed_ptp": "Trimmed PTP",
    "robust_ptpl": "Robust PTPL",
}
COLORS = {
    "cascade_strong": "#0072B2",
    "multiscale_trimmed_ptp": "#E69F00",
    "robust_ptpl": "#009E73",
}
MARKERS = {"cascade_strong": "o", "multiscale_trimmed_ptp": "s", "robust_ptpl": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d5-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in {key}")
    return value


def boolean(row: dict, key: str) -> bool:
    value = row[key].strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"invalid Boolean value in {key}: {row[key]}")
    return value == "true"


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n_events": int(array.size),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def validate(event_rows: list[dict], target_rows: list[dict], report: dict) -> list[str]:
    errors: list[str] = []
    expected_event_keys = {(event, frontend) for event in EVENTS for frontend in FRONTENDS}
    observed_event_keys = {(row["event"], row["frontend"]) for row in event_rows}
    if len(event_rows) != 9 or observed_event_keys != expected_event_keys:
        errors.append("event_table_not_complete_or_unique")
    expected_target_keys = {
        (event, frontend, target)
        for event in EVENTS for frontend in FRONTENDS for target in ("T1", "T2", "T3", "T4")
    }
    observed_target_keys = {(row["event"], row["frontend"], row["target"]) for row in target_rows}
    if len(target_rows) != 36 or observed_target_keys != expected_target_keys:
        errors.append("target_table_not_complete_or_unique")
    if report.get("dataset_revision") != "42a3947d960c8163157c915dea847cda96904a3d":
        errors.append("dataset_revision_mismatch")
    if report.get("calibration_or_tuning_on_rockfall") is not False:
        errors.append("unexpected_rockfall_tuning")
    boundary = report.get("claim_boundary", {})
    expected_boundary = {
        "independent_apparatus_count": 1,
        "normal_component_interval": True,
        "population_nominal_coverage": False,
        "calibrated_three_dimensional_vector_region": False,
        "three_dimensional_point_error": True,
    }
    for key, value in expected_boundary.items():
        if boundary.get(key) != value:
            errors.append(f"claim_boundary_mismatch:{key}")
    report_keys = {(row["event"], row["frontend"]) for row in report.get("events", [])}
    if report_keys != expected_event_keys:
        errors.append("json_event_keys_mismatch")
    for row in event_rows:
        target_subset = [
            item for item in target_rows
            if item["event"] == row["event"] and item["frontend"] == row["frontend"]
        ]
        covered = sum(boolean(item, "covered_normal") for item in target_subset)
        if covered != int(row["covered_targets"]):
            errors.append(f"covered_target_count_mismatch:{row['event']}:{row['frontend']}")
        if (covered == 4) != boolean(row, "simultaneous_covered_normal_four_targets"):
            errors.append(f"simultaneous_coverage_mismatch:{row['event']}:{row['frontend']}")
        if not boolean(row, "converged"):
            errors.append(f"nonconverged:{row['event']}:{row['frontend']}")
    return errors


def build_fallacy_scan() -> list[dict]:
    return [
        {"fallacy": "Simpson's paradox", "severity": "NOTE", "verdict": "Not detected; all three event rows and per-event directions are shown, with no pooled reversal claim."},
        {"fallacy": "Ecological fallacy", "severity": "CAUTION", "verdict": "Prevented by restricting inference to scan-pair events; four targets are nested checks, not independent events."},
        {"fallacy": "Berkson's paradox", "severity": "NOTE", "verdict": "No association claim is estimated from a jointly selected sample; the finite external sequence is described as such."},
        {"fallacy": "Collider bias", "severity": "NOTE", "verdict": "No adjusted association model or conditioned causal estimate is used."},
        {"fallacy": "Base-rate neglect", "severity": "NOTE", "verdict": "No diagnostic sensitivity, specificity, PPV, or NPV is reported."},
        {"fallacy": "Regression to the mean", "severity": "NOTE", "verdict": "Events were not selected by extreme prediction error and no pre-post improvement claim is made."},
        {"fallacy": "Survivorship bias", "severity": "NOTE", "verdict": "All 3 frozen consecutive events, 3 front ends, and 4 targets per event are retained; failures are not dropped."},
        {"fallacy": "Look-elsewhere effect", "severity": "NOTE", "verdict": "No significance search is performed; all predeclared event-level endpoints are reported."},
        {"fallacy": "Garden of forking paths", "severity": "CAUTION", "verdict": "No preregistration exists, but the D4/D5 protocol, parameters, hashes, and no-retuning rule were frozen before D5."},
        {"fallacy": "Correlation does not imply causation", "severity": "NOTE", "verdict": "No causal effect is inferred from front-end differences."},
        {"fallacy": "Reverse causality", "severity": "NOTE", "verdict": "Not applicable to the truth-referenced measurement comparison; no directional causal model is fitted."},
    ]


def create_figure(event_rows: list[dict], target_rows: list[dict], output_prefix: Path) -> dict:
    event_index = {(row["event"], row["frontend"]): row for row in event_rows}
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    x = np.arange(len(EVENTS))
    for frontend in FRONTENDS:
        kwargs = dict(color=COLORS[frontend], marker=MARKERS[frontend], linewidth=1.25, markersize=4, label=LABELS[frontend])
        axes[0, 0].plot(x, [numeric(event_index[(event, frontend)], "mean_vector_error_mm") for event in EVENTS], **kwargs)
        axes[0, 1].plot(x, [numeric(event_index[(event, frontend)], "normal_mae_mm") for event in EVENTS], **kwargs)
        axes[1, 0].plot(x, [numeric(event_index[(event, frontend)], "mean_interval_width_mm") for event in EVENTS], **kwargs)
        for event_pos, event in enumerate(EVENTS):
            row = event_index[(event, frontend)]
            if not boolean(row, "simultaneous_covered_normal_four_targets"):
                axes[1, 0].scatter(event_pos, numeric(row, "mean_interval_width_mm"), s=45, facecolors="white", edgecolors=COLORS[frontend], linewidths=1.2, zorder=5)
    for ax in axes.flat[:3]:
        ax.set_xticks(x, [item.replace("->", "–") for item in EVENTS])
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean 3D vector error (mm)")
    axes[0, 0].set_title("Sparse-target 3D error by physical event")
    axes[0, 1].set_ylabel("Normal-component MAE (mm)")
    axes[0, 1].set_title("Normal-component error")
    axes[1, 0].set_ylabel("Mean interval width (mm)")
    axes[1, 0].set_title("Interval width; open marker = event not covered")
    axes[0, 0].legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    limits = []
    for frontend in FRONTENDS:
        rows = [row for row in target_rows if row["frontend"] == frontend]
        reference = np.asarray([numeric(row, "reference_dy_normal_mm") for row in rows])
        predicted = np.asarray([numeric(row, "predicted_dy_normal_mm") for row in rows])
        limits.extend(reference.tolist() + predicted.tolist())
        ax.scatter(reference, predicted, s=22, alpha=0.8, color=COLORS[frontend], marker=MARKERS[frontend], label=LABELS[frontend])
    lower, upper = min(limits) - 2.0, max(limits) + 2.0
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(lower, upper); ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("TS60 reference normal displacement (mm)")
    ax.set_ylabel("TLS prediction (mm)")
    ax.set_title("All sparse targets (12 nested rows/front end)")
    ax.grid(linewidth=0.35, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    for label, ax_item in zip("ABCD", axes.flat):
        ax_item.text(-0.13, 1.05, label, transform=ax_item.transAxes, fontsize=10, fontweight="bold", va="top")
    fig.suptitle("ETH Rockfall Simulator: three consecutive events, four independent-reference targets", fontsize=9)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        path = output_prefix.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = str(path)
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    d5_run = args.d5_run.resolve()
    output_dir = args.output_dir.resolve()
    if not (d5_run / "D5_FORMAL.ok").exists() or (d5_run / "D5_FORMAL.failed").exists():
        raise RuntimeError("Refusing analysis without an unambiguous D5 pass sentinel")
    event_path = d5_run / "results" / "event_results.csv"
    target_path = d5_run / "results" / "target_results.csv"
    report_path = d5_run / "results" / "physical_validation_report.json"
    event_rows, target_rows = read_csv(event_path), read_csv(target_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    structural_errors = validate(event_rows, target_rows, report)
    if structural_errors:
        raise RuntimeError(";".join(structural_errors))

    by_frontend = defaultdict(list)
    for row in event_rows:
        by_frontend[row["frontend"]].append(row)
    metrics = ("mean_vector_error_mm", "max_vector_error_mm", "normal_mae_mm", "mean_interval_width_mm", "mean_interval_score_mm")
    frontend_summary = {}
    for frontend in FRONTENDS:
        rows = sorted(by_frontend[frontend], key=lambda row: EVENTS.index(row["event"]))
        frontend_summary[frontend] = {
            "simultaneously_covered_events": int(sum(boolean(row, "simultaneous_covered_normal_four_targets") for row in rows)),
            "event_count": 3,
            "metrics": {metric: describe([numeric(row, metric) for row in rows]) for metric in metrics},
        }

    fallacy_scan = build_fallacy_scan()
    outputs = create_figure(event_rows, target_rows, output_dir / "figure7_rockfall_physical_validation")
    analysis = {
        "schema_version": "1.0",
        "status": "ROCKFALL_D5_ANALYSIS_VALIDATED",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": report["dataset_revision"],
        "independent_unit": "scan-pair event",
        "event_count": 3,
        "nested_targets_per_event": 4,
        "independent_apparatus_count": 1,
        "structural_errors": structural_errors,
        "frontend_summary": frontend_summary,
        "statistical_inference": "No p-values, significance marks, or population confidence intervals; medians, ranges, and exact finite-event counts only.",
        "claim_boundary": "Sparse independently referenced physical validation of 3D point error and normal-component intervals for three consecutive events; not dense-field truth, a calibrated 3D vector region, population nominal coverage, independent-apparatus replication, or railway-field validation.",
        "fallacy_scan": {"coverage": "11/11", "overall_confidence": "CAUTION", "items": fallacy_scan},
        "figure_outputs": outputs,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (event_path, target_path, report_path)
        },
    }
    (output_dir / "rockfall_summary.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "frontend_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frontend", "covered_events", "event_count", *[f"{metric}_{stat}" for metric in metrics for stat in ("median", "minimum", "maximum")]])
        for frontend in FRONTENDS:
            row = frontend_summary[frontend]
            values = [frontend, row["simultaneously_covered_events"], row["event_count"]]
            for metric in metrics:
                values.extend(row["metrics"][metric][stat] for stat in ("median", "minimum", "maximum"))
            writer.writerow(values)

    lines = [
        "# Rockfall D5 statistical validation", "", "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: validate", f"- Origin Date: {datetime.now(timezone.utc).date().isoformat()}",
        "- Verification Status: VERIFIED", "- Version Label: rockfall_d5_validation_v1", "", "## Validation Report", "",
        "- **Source**: Rockfall D5 formal Attempt 02", "- **Overall Confidence**: CAUTION", "- **Independent unit**: scan-pair event (n=3)",
        "- **Nested observations**: four TS60 reference targets per event; not treated as independent replicates", "- **Inference**: no p-values or population confidence intervals", "",
        "### Event-level descriptive findings", "", "| Front end | Simultaneous events | Median mean 3D error [range], mm | Median normal MAE [range], mm | Median width [range], mm |", "|---|---:|---:|---:|---:|",
    ]
    for frontend in FRONTENDS:
        row = frontend_summary[frontend]
        def fmt(metric: str) -> str:
            item = row["metrics"][metric]
            return f"{item['median']:.3f} [{item['minimum']:.3f}, {item['maximum']:.3f}]"
        lines.append(f"| {LABELS[frontend]} | {row['simultaneously_covered_events']}/3 | {fmt('mean_vector_error_mm')} | {fmt('normal_mae_mm')} | {fmt('mean_interval_width_mm')} |")
    lines.extend(["", "### Fallacy Scan", "", "- **Coverage**: 11/11 fallacy types checked", "", "| Fallacy | Severity | Verdict |", "|---|---|---|"])
    lines.extend(f"| {item['fallacy']} | {item['severity']} | {item['verdict']} |" for item in fallacy_scan)
    lines.extend(["", "### Claim boundary", "", analysis["claim_boundary"], ""])
    (output_dir / "ROCKFALL_STATISTICAL_VALIDATION_20260721.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "event_count": 3, "fallacy_scan": "11/11", "frontend_summary": frontend_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
