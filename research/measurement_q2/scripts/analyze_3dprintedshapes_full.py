"""Cluster-aware analysis of the full 3DPrintedShapes mesh/scan audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


SCANNERS = (
    "2iPad_Scaniverse",
    "3FARO_Focus",
    "4Creaform_HandySCAN3D",
)
CONTRASTS = (
    ("4Creaform_HandySCAN3D", "3FARO_Focus"),
    ("4Creaform_HandySCAN3D", "2iPad_Scaniverse"),
    ("3FARO_Focus", "2iPad_Scaniverse"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, original in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[int(original)])
        adjusted[int(original)] = min(running, 1.0)
    return adjusted.tolist()


def bootstrap_mean_ci(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> list[float]:
    samples = rng.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    interval = np.quantile(samples, (0.025, 0.975))
    return [float(interval[0]), float(interval[1])]


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    rows = raw["rows"]
    indexed: dict[str, dict[str, float]] = {}
    for row in rows:
        indexed.setdefault(row["specimen_id"], {})[row["scanner"]] = float(
            row["symmetric_mean_mm"]
        )
    if len(indexed) != 38:
        raise ValueError(f"expected 38 specimens, got {len(indexed)}")
    if any(set(scanner_rows) != set(SCANNERS) for scanner_rows in indexed.values()):
        raise ValueError("every specimen must contain all three scanner domains")
    specimen_ids = sorted(indexed)
    matrix = np.asarray(
        [[indexed[specimen][scanner] for scanner in SCANNERS] for specimen in specimen_ids],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite real-data metric")

    rng = np.random.default_rng(args.seed)
    summaries = {}
    for column, scanner in enumerate(SCANNERS):
        values = matrix[:, column]
        summaries[scanner] = {
            "n_specimen_clusters": int(values.size),
            "mean_symmetric_distance_mm": float(values.mean()),
            "sd_symmetric_distance_mm": float(values.std(ddof=1)),
            "median_symmetric_distance_mm": float(np.median(values)),
            "iqr_symmetric_distance_mm": [
                float(np.quantile(values, 0.25)),
                float(np.quantile(values, 0.75)),
            ],
            "range_symmetric_distance_mm": [float(values.min()), float(values.max())],
            "cluster_bootstrap_mean_95_ci_mm": bootstrap_mean_ci(
                values, args.bootstrap, rng
            ),
        }

    contrasts = []
    raw_p = []
    scanner_index = {scanner: index for index, scanner in enumerate(SCANNERS)}
    for first, second in CONTRASTS:
        difference = matrix[:, scanner_index[first]] - matrix[:, scanner_index[second]]
        paired_t = stats.ttest_1samp(difference, 0.0)
        signed_rank = stats.wilcoxon(difference, zero_method="wilcox")
        item = {
            "first": first,
            "second": second,
            "difference_definition": "first minus second; negative means lower mesh/scan disagreement for first",
            "n_specimen_clusters": int(difference.size),
            "mean_difference_mm": float(difference.mean()),
            "cluster_bootstrap_mean_95_ci_mm": bootstrap_mean_ci(
                difference, args.bootstrap, rng
            ),
            "paired_t": {
                "t": float(paired_t.statistic),
                "df": int(difference.size - 1),
                "p_two_sided": float(paired_t.pvalue),
            },
            "wilcoxon_signed_rank": {
                "statistic": float(signed_rank.statistic),
                "p_two_sided": float(signed_rank.pvalue),
            },
        }
        contrasts.append(item)
        raw_p.append(item["paired_t"]["p_two_sided"])
    for item, adjusted in zip(contrasts, holm(raw_p), strict=True):
        item["paired_t"]["p_holm_three_contrasts"] = float(adjusted)

    friedman = stats.friedmanchisquare(*(matrix[:, column] for column in range(3)))
    strict_gradient = (
        matrix[:, scanner_index["4Creaform_HandySCAN3D"]]
        < matrix[:, scanner_index["3FARO_Focus"]]
    ) & (
        matrix[:, scanner_index["3FARO_Focus"]]
        < matrix[:, scanner_index["2iPad_Scaniverse"]]
    )
    report = {
        "schema_version": "1.0",
        "status": "FULL_REAL_DATA_AUDIT_ANALYZED",
        "independent_unit": "physical printed specimen cluster",
        "specimen_clusters": len(specimen_ids),
        "device_acquisitions": len(rows),
        "raw_processed_independence_warning": (
            "Raw and Processed are representations of the same acquisition and are not independent"
        ),
        "registration": raw["registration"],
        "scanner_summaries": summaries,
        "paired_scanner_contrasts": contrasts,
        "friedman_repeated_measures": {
            "chi_square": float(friedman.statistic),
            "df": 2,
            "p": float(friedman.pvalue),
            "kendall_w": float(friedman.statistic / (len(specimen_ids) * 2)),
        },
        "strict_creaform_better_than_faro_better_than_ipad": {
            "specimens": int(strict_gradient.sum()),
            "total": len(specimen_ids),
            "fraction": float(strict_gradient.mean()),
        },
        "evidence_boundary": {
            "supports": [
                "real three-device surface disagreement and domain shift",
                "specimen-clustered repeatability comparisons",
                "external motivation for domain-aware uncertainty and abstention",
            ],
            "does_not_support": [
                "double-epoch displacement ground truth",
                "DAS-FC displacement accuracy on railway sound barriers",
                "independence of the three device scans from the same specimen",
            ],
            "algorithm_performance_result": False,
        },
        "analysis_parameters": {
            "seed": args.seed,
            "cluster_bootstrap_repetitions": args.bootstrap,
            "multiple_comparison_correction": "Holm across three scanner contrasts",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Full 3DPrintedShapes real-data audit",
        "",
        f"All {len(specimen_ids)} physical specimen clusters and {len(rows)} device acquisitions were analyzed without hidden registration.",
        "",
    ]
    for scanner in SCANNERS:
        summary = summaries[scanner]
        ci = summary["cluster_bootstrap_mean_95_ci_mm"]
        lines.append(
            f"- {scanner}: mean symmetric distance {summary['mean_symmetric_distance_mm']:.3f} mm "
            f"(cluster bootstrap 95% CI {ci[0]:.3f} to {ci[1]:.3f})."
        )
    lines.extend([
        "",
        f"The strict Creaform < FARO < iPad gradient occurred in {int(strict_gradient.sum())}/{len(specimen_ids)} specimens.",
        "",
        "This is a real-data scanner-domain audit, not a double-epoch displacement benchmark or a direct DAS-FC performance estimate.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "scanner_summaries": summaries,
        "gradient": report["strict_creaform_better_than_faro_better_than_ipad"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
