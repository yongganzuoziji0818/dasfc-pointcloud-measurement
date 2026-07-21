"""Specimen-clustered analysis of real mesh/scan front-end transfer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METHODS = (
    "published_coordinates",
    "multiscale_p2p_icp",
    "cascade_strong_unmodified",
)
SCORES = ("scale_mean_mm", "scale_q95_mm", "match_q95_mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def aurc(risk: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    prefix = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)
    coverage = np.arange(1, risk.size + 1) / risk.size
    return float(np.trapezoid(prefix, coverage))


def cluster_bootstrap_indices(
    specimen_ids: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
):
    clusters = sorted(set(specimen_ids))
    by_cluster = {cluster: np.flatnonzero(specimen_ids == cluster) for cluster in clusters}
    for _ in range(repetitions):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        yield np.concatenate([by_cluster[cluster] for cluster in sampled_clusters])


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    rows = raw["rows"]
    if len(rows) != 114 or len({row["specimen_id"] for row in rows}) != 38:
        raise ValueError("expected 38 specimen clusters and 114 device acquisitions")
    specimen = np.asarray([row["specimen_id"] for row in rows], dtype=object)
    scanner = np.asarray([row["scanner"] for row in rows], dtype=object)
    risk = {
        method: np.asarray(
            [row[method]["symmetric_mean_mm"] for row in rows], dtype=np.float64
        )
        for method in METHODS
    }
    scores = {
        key: np.asarray(
            [row["cascade_strong_unmodified"][key] for row in rows], dtype=np.float64
        )
        for key in SCORES
    }
    rng = np.random.default_rng(args.seed)
    method_summaries = {}
    for method in METHODS:
        values = risk[method]
        method_summaries[method] = {
            "n_specimen_clusters": 38,
            "n_device_acquisitions": 114,
            "mean_symmetric_distance_mm": float(values.mean()),
            "median_symmetric_distance_mm": float(np.median(values)),
            "p95_symmetric_distance_mm": float(np.quantile(values, 0.95)),
            "max_symmetric_distance_mm": float(values.max()),
        }

    contrasts = {}
    contrast_defs = {
        "cascade_minus_icp": (
            risk["cascade_strong_unmodified"] - risk["multiscale_p2p_icp"]
        ),
        "cascade_minus_published": (
            risk["cascade_strong_unmodified"] - risk["published_coordinates"]
        ),
        "icp_minus_published": risk["multiscale_p2p_icp"] - risk["published_coordinates"],
    }
    boot_indices = list(cluster_bootstrap_indices(specimen, args.bootstrap, rng))
    for name, difference in contrast_defs.items():
        boot = np.asarray([difference[index].mean() for index in boot_indices])
        interval = np.quantile(boot, (0.025, 0.975))
        contrasts[name] = {
            "difference_definition": "first method minus second; negative favors first",
            "mean_difference_mm": float(difference.mean()),
            "specimen_cluster_bootstrap_95_ci_mm": [
                float(interval[0]), float(interval[1])
            ],
        }

    device_summaries = defaultdict(dict)
    for device in sorted(set(scanner)):
        index = scanner == device
        for method in METHODS:
            values = risk[method][index]
            device_summaries[device][method] = {
                "mean_symmetric_distance_mm": float(values.mean()),
                "median_symmetric_distance_mm": float(np.median(values)),
                "max_symmetric_distance_mm": float(values.max()),
            }

    observed_aurc = {key: aurc(risk["cascade_strong_unmodified"], value) for key, value in scores.items()}
    bootstrap_aurc = {
        key: np.asarray([
            aurc(risk["cascade_strong_unmodified"][index], value[index])
            for index in boot_indices
        ])
        for key, value in scores.items()
    }
    score_results = {}
    for key in SCORES:
        rho = stats.spearmanr(scores[key], risk["cascade_strong_unmodified"])
        interval = np.quantile(bootstrap_aurc[key], (0.025, 0.975))
        score_results[key] = {
            "aurc": observed_aurc[key],
            "specimen_cluster_bootstrap_95_ci": [float(interval[0]), float(interval[1])],
            "spearman_rho": float(rho.statistic),
            "spearman_p_acquisition_level_descriptive_only": float(rho.pvalue),
        }
    scale_vs_match = bootstrap_aurc["scale_mean_mm"] - bootstrap_aurc["match_q95_mm"]
    scale_match_ci = np.quantile(scale_vs_match, (0.025, 0.975))

    report = {
        "schema_version": "1.0",
        "status": "REAL_TRANSFER_ANALYSIS_COMPLETE",
        "independent_cluster": "physical printed specimen",
        "specimen_clusters": 38,
        "device_acquisitions": 114,
        "method_summaries": method_summaries,
        "paired_method_contrasts": contrasts,
        "device_summaries": dict(device_summaries),
        "cascade_nonconvergence_count": sum(
            not bool(row["cascade_strong_unmodified"]["converged"]) for row in rows
        ),
        "selective_risk_on_cascade_surface_disagreement": {
            "scores": score_results,
            "scale_mean_minus_match_q95_aurc": {
                "observed": float(
                    observed_aurc["scale_mean_mm"] - observed_aurc["match_q95_mm"]
                ),
                "specimen_cluster_bootstrap_95_ci": [
                    float(scale_match_ci[0]), float(scale_match_ci[1])
                ],
            },
            "lower_aurc_is_better": True,
        },
        "exploratory_external_transfer": True,
        "claim_boundary": raw["claim_boundary"],
        "writing_use": (
            "scanner-domain and failure-detection evidence only; not deformation accuracy or conformal coverage"
        ),
        "analysis_parameters": {
            "seed": args.seed,
            "specimen_cluster_bootstrap_repetitions": args.bootstrap,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "method_summaries": method_summaries,
        "paired_method_contrasts": contrasts,
        "selective_risk": report["selective_risk_on_cascade_surface_disagreement"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
