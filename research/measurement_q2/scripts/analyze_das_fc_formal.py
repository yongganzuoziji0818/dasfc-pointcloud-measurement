"""Analyze the frozen DAS-FC experiment at the scan-pair level.

The script never treats field locations as independent observations.  It keeps
the known-domain confirmatory contrasts separate from unseen-domain empirical
stress tests and applies Holm correction to the three predeclared interval-score
comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


KNOWN_STATUS = "known_calibration_group"
UNSEEN_STATUS = "unseen_empirical"
PRIMARY_METHOD = "das_grouped"
CONTRASTS = (
    ("homoscedastic_grouped", "primary"),
    ("raw_local_grouped", "secondary"),
    ("learned_grouped", "secondary"),
)
SCORE_KEYS = ("score_scale", "score_residual", "score_ood", "score_combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=50_000)
    parser.add_argument("--aurc-bootstrap", type=int, default=2_000)
    return parser.parse_args()


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite statistic: {value}")
    return value


def quantile_ci(values: np.ndarray, level: float = 0.95) -> list[float]:
    tail = (1.0 - level) / 2.0
    return [
        finite_float(np.quantile(values, tail)),
        finite_float(np.quantile(values, 1.0 - tail)),
    ]


def stratified_bootstrap_means(
    differences: np.ndarray,
    domains: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    domain_indices = [np.flatnonzero(domains == domain) for domain in sorted(set(domains))]
    output = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        parts = [
            differences[rng.choice(indices, size=indices.size, replace=True)]
            for indices in domain_indices
        ]
        output[index] = np.concatenate(parts).mean()
    return output


def sign_flip_pvalue(
    differences: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))
    extreme = 0
    completed = 0
    while completed < repetitions:
        batch = min(2_000, repetitions - completed)
        signs = rng.choice((-1.0, 1.0), size=(batch, differences.size))
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += batch
    return float((extreme + 1) / (repetitions + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = (total - rank) * p_values[int(original_index)]
        running = max(running, candidate)
        adjusted[int(original_index)] = min(running, 1.0)
    return adjusted.tolist()


def wilson_interval(successes: int, total: int, level: float = 0.95) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [finite_float(centre - half), finite_float(centre + half)]


def aurc(risk: np.ndarray, reject_score: np.ndarray) -> float:
    order = np.argsort(reject_score, kind="stable")
    prefix_risk = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)
    coverage = np.arange(1, risk.size + 1) / risk.size
    return finite_float(np.trapezoid(prefix_risk, coverage))


def method_rows_by_case(rows: list[dict]) -> dict[str, dict[str, dict]]:
    indexed: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        case_id = row["case_id"]
        method = row["method"]
        if method in indexed[case_id]:
            raise ValueError(f"duplicate case/method row: {case_id}/{method}")
        indexed[case_id][method] = row
    return dict(indexed)


def paired_contrast(
    indexed: dict[str, dict[str, dict]],
    baseline: str,
    rng: np.random.Generator,
    bootstrap: int,
    permutations: int,
) -> tuple[dict, np.ndarray]:
    case_ids = sorted(
        case_id
        for case_id, methods in indexed.items()
        if methods[PRIMARY_METHOD]["domain_status"] == KNOWN_STATUS
    )
    differences = np.asarray(
        [
            indexed[case_id][PRIMARY_METHOD]["interval_score_mm"]
            - indexed[case_id][baseline]["interval_score_mm"]
            for case_id in case_ids
        ],
        dtype=np.float64,
    )
    domains = np.asarray(
        [indexed[case_id][PRIMARY_METHOD]["domain"] for case_id in case_ids],
        dtype=object,
    )
    all_zero = bool(np.allclose(differences, 0.0, rtol=0.0, atol=1e-12))
    if all_zero:
        t_statistic, t_pvalue = 0.0, 1.0
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
        skewness = 0.0
        shapiro_statistic, shapiro_pvalue = 1.0, 1.0
        qq_correlation = 1.0
    else:
        t_result = stats.ttest_1samp(
            differences, popmean=0.0, alternative="two-sided"
        )
        wilcoxon = stats.wilcoxon(
            differences, zero_method="wilcox", alternative="two-sided"
        )
        qq = stats.probplot(differences, dist="norm", fit=True)
        shapiro = stats.shapiro(differences)
        t_statistic = finite_float(t_result.statistic)
        t_pvalue = finite_float(t_result.pvalue)
        wilcoxon_statistic = finite_float(wilcoxon.statistic)
        wilcoxon_pvalue = finite_float(wilcoxon.pvalue)
        skewness = finite_float(stats.skew(differences, bias=False))
        shapiro_statistic = finite_float(shapiro.statistic)
        shapiro_pvalue = finite_float(shapiro.pvalue)
        qq_correlation = finite_float(qq[1][2])
    boot = stratified_bootstrap_means(differences, domains, bootstrap, rng)
    sd = finite_float(differences.std(ddof=1))
    dz = finite_float(differences.mean() / sd) if sd > 0 else 0.0
    by_domain = {}
    for domain in sorted(set(domains)):
        values = differences[domains == domain]
        by_domain[str(domain)] = {
            "n_scan_pairs": int(values.size),
            "mean_difference_mm": finite_float(values.mean()),
            "median_difference_mm": finite_float(np.median(values)),
            "sd_difference_mm": finite_float(values.std(ddof=1)),
        }
    result = {
        "baseline": baseline,
        "difference_definition": "DAS-FC interval score minus baseline; negative favors DAS-FC",
        "n_scan_pairs": int(differences.size),
        "mean_difference_mm": finite_float(differences.mean()),
        "median_difference_mm": finite_float(np.median(differences)),
        "sd_difference_mm": sd,
        "stratified_bootstrap_95_ci_mm": quantile_ci(boot),
        "paired_cohen_dz": dz,
        "paired_t": {
            "t": t_statistic,
            "df": int(differences.size - 1),
            "p_two_sided": t_pvalue,
        },
        "paired_sign_flip": {
            "repetitions": permutations,
            "p_two_sided": sign_flip_pvalue(differences, permutations, rng),
        },
        "wilcoxon_signed_rank": {
            "statistic": wilcoxon_statistic,
            "p_two_sided": wilcoxon_pvalue,
        },
        "difference_diagnostics": {
            "skewness": skewness,
            "shapiro_w": shapiro_statistic,
            "shapiro_p": shapiro_pvalue,
            "qq_correlation": qq_correlation,
            "all_differences_zero": all_zero,
            "interpretation": (
                "All paired differences are numerically zero; neutral finite test "
                "statistics are reported instead of undefined zero-variance values."
                if all_zero else
                "Shapiro-Wilk is secondary; inference is triangulated with a "
                "domain-stratified bootstrap and paired sign-flip test."
            ),
        },
        "by_domain": by_domain,
    }
    return result, differences


def coverage_table(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["domain_status"], row["domain"], row["method"])].append(row)
    output = []
    for (status, domain, method), group in sorted(buckets.items()):
        successes = sum(bool(row["simultaneous_covered"]) for row in group)
        output.append({
            "domain_status": status,
            "domain": domain,
            "method": method,
            "n_scan_pairs": len(group),
            "simultaneous_successes": successes,
            "simultaneous_coverage": finite_float(successes / len(group)),
            "simultaneous_coverage_wilson_95_ci": wilson_interval(successes, len(group)),
            "mean_point_coverage": finite_float(np.mean([row["point_coverage"] for row in group])),
            "mean_interval_width_mm": finite_float(
                np.mean([row["mean_interval_width_mm"] for row in group])
            ),
            "mean_interval_score_mm": finite_float(
                np.mean([row["interval_score_mm"] for row in group])
            ),
            "failure_count": sum(not bool(row["converged"]) for row in group),
        })
    return output


def aggregate_method_table(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["domain_status"], row["method"])].append(row)
    output = []
    for (status, method), group in sorted(buckets.items()):
        successes = sum(bool(row["simultaneous_covered"]) for row in group)
        per_domain = defaultdict(list)
        for row in group:
            per_domain[row["domain"]].append(row)
        domain_scores = {
            domain: float(np.mean([row["interval_score_mm"] for row in values]))
            for domain, values in per_domain.items()
        }
        output.append({
            "domain_status": status,
            "method": method,
            "n_scan_pairs": len(group),
            "simultaneous_coverage": finite_float(successes / len(group)),
            "simultaneous_coverage_wilson_95_ci": wilson_interval(successes, len(group)),
            "mean_point_coverage": finite_float(np.mean([row["point_coverage"] for row in group])),
            "mean_interval_width_mm": finite_float(
                np.mean([row["mean_interval_width_mm"] for row in group])
            ),
            "mean_interval_score_mm": finite_float(
                np.mean([row["interval_score_mm"] for row in group])
            ),
            "worst_domain_interval_score_mm": finite_float(max(domain_scores.values())),
            "worst_domain": max(domain_scores, key=domain_scores.get),
            "failure_count": sum(not bool(row["converged"]) for row in group),
        })
    return output


def rejector_analysis(
    selection_rows: list[dict],
    rng: np.random.Generator,
    repetitions: int,
) -> dict:
    risk = np.asarray([row["risk_normal_mae_mm"] for row in selection_rows], dtype=np.float64)
    domains = np.asarray([row["domain"] for row in selection_rows], dtype=object)
    scores = {
        key: np.asarray([row[key] for row in selection_rows], dtype=np.float64)
        for key in SCORE_KEYS
    }
    observed = {key: aurc(risk, values) for key, values in scores.items()}
    domain_indices = [np.flatnonzero(domains == domain) for domain in sorted(set(domains))]
    boot = {key: np.empty(repetitions, dtype=np.float64) for key in SCORE_KEYS}
    for repetition in range(repetitions):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True) for indices in domain_indices
        ])
        for key in SCORE_KEYS:
            boot[key][repetition] = aurc(risk[sampled], scores[key][sampled])
    comparisons = {}
    for key in SCORE_KEYS[:-1]:
        difference = boot["score_combined"] - boot[key]
        comparisons[key] = {
            "observed_combined_minus_single": finite_float(observed["score_combined"] - observed[key]),
            "stratified_bootstrap_95_ci": quantile_ci(difference),
            "bootstrap_probability_combined_is_lower": finite_float(np.mean(difference < 0.0)),
        }
    retained = [row for row in selection_rows if not row["rejected"]]
    rejected = [row for row in selection_rows if row["rejected"]]
    return {
        "aurc": observed,
        "lower_is_better": True,
        "combined_vs_single": comparisons,
        "frozen_hypothesis_passed": all(
            observed["score_combined"] < observed[key] for key in SCORE_KEYS[:-1]
        ),
        "frozen_threshold_outcome": {
            "retained": len(retained),
            "rejected": len(rejected),
            "retained_mean_risk_mm": finite_float(
                np.mean([row["risk_normal_mae_mm"] for row in retained])
            ),
            "rejected_mean_risk_mm": finite_float(
                np.mean([row["risk_normal_mae_mm"] for row in rejected])
            ),
        },
    }


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    rows = raw["rows"]
    selection_rows = raw["selection_rows"]
    indexed = method_rows_by_case(rows)
    methods = sorted({row["method"] for row in rows})
    if len(rows) != len(indexed) * len(methods):
        raise ValueError("incomplete rectangular case-by-method result table")
    if any(set(per_method) != set(methods) for per_method in indexed.values()):
        raise ValueError("at least one case is missing a method")

    rng = np.random.default_rng(args.seed)
    contrasts = []
    raw_p = []
    for baseline, role in CONTRASTS:
        result, _ = paired_contrast(
            indexed, baseline, rng, args.bootstrap, args.permutations
        )
        result["role"] = role
        contrasts.append(result)
        raw_p.append(result["paired_t"]["p_two_sided"])
    adjusted = holm_adjust(raw_p)
    for result, p_adjusted in zip(contrasts, adjusted, strict=True):
        result["paired_t"]["p_holm_three_contrasts"] = finite_float(p_adjusted)

    coverage = coverage_table(rows)
    aggregate = aggregate_method_table(rows)
    rejector = rejector_analysis(selection_rows, rng, args.aurc_bootstrap)
    case_durations = np.asarray([
        methods_by_case[PRIMARY_METHOD]["duration_seconds"]
        for methods_by_case in indexed.values()
    ], dtype=np.float64)

    primary = contrasts[0]
    primary_pass = (
        primary["stratified_bootstrap_95_ci_mm"][1] < 0.0
        and primary["paired_t"]["p_holm_three_contrasts"] < 0.05
    )
    known_aggregate = {
        item["method"]: item for item in aggregate if item["domain_status"] == KNOWN_STATUS
    }
    known_domains = {
        item["domain"]: item
        for item in coverage
        if item["domain_status"] == KNOWN_STATUS and item["method"] == PRIMARY_METHOD
    }
    homoscedastic_domains = {
        item["domain"]: item
        for item in coverage
        if item["domain_status"] == KNOWN_STATUS
        and item["method"] == "homoscedastic_grouped"
    }
    domain_advantages = {
        domain: known_domains[domain]["mean_interval_score_mm"]
        - homoscedastic_domains[domain]["mean_interval_score_mm"]
        for domain in known_domains
    }
    calibration_efficiency_pass = (
        known_aggregate[PRIMARY_METHOD]["simultaneous_coverage"] >= 0.90
        and known_aggregate[PRIMARY_METHOD]["mean_interval_width_mm"]
        < known_aggregate["homoscedastic_grouped"]["mean_interval_width_mm"]
    )
    robustness_pass = (
        sum(value < 0.0 for value in domain_advantages.values()) >= 3
        and max(domain_advantages.values()) < 0.0
    )

    report = {
        "schema_version": "1.0",
        "status": "FORMAL_STATISTICAL_ANALYSIS_COMPLETE",
        "source_experiment_status": raw["status"],
        "independent_unit": "complete synthetic scan pair",
        "confirmatory_scope": "known calibration groups only",
        "unseen_scope": "empirical stress tests without formal coverage guarantee",
        "integrity": {
            "case_count": len(indexed),
            "known_case_count": sum(
                methods_by_case[PRIMARY_METHOD]["domain_status"] == KNOWN_STATUS
                for methods_by_case in indexed.values()
            ),
            "unseen_case_count": sum(
                methods_by_case[PRIMARY_METHOD]["domain_status"] == UNSEEN_STATUS
                for methods_by_case in indexed.values()
            ),
            "method_count": len(methods),
            "row_count": len(rows),
            "selection_row_count": len(selection_rows),
            "methods": methods,
            "nonconverged_method_rows": sum(not bool(row["converged"]) for row in rows),
            "point_estimation_runtime_seconds_per_scan_pair": {
                "mean": finite_float(case_durations.mean()),
                "median": finite_float(np.median(case_durations)),
                "p95": finite_float(np.quantile(case_durations, 0.95)),
                "max": finite_float(case_durations.max()),
                "sum": finite_float(case_durations.sum()),
            },
        },
        "paired_interval_score_contrasts": contrasts,
        "coverage_by_domain": coverage,
        "aggregate_by_method": aggregate,
        "known_domain_das_minus_homoscedastic_interval_score_mm": domain_advantages,
        "rejector": rejector,
        "frozen_success_conditions": {
            "primary_interval_score_effect": primary_pass,
            "known_coverage_and_width_jointly": calibration_efficiency_pass,
            "majority_and_worst_known_domain_advantage": robustness_pass,
            "combined_rejector_beats_all_single_scores": rejector["frozen_hypothesis_passed"],
        },
        "writing_gate": {
            "synthetic_formal_gate_passed": all((
                primary_pass,
                calibration_efficiency_pass,
                robustness_pass,
                rejector["frozen_hypothesis_passed"],
            )),
            "note": (
                "This is only the synthetic formal gate. Real-data external validation, "
                "baseline closure, and deterministic reproduction remain separate gates."
            ),
        },
        "analysis_parameters": {
            "random_seed": args.seed,
            "stratified_bootstrap_repetitions": args.bootstrap,
            "paired_sign_flip_repetitions": args.permutations,
            "aurc_stratified_bootstrap_repetitions": args.aurc_bootstrap,
            "multiple_comparison_correction": "Holm across three predeclared interval-score contrasts",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    primary_ci = primary["stratified_bootstrap_95_ci_mm"]
    lines = [
        "# DAS-FC formal statistical analysis",
        "",
        f"- Independent unit: complete scan pair; known n={report['integrity']['known_case_count']}, unseen n={report['integrity']['unseen_case_count']}.",
        f"- Primary DAS-FC minus homoscedastic interval-score difference: {primary['mean_difference_mm']:.4f} mm (domain-stratified bootstrap 95% CI {primary_ci[0]:.4f} to {primary_ci[1]:.4f}; paired dz={primary['paired_cohen_dz']:.3f}; Holm p={primary['paired_t']['p_holm_three_contrasts']:.4g}).",
        f"- Known-domain DAS-FC simultaneous coverage: {known_aggregate[PRIMARY_METHOD]['simultaneous_coverage']:.3f}; mean width: {known_aggregate[PRIMARY_METHOD]['mean_interval_width_mm']:.3f} mm.",
        f"- Homoscedastic mean width: {known_aggregate['homoscedastic_grouped']['mean_interval_width_mm']:.3f} mm.",
        f"- Combined rejector AURC: {rejector['aurc']['score_combined']:.4f}; best single-score AURC: {min(rejector['aurc'][key] for key in SCORE_KEYS[:-1]):.4f}.",
        "",
        "## Frozen gate verdict",
        "",
    ]
    for key, value in report["frozen_success_conditions"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend([
        "",
        f"Synthetic formal gate: {'PASS' if report['writing_gate']['synthetic_formal_gate_passed'] else 'FAIL'}.",
        "",
        "Unseen-domain results are empirical stress tests and are not assigned a formal 95% coverage guarantee.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "primary": primary,
        "frozen_success_conditions": report["frozen_success_conditions"],
        "synthetic_formal_gate_passed": report["writing_gate"]["synthetic_formal_gate_passed"],
        "rejector_aurc": rejector["aurc"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
