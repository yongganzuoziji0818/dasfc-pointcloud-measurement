"""Exact paired-t sensitivity analysis for the frozen primary contrast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import nct, norm, t


def paired_t_power(n: int, effect_size: float, alpha: float) -> float:
    degrees = n - 1
    critical = t.ppf(1.0 - alpha / 2.0, degrees)
    noncentrality = abs(effect_size) * np.sqrt(n)
    value = float(
        nct.cdf(-critical, degrees, noncentrality)
        + nct.sf(critical, degrees, noncentrality)
    )
    if not np.isfinite(value):
        # SciPy's noncentral-t tails can return NaN at isolated high-power
        # parameter combinations. At these large degrees of freedom the normal
        # approximation is numerically stable and conservative enough for a
        # sensitivity table far above the sample-size decision boundary.
        value = float(
            norm.cdf(-critical - noncentrality)
            + norm.sf(critical - noncentrality)
        )
    return value


def required_n(effect_size: float, alpha: float, target_power: float) -> int:
    for n in range(4, 100_001):
        if paired_t_power(n, effect_size, alpha) >= target_power:
            return n
    raise RuntimeError("required n exceeds search limit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05 / 3.0)
    parser.add_argument("--power", type=float, default=0.90)
    parser.add_argument("--effects", type=float, nargs="+", default=(0.25, 0.35, 0.50))
    parser.add_argument("--planned-n", type=int, nargs="+", default=(100, 120, 200, 240))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for effect in args.effects:
        rows.append({
            "paired_dz": effect,
            "required_scan_pairs": required_n(effect, args.alpha, args.power),
            "power_at_planned_n": {
                str(n): paired_t_power(n, effect, args.alpha) for n in args.planned_n
            },
        })
    report = {
        "schema_version": "1.0",
        "test": "two-sided paired t sensitivity calculation",
        "independent_unit": "complete scan pair",
        "familywise_alpha": args.alpha,
        "target_power": args.power,
        "effect_basis": "predeclared conservative sensitivity values; not raw pilot effect",
        "rows": rows,
        "limitations": [
            "The confirmatory analysis also uses paired bootstrap/permutation and domain blocks.",
            "This closed-form calculation is a conservative planning reference, not a substitute for design-matched simulation power.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
