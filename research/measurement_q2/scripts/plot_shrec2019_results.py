"""Create the publication figure for the validated SHREC'19 v1.1 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FRONTENDS = (
    "published_coordinates",
    "cascade_strong",
    "multiscale_trimmed_ptp",
    "robust_ptpl",
)
LABELS = {
    "published_coordinates": "Coordinates\n(identity)",
    "cascade_strong": "Cascade-\nStrong",
    "multiscale_trimmed_ptp": "Trimmed\nPTP",
    "robust_ptpl": "Robust\nPTPL",
}
COLORS = {
    "published_coordinates": "#7F7F7F",
    "cascade_strong": "#0072B2",
    "multiscale_trimmed_ptp": "#E69F00",
    "robust_ptpl": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    return parser.parse_args()


def add_distribution(
    ax: plt.Axes,
    rows: list[dict],
    frontends: tuple[str, ...],
    metric: str,
    ylabel: str,
) -> None:
    rng = np.random.default_rng(2019)
    data = [
        np.asarray([row[metric] for row in rows if row["frontend"] == frontend])
        for frontend in frontends
    ]
    box = ax.boxplot(
        data,
        positions=np.arange(1, len(frontends) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, frontend in zip(box["boxes"], frontends):
        patch.set_facecolor(COLORS[frontend])
        patch.set_alpha(0.35)
    for index, (frontend, values) in enumerate(zip(frontends, data), start=1):
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=7,
            alpha=0.45,
            color=COLORS[frontend],
            edgecolors="none",
            rasterized=True,
        )
    ax.set_xticks(np.arange(1, len(frontends) + 1))
    ax.set_xticklabels([LABELS[item] for item in frontends])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> int:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation.get("status") != "SHREC2019_RESULTS_VALIDATED":
        raise RuntimeError("Refusing to plot an unvalidated SHREC'19 result")
    rows = analysis["pair_metrics"]

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    add_distribution(
        axes[0, 0], rows, FRONTENDS, "resolution_fit_gap",
        "High-low fit gap (dimensionless)",
    )
    add_distribution(
        axes[0, 1], rows, FRONTENDS, "mean_symmetric_fit",
        "Symmetric surface fit (dimensionless)",
    )
    add_distribution(
        axes[1, 0], rows, FRONTENDS[1:], "translation_gap",
        "High-low translation gap (dimensionless)",
    )
    add_distribution(
        axes[1, 1], rows, FRONTENDS[1:], "rotation_gap_degrees",
        "High-low rotation gap (degrees)",
    )
    for label, ax in zip("ABCD", axes.flat):
        ax.text(
            -0.13, 1.05, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
        )
    axes[0, 0].set_title("Resolution sensitivity of observable fit")
    axes[0, 1].set_title("Post-alignment observable fit")
    axes[1, 0].set_title("Translation consistency across resolutions")
    axes[1, 1].set_title("Rotation consistency across resolutions")
    fig.suptitle(
        "SHREC'19 real meshes: finite-benchmark descriptive distributions (76 pair rows)",
        fontsize=9,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        path = args.output_prefix.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = str(path)
    plt.close(fig)

    metadata = {
        "schema_version": "1.0",
        "protocol_id": validation["protocol_id"],
        "validation_status": validation["status"],
        "pair_rows_per_frontend": 76,
        "uncertainty_display": "Boxes show median and interquartile range; whiskers use the matplotlib 1.5-IQR rule; all pair rows are overlaid.",
        "independence_warning": "The 76 graph edges are reporting rows, not 76 independent physical specimens.",
        "identity_control_warning": "The identity control is omitted from transform-gap panels because its transform gaps are zero by construction.",
        "inference": "No p-values, significance marks, or population confidence intervals.",
        "formats": outputs,
    }
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
