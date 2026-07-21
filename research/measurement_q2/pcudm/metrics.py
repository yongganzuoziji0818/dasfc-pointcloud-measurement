"""Metrics whose independent observation is one complete scan pair."""

from __future__ import annotations

import numpy as np


def scan_pair_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have identical shape")
    mask = np.ones(truth.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    mask &= np.isfinite(truth) & np.isfinite(prediction)
    if not mask.any():
        raise ValueError("No valid field locations")
    error = prediction[mask] - truth[mask]
    result = {
        "field_locations": int(mask.sum()),
        "normal_mae_mm": float(np.mean(np.abs(error))),
        "normal_rmse_mm": float(np.sqrt(np.mean(error**2))),
        "signed_bias_mm": float(np.mean(error)),
        "max_abs_error_mm": float(np.max(np.abs(error))),
    }
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError("lower and upper must be provided together")
        lower = np.asarray(lower, dtype=np.float64)[mask]
        upper = np.asarray(upper, dtype=np.float64)[mask]
        covered = (truth[mask] >= lower) & (truth[mask] <= upper)
        width = upper - lower
        below = np.maximum(lower - truth[mask], 0.0)
        above = np.maximum(truth[mask] - upper, 0.0)
        interval_score = width + (2.0 / alpha) * (below + above)
        result.update(
            simultaneous_covered=bool(covered.all()),
            point_coverage=float(covered.mean()),
            mean_interval_width_mm=float(width.mean()),
            normalized_mean_interval_width=float(
                width.mean() / max(np.ptp(truth[mask]), 1e-9)
            ),
            interval_score_mm=float(interval_score.mean()),
        )
    return result


def aurc(risk: np.ndarray, reject_score: np.ndarray) -> float:
    risk = np.asarray(risk, dtype=np.float64)
    reject_score = np.asarray(reject_score, dtype=np.float64)
    if risk.ndim != 1 or risk.shape != reject_score.shape or risk.size == 0:
        raise ValueError("risk and reject_score must be same-length vectors")
    # Low reject score is retained first; prefix means form the risk-coverage curve.
    order = np.argsort(reject_score)
    prefix_risk = np.cumsum(risk[order]) / np.arange(1, risk.size + 1)
    coverage = np.arange(1, risk.size + 1) / risk.size
    return float(np.trapezoid(prefix_risk, coverage))
