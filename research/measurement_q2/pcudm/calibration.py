"""Scan-pair-level simultaneous conformal calibration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibrationState:
    alpha: float
    group_quantiles: dict[str, float]
    group_counts: dict[str, int]
    group_order_statistics: dict[str, int]
    pooled_quantile: float
    pooled_count: int
    pooled_order_statistic: int


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a non-empty finite vector")
    order = ceil((scores.size + 1) * (1.0 - alpha))
    order = min(max(order, 1), scores.size)
    return float(scores[order - 1]), int(order)


class PairwiseSimultaneousCalibrator:
    """Calibrate a full-field interval using one score per independent scan pair."""

    def __init__(self, alpha: float = 0.05, epsilon: float = 1e-6):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha
        self.epsilon = epsilon
        self.state: CalibrationState | None = None

    @staticmethod
    def pair_score(error: np.ndarray, scale: np.ndarray, valid: np.ndarray | None = None) -> float:
        error = np.asarray(error, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if error.shape != scale.shape:
            raise ValueError("error and scale must have identical shape")
        mask = np.ones(error.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        mask &= np.isfinite(error) & np.isfinite(scale) & (scale > 0)
        if not mask.any():
            raise ValueError("No valid field locations for calibration")
        return float(np.max(np.abs(error[mask]) / scale[mask]))

    def fit(self, cases: Iterable[dict]) -> CalibrationState:
        grouped: dict[str, list[float]] = {}
        pooled = []
        for case in cases:
            group = str(case["group"])
            score = self.pair_score(case["error"], case["scale"], case.get("valid"))
            grouped.setdefault(group, []).append(score)
            pooled.append(score)
        if not pooled:
            raise ValueError("At least one independent calibration scan pair is required")
        group_quantiles = {}
        group_counts = {}
        group_orders = {}
        for group, values in sorted(grouped.items()):
            quantile, order = _finite_sample_quantile(np.asarray(values), self.alpha)
            group_quantiles[group] = quantile
            group_counts[group] = len(values)
            group_orders[group] = order
        pooled_quantile, pooled_order = _finite_sample_quantile(np.asarray(pooled), self.alpha)
        self.state = CalibrationState(
            alpha=self.alpha,
            group_quantiles=group_quantiles,
            group_counts=group_counts,
            group_order_statistics=group_orders,
            pooled_quantile=pooled_quantile,
            pooled_count=len(pooled),
            pooled_order_statistic=pooled_order,
        )
        return self.state

    def quantile(self, group: str, *, allow_pooled_fallback: bool = False) -> float:
        if self.state is None:
            raise RuntimeError("Calibrator has not been fit")
        if group in self.state.group_quantiles:
            return self.state.group_quantiles[group]
        if allow_pooled_fallback:
            return self.state.pooled_quantile
        raise KeyError(f"Unseen calibration group: {group}")

    def interval(
        self,
        prediction: np.ndarray,
        scale: np.ndarray,
        group: str,
        *,
        allow_pooled_fallback: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(prediction, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if prediction.shape != scale.shape:
            raise ValueError("prediction and scale must have identical shape")
        radius = self.quantile(group, allow_pooled_fallback=allow_pooled_fallback) * np.maximum(
            scale, self.epsilon
        )
        return prediction - radius, prediction + radius

