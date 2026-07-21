"""Classical full-field simultaneous-band baselines.

All inputs use one complete scan pair or trajectory as one row.  Field
locations are never treated as independent experimental observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class LocationNuisance:
    mean_error: np.ndarray
    error_scale: np.ndarray
    location_counts: np.ndarray
    trajectory_count: int


@dataclass(frozen=True)
class ClassicalBandState:
    alpha: float
    group_quantiles: dict[str, float]
    group_order_statistics: dict[str, int]
    group_calibration_counts: dict[str, int]
    nuisance: dict[str, LocationNuisance]
    method: str


def _case_arrays(case: dict) -> tuple[str, np.ndarray, np.ndarray]:
    group = str(case["group"])
    error = np.asarray(case["error"], dtype=np.float64)
    valid = np.ones(error.shape, dtype=bool)
    if case.get("valid") is not None:
        valid = np.asarray(case["valid"], dtype=bool)
    if valid.shape != error.shape:
        raise ValueError("valid and error must have identical shapes")
    valid &= np.isfinite(error)
    return group, error, valid


def _group_cases(cases: Iterable[dict]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    expected_shape: tuple[int, ...] | None = None
    for case in cases:
        group, error, valid = _case_arrays(case)
        if expected_shape is None:
            expected_shape = error.shape
        elif error.shape != expected_shape:
            raise ValueError("classical simultaneous bands require a fixed field topology")
        grouped.setdefault(group, []).append((error, valid))
    if not grouped:
        raise ValueError("at least one trajectory is required")
    return grouped


def _fit_nuisance(
    tuning_cases: Iterable[dict],
    scale_floor: float,
) -> dict[str, LocationNuisance]:
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    grouped = _group_cases(tuning_cases)
    output: dict[str, LocationNuisance] = {}
    for group, cases in grouped.items():
        error = np.stack([item[0] for item in cases])
        valid = np.stack([item[1] for item in cases])
        masked = np.where(valid, error, np.nan)
        counts = valid.sum(axis=0)
        mean = np.divide(
            np.nansum(masked, axis=0),
            counts,
            out=np.full(error.shape[1:], np.nan, dtype=np.float64),
            where=counts > 0,
        )
        # Keep the shared point estimate fixed.  The classical scale therefore
        # absorbs systematic error through zero-centred RMS rather than moving
        # the interval centre by a method-specific bias correction.
        squared = np.sum(np.where(valid, error, 0.0) ** 2, axis=0)
        variance = np.divide(
            squared,
            counts,
            out=np.full(error.shape[1:], np.nan, dtype=np.float64),
            where=counts > 0,
        )
        scale = np.sqrt(variance)
        scale = np.where(np.isfinite(scale), np.maximum(scale, scale_floor), np.nan)
        output[group] = LocationNuisance(
            mean_error=mean,
            error_scale=scale,
            location_counts=counts,
            trajectory_count=len(cases),
        )
    return output


def _finite_pairmax_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("pair-max scores must be a non-empty finite vector")
    order = ceil((scores.size + 1) * (1.0 - alpha))
    if order > scores.size:
        return float("inf"), int(order)
    return float(scores[order - 1]), int(order)


class ClassicalMaxTBand:
    """Two-stage max-t prediction band with trajectory-level calibration.

    Location-wise bias and scale are estimated only on tuning trajectories.
    An independent calibration set supplies one maximum standardized residual
    per trajectory.  The resulting band is therefore a close classical
    simultaneous comparator without borrowing test outcomes.
    """

    def __init__(self, alpha: float = 0.05, scale_floor: float = 1e-6):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha
        self.scale_floor = scale_floor
        self.state: ClassicalBandState | None = None

    def fit(self, tuning_cases: Iterable[dict], calibration_cases: Iterable[dict]):
        nuisance = _fit_nuisance(tuning_cases, self.scale_floor)
        grouped_calibration = _group_cases(calibration_cases)
        quantiles: dict[str, float] = {}
        orders: dict[str, int] = {}
        counts: dict[str, int] = {}
        for group, cases in grouped_calibration.items():
            if group not in nuisance:
                raise KeyError(f"calibration group lacks tuning nuisance estimates: {group}")
            fitted = nuisance[group]
            scores = []
            for error, valid in cases:
                mask = valid & np.isfinite(fitted.mean_error) & np.isfinite(fitted.error_scale)
                if not mask.any():
                    raise ValueError(f"no valid max-t locations in group {group}")
                standardized = np.abs(error[mask])
                standardized /= fitted.error_scale[mask]
                scores.append(float(np.max(standardized)))
            quantile, order = _finite_pairmax_quantile(np.asarray(scores), self.alpha)
            quantiles[group] = quantile
            orders[group] = order
            counts[group] = len(scores)
        self.state = ClassicalBandState(
            alpha=self.alpha,
            group_quantiles=quantiles,
            group_order_statistics=orders,
            group_calibration_counts=counts,
            nuisance=nuisance,
            method="two_stage_classical_max_t",
        )
        return self.state

    def interval(self, prediction: np.ndarray, group: str) -> tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("max-t band has not been fit")
        if group not in self.state.nuisance or group not in self.state.group_quantiles:
            raise KeyError(f"unseen max-t group: {group}")
        prediction = np.asarray(prediction, dtype=np.float64)
        fitted = self.state.nuisance[group]
        if prediction.shape != fitted.mean_error.shape:
            raise ValueError("prediction does not match the frozen field topology")
        centre = prediction
        radius = self.state.group_quantiles[group] * fitted.error_scale
        return centre - radius, centre + radius


class BonferroniGaussianBand:
    """Classical Gaussian prediction band with Bonferroni FWER control."""

    def __init__(self, alpha: float = 0.05, scale_floor: float = 1e-6):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha
        self.scale_floor = scale_floor
        self.nuisance: dict[str, LocationNuisance] | None = None

    def fit(self, tuning_cases: Iterable[dict]):
        self.nuisance = _fit_nuisance(tuning_cases, self.scale_floor)
        return self.nuisance

    def interval(self, prediction: np.ndarray, group: str) -> tuple[np.ndarray, np.ndarray]:
        if self.nuisance is None:
            raise RuntimeError("Bonferroni band has not been fit")
        if group not in self.nuisance:
            raise KeyError(f"unseen Bonferroni group: {group}")
        prediction = np.asarray(prediction, dtype=np.float64)
        fitted = self.nuisance[group]
        if prediction.shape != fitted.mean_error.shape:
            raise ValueError("prediction does not match the frozen field topology")
        valid_locations = np.isfinite(fitted.mean_error) & np.isfinite(fitted.error_scale)
        family_size = int(valid_locations.sum())
        if family_size == 0:
            raise ValueError(f"no valid Bonferroni locations in group {group}")
        critical = float(stats.norm.ppf(1.0 - self.alpha / (2.0 * family_size)))
        centre = prediction
        radius = critical * fitted.error_scale
        return centre - radius, centre + radius

    def diagnostics(self, group: str) -> dict:
        if self.nuisance is None or group not in self.nuisance:
            raise KeyError(group)
        fitted = self.nuisance[group]
        family_size = int(
            np.sum(np.isfinite(fitted.mean_error) & np.isfinite(fitted.error_scale))
        )
        critical = float(stats.norm.ppf(1.0 - self.alpha / (2.0 * family_size)))
        return {
            "method": "bonferroni_gaussian_prediction_band",
            "alpha": self.alpha,
            "family_size": family_size,
            "critical_value": critical,
            "tuning_trajectory_count": fitted.trajectory_count,
            "minimum_location_count": int(np.min(fitted.location_counts)),
        }
