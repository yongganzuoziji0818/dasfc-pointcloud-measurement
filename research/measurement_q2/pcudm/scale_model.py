"""Cross-fitted structural scale model for field-wise conformal calibration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .estimator import EstimationResult


@dataclass(frozen=True)
class ScaleModelDiagnostics:
    scan_pairs: int
    training_locations: int
    target_quantile: float
    feature_names: tuple[str, ...]


FEATURE_NAMES = (
    "log_raw_local_scale",
    "log_match_distance",
    "support_probability",
    "absolute_predicted_displacement",
    "panel_local_x",
    "panel_local_z",
    "panel_index_normalized",
)


def _local_coordinates(reference: np.ndarray, panel_ids: np.ndarray) -> np.ndarray:
    coordinates = np.zeros((reference.shape[0], 2), dtype=np.float64)
    for panel in np.unique(panel_ids):
        index = panel_ids == panel
        x = reference[index, 0]
        z = reference[index, 2]
        coordinates[index, 0] = (x - x.min()) / max(np.ptp(x), 1e-9)
        coordinates[index, 1] = (z - z.min()) / max(np.ptp(z), 1e-9)
    return coordinates


class StructuralScaleModel:
    """Predict a positive conditional error scale without using test-pair truth.

    Model fitting uses a dedicated tuning split. Conformal quantiles must be fit on
    a separate calibration split; using the same scan pairs for both would invalidate
    the intended split-conformal evidence chain.
    """

    def __init__(
        self,
        *,
        target_quantile: float = 0.80,
        max_locations_per_pair: int = 256,
        floor_mm: float = 0.10,
        random_state: int = 20260719,
    ):
        if not 0.5 < target_quantile < 1.0:
            raise ValueError("target_quantile must be in (0.5, 1)")
        self.target_quantile = target_quantile
        self.max_locations_per_pair = max_locations_per_pair
        self.floor_mm = floor_mm
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=target_quantile,
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self.diagnostics: ScaleModelDiagnostics | None = None

    @staticmethod
    def features(
        reference: np.ndarray, panel_ids: np.ndarray, result: EstimationResult
    ) -> np.ndarray:
        reference = np.asarray(reference, dtype=np.float64)
        panel_ids = np.asarray(panel_ids)
        local = _local_coordinates(reference, panel_ids)
        panel_denominator = max(float(np.max(panel_ids)), 1.0)
        return np.column_stack(
            (
                np.log(np.maximum(result.scale, 1e-6)),
                np.log1p(np.maximum(result.match_distance, 0.0)),
                result.support_probability,
                np.abs(result.normal_displacement),
                local[:, 0],
                local[:, 1],
                panel_ids.astype(np.float64) / panel_denominator,
            )
        )

    def _rng_for_pair(self, case_id: str) -> np.random.Generator:
        digest = hashlib.sha256(case_id.encode("utf-8")).digest()
        case_seed = int.from_bytes(digest[:8], "little") ^ self.random_state
        return np.random.default_rng(case_seed)

    def fit(self, cases: Iterable[dict]) -> ScaleModelDiagnostics:
        feature_rows = []
        targets = []
        weights = []
        scan_pairs = 0
        for case in cases:
            case_id = str(case["case_id"])
            error = np.asarray(case["error"], dtype=np.float64)
            valid = np.asarray(case["valid"], dtype=bool)
            features = self.features(case["reference"], case["panel_ids"], case["result"])
            valid &= np.isfinite(error) & np.isfinite(features).all(axis=1)
            index = np.flatnonzero(valid)
            if index.size == 0:
                continue
            if index.size > self.max_locations_per_pair:
                index = self._rng_for_pair(case_id).choice(
                    index, size=self.max_locations_per_pair, replace=False
                )
            feature_rows.append(features[index])
            targets.append(np.log(np.abs(error[index]) + self.floor_mm))
            weights.append(np.full(index.size, 1.0 / index.size, dtype=np.float64))
            scan_pairs += 1
        if scan_pairs < 2:
            raise ValueError("At least two independent tuning scan pairs are required")
        design = np.concatenate(feature_rows, axis=0)
        response = np.concatenate(targets, axis=0)
        sample_weight = np.concatenate(weights, axis=0)
        self.model.fit(design, response, sample_weight=sample_weight)
        self.diagnostics = ScaleModelDiagnostics(
            scan_pairs=scan_pairs,
            training_locations=int(design.shape[0]),
            target_quantile=self.target_quantile,
            feature_names=FEATURE_NAMES,
        )
        return self.diagnostics

    def predict(
        self, reference: np.ndarray, panel_ids: np.ndarray, result: EstimationResult
    ) -> np.ndarray:
        if self.diagnostics is None:
            raise RuntimeError("StructuralScaleModel has not been fit")
        log_scale = self.model.predict(self.features(reference, panel_ids, result))
        return np.maximum(np.exp(log_scale), self.floor_mm)
