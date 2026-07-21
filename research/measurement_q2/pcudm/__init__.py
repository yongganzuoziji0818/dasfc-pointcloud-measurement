"""Auditable research implementation of PCU-DM-Field.

This package is intentionally independent of Qt and the production GUI. It keeps
synthetic truth, estimators, calibration, and evaluation at scan-pair level.
"""

from .calibration import PairwiseSimultaneousCalibrator
from .estimator import EstimationResult, PCUDMFieldEstimator
from .scale_model import StructuralScaleModel
from .synthetic import SyntheticCase, SyntheticDomain, generate_case

__all__ = [
    "EstimationResult",
    "PCUDMFieldEstimator",
    "PairwiseSimultaneousCalibrator",
    "SyntheticCase",
    "SyntheticDomain",
    "StructuralScaleModel",
    "generate_case",
]
