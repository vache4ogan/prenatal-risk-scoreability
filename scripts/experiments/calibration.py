"""Monotone Platt transformation for ranking sensitivity checks."""
from __future__ import annotations
import numpy as np
from scipy.special import expit

def clip_probabilities(probabilities: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities contain NaN or infinity.")
    return np.clip(values, eps, 1.0 - eps)


def probability_logit(probabilities: np.ndarray, eps: float) -> np.ndarray:
    clipped = clip_probabilities(probabilities, eps)
    return np.log(clipped) - np.log1p(-clipped)


def apply_platt(
    probabilities: np.ndarray,
    intercept: float,
    slope: float,
    eps: float,
) -> np.ndarray:
    calibrated = expit(intercept + slope * probability_logit(probabilities, eps))
    return clip_probabilities(calibrated, eps)
