"""Stage 3: apparent (vs. noisy survey label) vs. latent (vs. true wealth)
model performance.

The math here (R2, Pearson r, RMSE) was correct in the original prototype;
what was wrong was the narrative built on top of it, driven by a noise
parameterization that mathematically caps the apparent/latent gap at
~1-2%. This module doesn't change the metrics - it's stage 4
(``performance``) that reports the gap honestly instead of as a "severe
deception," and quantifies it as a fraction of the noise-to-signal ratio
that determines it.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_apparent_vs_latent(
    y_observed: np.ndarray, y_latent: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    return {
        "apparent": {
            "R2": r_squared(y_observed, y_pred),
            "correlation": pearson_correlation(y_observed, y_pred),
            "RMSE": rmse(y_observed, y_pred),
        },
        "latent": {
            "R2": r_squared(y_latent, y_pred),
            "correlation": pearson_correlation(y_latent, y_pred),
            "RMSE": rmse(y_latent, y_pred),
        },
    }
