"""Stage 2: dependency-free Ridge regression.

Plain NumPy - no scikit-learn/statsmodels compiled-extension dependency,
same portability rationale as the sibling repos' ``SimpleOLS``. No bugs
found in the original prototype's math here; ported with an explicit
``add_constant`` intercept exemption from the L2 penalty, unchanged.
"""

from __future__ import annotations

import numpy as np


class SimpleRidgeRegression:
    def __init__(self, alpha: float = 1.0, add_constant: bool = True):
        self.alpha = alpha
        self.add_constant = add_constant
        self.beta: np.ndarray | None = None
        self.means_x: np.ndarray | None = None
        self.stds_x: np.ndarray | None = None

    def _normalize(self, X: np.ndarray, training: bool) -> np.ndarray:
        if training:
            self.means_x = np.mean(X, axis=0)
            self.stds_x = np.std(X, axis=0)
            self.stds_x = np.where(self.stds_x == 0.0, 1.0, self.stds_x)
        return (X - self.means_x) / self.stds_x

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRidgeRegression":
        y = np.asarray(y, dtype=float).flatten()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_norm = self._normalize(X, training=True)
        n_samples, n_features = X_norm.shape

        if self.add_constant:
            X_model = np.column_stack((np.ones(n_samples), X_norm))
            penalty = np.eye(n_features + 1)
            penalty[0, 0] = 0.0  # don't penalize the intercept
        else:
            X_model = X_norm
            penalty = np.eye(n_features)

        lhs = X_model.T @ X_model + self.alpha * penalty
        rhs = X_model.T @ y
        try:
            self.beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.pinv(lhs) @ rhs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_norm = self._normalize(X, training=False)
        if self.add_constant:
            X_model = np.column_stack((np.ones(len(X_norm)), X_norm))
        else:
            X_model = X_norm
        return X_model @ self.beta
