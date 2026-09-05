import numpy as np
import pytest

from geo_poverty_mapping.evaluation.metrics import evaluate_apparent_vs_latent, pearson_correlation, r_squared, rmse


def test_perfect_prediction_gives_perfect_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)
    assert pearson_correlation(y, y) == pytest.approx(1.0)
    assert rmse(y, y) == pytest.approx(0.0)


def test_pearson_correlation_handles_constant_input_without_nan():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert pearson_correlation(y_true, y_pred) == 0.0


def test_evaluate_apparent_vs_latent_structure():
    rng = np.random.default_rng(0)
    y_pred = rng.normal(size=100)
    y_observed = y_pred + rng.normal(scale=0.1, size=100)
    y_latent = y_pred + rng.normal(scale=0.01, size=100)

    result = evaluate_apparent_vs_latent(y_observed, y_latent, y_pred)

    assert set(result) == {"apparent", "latent"}
    for key in ("R2", "correlation", "RMSE"):
        assert key in result["apparent"]
        assert key in result["latent"]
    # Less noise vs. latent than vs. observed -> latent R2 should be higher.
    assert result["latent"]["R2"] > result["apparent"]["R2"]
