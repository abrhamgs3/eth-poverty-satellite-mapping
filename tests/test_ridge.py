import numpy as np
import pytest

from geo_poverty_mapping.modeling.ridge import SimpleRidgeRegression


def test_recovers_approximate_coefficients_with_low_alpha():
    rng = np.random.default_rng(42)
    X = rng.normal(5, 2, (500, 2))
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(0, 0.3, 500)

    model = SimpleRidgeRegression(alpha=0.01).fit(X, y)
    preds = model.predict(X)

    assert np.corrcoef(preds, y)[0, 1] > 0.95


def test_higher_alpha_shrinks_coefficients_toward_zero():
    rng = np.random.default_rng(42)
    X = rng.normal(5, 2, (200, 2))
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(0, 0.5, 200)

    weak = SimpleRidgeRegression(alpha=0.01).fit(X, y)
    strong = SimpleRidgeRegression(alpha=1000.0).fit(X, y)

    assert abs(strong.beta[1]) < abs(weak.beta[1])


def test_predict_before_fit_raises():
    model = SimpleRidgeRegression()
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))
