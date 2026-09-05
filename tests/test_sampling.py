import numpy as np
import pandas as pd

from geo_poverty_mapping.data_extraction.spatial_field import SmoothWealthField
from geo_poverty_mapping.feature_extraction.sampling import sample_buffer_wealth, sample_naive_wealth


def _toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(5, 13, n)
    lon = rng.uniform(35, 46, n)
    return pd.DataFrame({
        "lat_jittered": lat,
        "lon_jittered": lon,
        "jitter_radius_km": rng.choice([2.0, 5.0], n),
        "base_wealth": rng.normal(0, 0.5, n),
        "urban_premium": rng.choice([1.8, -0.5], n),
    })


def test_naive_matches_field_plus_deterministic_components():
    rng = np.random.default_rng(1)
    field = SmoothWealthField(rng, (3.4, 15.0), (33.0, 48.0), correlation_length_km=2.0, grid_resolution_km=1.0)
    df = _toy_df()

    naive = sample_naive_wealth(field, df)
    expected = df["base_wealth"].to_numpy() + df["urban_premium"].to_numpy() + field.evaluate(
        df["lat_jittered"].to_numpy(), df["lon_jittered"].to_numpy()
    )

    np.testing.assert_allclose(naive, expected)


def test_buffer_reduces_variance_relative_to_naive_field_component():
    # Averaging over a disk is a smoothing operation - its variance should
    # be no larger than a single point sample's, for the same field.
    rng = np.random.default_rng(2)
    field = SmoothWealthField(rng, (3.4, 15.0), (33.0, 48.0), correlation_length_km=2.0, grid_resolution_km=1.0, amplitude_sd=2.0)
    df = _toy_df(n=500, seed=3)

    naive = sample_naive_wealth(field, df)
    buffer = sample_buffer_wealth(field, df, rng, n_samples=40)

    deterministic = df["base_wealth"].to_numpy() + df["urban_premium"].to_numpy()
    assert np.var(buffer - deterministic) <= np.var(naive - deterministic)


def test_buffer_output_shape_matches_input():
    rng = np.random.default_rng(4)
    field = SmoothWealthField(rng, (3.4, 15.0), (33.0, 48.0), correlation_length_km=2.0, grid_resolution_km=1.0)
    df = _toy_df(n=37, seed=5)

    buffer = sample_buffer_wealth(field, df, rng, n_samples=10)

    assert buffer.shape == (37,)
