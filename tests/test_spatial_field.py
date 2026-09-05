import numpy as np
import pytest

from geo_poverty_mapping.data_extraction.spatial_field import SmoothWealthField


def test_field_variance_matches_requested_amplitude():
    rng = np.random.default_rng(0)
    field = SmoothWealthField(
        rng, lat_range=(3.4, 15.0), lon_range=(33.0, 48.0),
        correlation_length_km=2.0, grid_resolution_km=1.0, amplitude_sd=2.5,
    )
    # The grid itself (not a sparse sample of it) should have ~ the
    # requested variance - this is what regressed to ~0.01 in the earlier
    # sparse-bump design instead of the requested 1.0, so it's worth
    # asserting directly rather than only checking it indirectly.
    assert field.grid.std() == pytest.approx(2.5, rel=1e-6)


def test_field_is_smooth_nearby_points_are_similar():
    rng = np.random.default_rng(1)
    field = SmoothWealthField(
        rng, lat_range=(3.4, 15.0), lon_range=(33.0, 48.0),
        correlation_length_km=4.0, grid_resolution_km=1.0, amplitude_sd=1.0,
    )
    lat0, lon0 = 9.0, 39.0
    near = field.evaluate(np.array([lat0]), np.array([lon0 + 0.001]))  # ~0.1km away
    far = field.evaluate(np.array([lat0]), np.array([lon0 + 1.0]))  # ~111km away
    base = field.evaluate(np.array([lat0]), np.array([lon0]))

    assert abs(near[0] - base[0]) < abs(far[0] - base[0])


def test_evaluate_handles_2d_query_arrays():
    rng = np.random.default_rng(2)
    field = SmoothWealthField(rng, (3.4, 15.0), (33.0, 48.0), correlation_length_km=2.0, grid_resolution_km=1.0)
    lats = np.full((5, 10), 9.0)
    lons = np.full((5, 10), 39.0)

    result = field.evaluate(lats, lons)

    assert result.shape == (5, 10)


def test_evaluate_clips_out_of_range_queries_instead_of_raising():
    rng = np.random.default_rng(3)
    field = SmoothWealthField(rng, (3.4, 15.0), (33.0, 48.0), correlation_length_km=2.0, grid_resolution_km=1.0)

    result = field.evaluate(np.array([100.0, -100.0]), np.array([200.0, -200.0]))

    assert np.all(np.isfinite(result))
