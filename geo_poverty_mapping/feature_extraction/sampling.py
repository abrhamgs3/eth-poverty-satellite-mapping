"""Stage 1a: naive point vs. buffer-pooled wealth-signal sampling.

This is the fix for the original prototype's central problem: instead of
declaring "naive extraction captures 25-40% of the signal, buffer
extraction captures 75-85%" as parameters, both numbers are now
*measured* by actually sampling the spatial field (see
``data_extraction/spatial_field.py``) at the relevant coordinates.

- Naive: the field evaluated at the single *jittered* coordinate - exactly
  what a real point-extraction pipeline would pull from Earth Engine at
  the public, displaced coordinate.
- Buffer: a Monte Carlo estimate of the field's mean over the disk of
  radius ``jitter_radius_km`` centered on the *jittered* coordinate. This
  is the correct real-world construction: the *true* location is
  guaranteed (by the DHS/LSMS jitter methodology itself) to lie within
  that radius of the public jittered coordinate, so averaging over the
  disk is guaranteed to be influenced by the true location's value - a
  buffer around the true (unknown) coordinate would be circular to
  compute and isn't what a real pipeline could do anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_extraction.geo_utils import km_to_deg_lat, km_to_deg_lon
from ..data_extraction.spatial_field import SmoothWealthField


def sample_naive_wealth(field: SmoothWealthField, df: pd.DataFrame) -> np.ndarray:
    field_value = field.evaluate(df["lat_jittered"].to_numpy(), df["lon_jittered"].to_numpy())
    return df["base_wealth"].to_numpy() + df["urban_premium"].to_numpy() + field_value


def sample_buffer_wealth(
    field: SmoothWealthField, df: pd.DataFrame, rng: np.random.Generator, n_samples: int = 30,
) -> np.ndarray:
    n_clusters = len(df)
    lat_center = df["lat_jittered"].to_numpy()
    lon_center = df["lon_jittered"].to_numpy()
    radius_km = df["jitter_radius_km"].to_numpy()

    # Uniform-in-area sampling within the disk (r = R*sqrt(U)), so the
    # Monte Carlo mean is an unbiased estimate of the true area-average of
    # the field over the disk - uniform-in-radius would over-sample near
    # the center and bias the estimate.
    u = rng.random((n_clusters, n_samples))
    theta = rng.uniform(0, 2 * np.pi, (n_clusters, n_samples))
    r_km = radius_km[:, None] * np.sqrt(u)

    dlat_km = r_km * np.sin(theta)
    dlon_km = r_km * np.cos(theta)
    sample_lat = lat_center[:, None] + km_to_deg_lat(dlat_km)
    sample_lon = lon_center[:, None] + km_to_deg_lon(dlon_km, lat_center[:, None])

    field_samples = field.evaluate(sample_lat, sample_lon)  # (n_clusters, n_samples)
    mean_field = field_samples.mean(axis=1)

    return df["base_wealth"].to_numpy() + df["urban_premium"].to_numpy() + mean_field
