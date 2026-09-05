"""Stage 0: simulate georeferenced DHS-style survey clusters over Ethiopia.

Real regional centroids and the real DHS GPS-displacement methodology
(2km urban / 5km rural, uniform angle and distance within that radius) -
see ``docs`` note in the README for what's real vs. stylized here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo_utils import km_to_deg_lat, km_to_deg_lon
from .spatial_field import SmoothWealthField

LAT_MIN, LAT_MAX = 3.4, 15.0
LON_MIN, LON_MAX = 33.0, 48.0

REGIONS = {
    "Addis_Ababa": {"lat": 9.03, "lon": 38.74, "urban_prob": 1.0, "base_wealth": 2.5},
    "Amhara": {"lat": 11.5, "lon": 38.0, "urban_prob": 0.15, "base_wealth": -0.2},
    "Oromia": {"lat": 7.5, "lon": 39.5, "urban_prob": 0.18, "base_wealth": -0.1},
    "Tigray": {"lat": 13.5, "lon": 39.0, "urban_prob": 0.20, "base_wealth": -0.3},
    "Somali": {"lat": 7.0, "lon": 44.0, "urban_prob": 0.10, "base_wealth": -0.8},
    "SNNPR": {"lat": 6.5, "lon": 37.5, "urban_prob": 0.12, "base_wealth": -0.4},
}


class DHSClusterSimulator:
    def __init__(
        self,
        rng: np.random.Generator,
        field_correlation_length_km: float = 4.0,
        field_grid_resolution_km: float = 1.5,
        field_amplitude_sd: float = 1.0,
    ):
        self.rng = rng
        self.field = SmoothWealthField(
            rng, (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX),
            correlation_length_km=field_correlation_length_km,
            grid_resolution_km=field_grid_resolution_km,
            amplitude_sd=field_amplitude_sd,
        )

    def generate_clusters(
        self,
        n_clusters: int = 500,
        households_per_cluster_range: tuple[int, int] = (15, 25),
        household_outcome_sd: float = 1.0,
    ) -> pd.DataFrame:
        rng = self.rng
        region_names = list(REGIONS.keys())
        regions = rng.choice(region_names, size=n_clusters)

        lat_centroid = np.array([REGIONS[r]["lat"] for r in regions])
        lon_centroid = np.array([REGIONS[r]["lon"] for r in regions])
        urban_prob = np.array([REGIONS[r]["urban_prob"] for r in regions])
        base_wealth = np.array([REGIONS[r]["base_wealth"] for r in regions])

        lat_true = np.clip(lat_centroid + rng.normal(0, 1.2, n_clusters), LAT_MIN, LAT_MAX)
        lon_true = np.clip(lon_centroid + rng.normal(0, 1.5, n_clusters), LON_MIN, LON_MAX)

        is_urban = rng.random(n_clusters) < urban_prob
        urban_premium = np.where(is_urban, 1.8, -0.5)

        # The true wealth signal a satellite would (in principle) see: a
        # regional/urban trend plus the fine-grained spatially-correlated
        # field - the field is what jitter distorts and buffer pooling
        # partially recovers; the region/urban trend is constant across
        # jitter distances of a few km and unaffected either way.
        field_true = self.field.evaluate(lat_true, lon_true)
        true_latent_wealth = base_wealth + urban_premium + field_true

        # Household-survey sampling noise: an independent channel from GPS
        # jitter, contaminating the *label* (observed_wealth), not any
        # spatial feature.
        n_households = rng.integers(*households_per_cluster_range, size=n_clusters, endpoint=True)
        sampling_noise = rng.normal(0, household_outcome_sd / np.sqrt(n_households))
        observed_wealth = true_latent_wealth + sampling_noise

        # GPS jittering: real DHS/LSMS methodology - uniform angle, uniform
        # distance up to the radius (urban 2km, rural 5km).
        jitter_radius_km = np.where(is_urban, 2.0, 5.0)
        angle = rng.uniform(0, 2 * np.pi, n_clusters)
        jitter_distance_km = rng.uniform(0, jitter_radius_km)
        dlat_km = jitter_distance_km * np.sin(angle)
        dlon_km = jitter_distance_km * np.cos(angle)
        lat_jittered = lat_true + km_to_deg_lat(dlat_km)
        lon_jittered = lon_true + km_to_deg_lon(dlon_km, lat_true)

        return pd.DataFrame({
            "cluster_id": np.arange(n_clusters),
            "region": regions,
            "urban_rural": np.where(is_urban, "urban", "rural"),
            "lat_true": lat_true,
            "lon_true": lon_true,
            "lat_jittered": lat_jittered,
            "lon_jittered": lon_jittered,
            "jitter_radius_km": jitter_radius_km,
            "jitter_distance_km": jitter_distance_km,
            "base_wealth": base_wealth,
            "urban_premium": urban_premium,
            "true_latent_wealth": true_latent_wealth,
            "observed_wealth": observed_wealth,
            "surveyed_households": n_households,
        })
