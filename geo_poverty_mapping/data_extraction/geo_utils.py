"""Small planar-approximation geo helpers.

A full haversine isn't worth its cost here: Ethiopia's cluster coordinates
span a few hundred km, GPS jitter is 2-5km, and field bandwidths are
1.5-8km - at that scale an equirectangular approximation (flat-Earth,
locally scaled by latitude) is accurate to well under 1%, which is smaller
than every other source of noise in this simulation.
"""

from __future__ import annotations

import numpy as np

KM_PER_DEG_LAT = 111.0


def km_per_deg_lon(lat_deg: np.ndarray | float) -> np.ndarray | float:
    return KM_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def planar_distance_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Distance in km between (lat1,lon1) and (lat2,lon2), broadcasting
    over arrays. Uses the mean latitude for the longitude scale factor."""
    mean_lat = (np.asarray(lat1) + np.asarray(lat2)) / 2.0
    dy_km = (np.asarray(lat2) - np.asarray(lat1)) * KM_PER_DEG_LAT
    dx_km = (np.asarray(lon2) - np.asarray(lon1)) * km_per_deg_lon(mean_lat)
    return np.sqrt(dx_km**2 + dy_km**2)


def km_to_deg_lat(km: float) -> float:
    return km / KM_PER_DEG_LAT


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / km_per_deg_lon(lat_deg)
