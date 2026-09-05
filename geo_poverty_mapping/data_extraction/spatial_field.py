"""A spatially-autocorrelated wealth surface, evaluable at any coordinate.

This is what makes GPS jitter and buffer-pooling have a real, geometry-
derived effect: the field varies smoothly but genuinely over a few
kilometers (comparable to real DHS jitter radii), so a naive point sample
at a *displaced* coordinate captures a measurably different value than the
true location - and averaging over a disk that's guaranteed to contain the
true location recovers a value closer to it. Neither fact is hardcoded;
both are consequences of sampling this field at different points.

Implementation: white noise on a regular lat/lon grid, low-pass filtered in
the frequency domain to a target correlation length, then bilinearly
interpolated at query points. (An earlier version of this file used a
sparse sum of ~150 randomly-placed Gaussian bumps - with bumps that
localized spread over a domain this large, almost every query point landed
farther than several bandwidths from every bump, so the "field" was
essentially zero almost everywhere: verified empirically, its variance was
~400x smaller than the deterministic region/urban trend, meaning GPS
jitter had virtually nothing to displace. A dense smoothed-noise grid does
not have that sparsity problem - every cell has a well-defined value.)
"""

from __future__ import annotations

import numpy as np

from .geo_utils import km_per_deg_lon, KM_PER_DEG_LAT


class SmoothWealthField:
    def __init__(
        self,
        rng: np.random.Generator,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        correlation_length_km: float = 4.0,
        grid_resolution_km: float = 1.5,
        amplitude_sd: float = 1.0,
    ):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range

        mean_lat = (self.lat_min + self.lat_max) / 2.0
        lat_span_km = (self.lat_max - self.lat_min) * KM_PER_DEG_LAT
        lon_span_km = (self.lon_max - self.lon_min) * km_per_deg_lon(mean_lat)

        self.n_lat = max(int(lat_span_km / grid_resolution_km), 8)
        self.n_lon = max(int(lon_span_km / grid_resolution_km), 8)
        # Actual per-cell km size (close to but not exactly grid_resolution_km,
        # since the grid must divide the domain into a whole number of cells).
        cell_km_lat = lat_span_km / self.n_lat
        cell_km_lon = lon_span_km / self.n_lon

        white_noise = rng.normal(0, 1, size=(self.n_lat, self.n_lon))
        sigma_cells_lat = correlation_length_km / cell_km_lat
        sigma_cells_lon = correlation_length_km / cell_km_lon
        smoothed = _gaussian_smooth_fft(white_noise, sigma_cells_lat, sigma_cells_lon)

        # Renormalize post-smoothing variance to the requested amplitude -
        # Gaussian smoothing predictably shrinks variance, but relying on
        # the closed-form factor is more fragile than just measuring it.
        smoothed -= smoothed.mean()
        std = smoothed.std()
        self.grid = smoothed / std * amplitude_sd if std > 0 else smoothed

    def evaluate(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Bilinearly-interpolated field value at each (lat, lon) pair."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)

        lat_f = np.clip(
            (lats - self.lat_min) / (self.lat_max - self.lat_min) * (self.n_lat - 1), 0, self.n_lat - 1
        )
        lon_f = np.clip(
            (lons - self.lon_min) / (self.lon_max - self.lon_min) * (self.n_lon - 1), 0, self.n_lon - 1
        )

        lat0 = np.floor(lat_f).astype(int)
        lon0 = np.floor(lon_f).astype(int)
        lat1 = np.minimum(lat0 + 1, self.n_lat - 1)
        lon1 = np.minimum(lon0 + 1, self.n_lon - 1)
        wlat = lat_f - lat0
        wlon = lon_f - lon0

        v00 = self.grid[lat0, lon0]
        v01 = self.grid[lat0, lon1]
        v10 = self.grid[lat1, lon0]
        v11 = self.grid[lat1, lon1]

        return (
            v00 * (1 - wlat) * (1 - wlon) + v01 * (1 - wlat) * wlon
            + v10 * wlat * (1 - wlon) + v11 * wlat * wlon
        )


def _gaussian_smooth_fft(grid: np.ndarray, sigma_cells_y: float, sigma_cells_x: float) -> np.ndarray:
    """Circular Gaussian smoothing via the frequency domain: the Fourier
    transform of a Gaussian is a Gaussian, so this is one FFT, one
    multiply, one inverse FFT - no explicit spatial kernel or padding
    needed. Circular (wraparound) boundary handling is an acceptable
    approximation here - the domain is a stylized country-sized bounding
    box, not a physical torus, but edge artifacts from wraparound are
    negligible next to everything else this simulation already stylizes.
    """
    ny, nx = grid.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    transfer = np.exp(-2 * np.pi**2 * (sigma_cells_y**2 * fy**2 + sigma_cells_x**2 * fx**2))
    return np.real(np.fft.ifft2(np.fft.fft2(grid) * transfer))
