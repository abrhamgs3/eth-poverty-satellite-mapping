"""Stage 1b: stylized satellite-band simulation from a wealth value.

This part of the original prototype wasn't the problem - correlating
Sentinel-2-like bands and VIIRS-like nightlights with a wealth value is a
reasonable stylization of real remote-sensing wealth proxies. What changed
is what wealth value gets fed in: ``sampling.py`` now measures it from the
spatial field instead of it being a hardcoded recovery fraction, and this
takes an explicit ``rng`` instead of touching global ``numpy.random`` state.
"""

from __future__ import annotations

import numpy as np


def simulate_optical_bands(wealth: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = np.shape(wealth)
    red = np.clip(1000 + 300 * wealth + rng.normal(0, 150, n), 400, 3000)
    green = np.clip(1100 + 250 * wealth + rng.normal(0, 150, n), 400, 3000)
    blue = np.clip(900 + 200 * wealth + rng.normal(0, 150, n), 400, 3000)
    nir = np.clip(2500 - 150 * wealth + rng.normal(0, 300, n), 1000, 4500)
    swir1 = np.clip(1800 - 100 * wealth + rng.normal(0, 200, n), 800, 3500)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return {
        "S2_red": red / 10000.0, "S2_green": green / 10000.0, "S2_blue": blue / 10000.0,
        "S2_nir": nir / 10000.0, "S2_swir": swir1 / 10000.0, "NDVI": ndvi,
    }


def simulate_nightlights(wealth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ntl_base = np.exp(0.8 * wealth) + rng.normal(0, 0.5, np.shape(wealth))
    return np.clip(ntl_base, 0.0, 63.0)
