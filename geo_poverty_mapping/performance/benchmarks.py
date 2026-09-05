"""Pipeline runtime benchmarking, and - the actual point of this rewrite -
an honest decomposition of the two effects the original prototype
conflated into one overstated headline.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


class StageTimer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def time(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage_name] = time.perf_counter() - start

    def as_dict(self) -> Dict[str, float]:
        return dict(self.timings)


def decompose_gaps(
    metrics_naive: Dict[str, Any],
    metrics_buffer: Dict[str, Any],
    true_latent_wealth_var: float,
    sampling_noise_var: float,
) -> Dict[str, float]:
    """Two genuinely different effects, reported separately:

    1. **Noisy-label evaluation gap** (Burke et al., 2020's actual point):
       apparent R2 vs. latent R2 for the *same* model. Its ceiling is set
       by the noise-to-signal variance ratio, computed here directly from
       the simulated data rather than asserted - if this comes out small,
       that's the correct, honest report, not a reason to inflate the noise.
    2. **GPS-jitter mitigation effect**: the naive vs. buffer model's
       *latent* R2, i.e., how much actual predictive power buffer pooling
       recovers once you're no longer looking through the noisy label at
       all. This is the effect that was previously hardcoded; here it's
       measured after both models are fit on real (simulated) sampled data.
    """
    naive_gap_pct = (
        (1 - metrics_naive["apparent"]["R2"] / metrics_naive["latent"]["R2"]) * 100.0
        if metrics_naive["latent"]["R2"] > 0 else float("nan")
    )
    buffer_gap_pct = (
        (1 - metrics_buffer["apparent"]["R2"] / metrics_buffer["latent"]["R2"]) * 100.0
        if metrics_buffer["latent"]["R2"] > 0 else float("nan")
    )
    theoretical_max_gap_pct = (
        sampling_noise_var / (true_latent_wealth_var + sampling_noise_var) * 100.0
    )

    jitter_mitigation_r2_gain = metrics_buffer["latent"]["R2"] - metrics_naive["latent"]["R2"]

    return {
        "noisy_label_gap_naive_pct": naive_gap_pct,
        "noisy_label_gap_buffer_pct": buffer_gap_pct,
        "theoretical_max_noisy_label_gap_pct": theoretical_max_gap_pct,
        "jitter_mitigation_r2_gain": jitter_mitigation_r2_gain,
    }
