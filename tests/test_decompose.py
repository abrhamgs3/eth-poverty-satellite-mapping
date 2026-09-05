import pytest

from geo_poverty_mapping.performance.benchmarks import decompose_gaps


def test_decompose_gaps_matches_theoretical_ratio_in_the_noiseless_limit():
    # If observed == latent exactly (no label noise), the apparent/latent
    # gap must be exactly zero, and the theoretical ceiling (noise_var=0)
    # must be zero too.
    metrics_naive = {"apparent": {"R2": 0.5}, "latent": {"R2": 0.5}}
    metrics_buffer = {"apparent": {"R2": 0.7}, "latent": {"R2": 0.7}}

    result = decompose_gaps(metrics_naive, metrics_buffer, true_latent_wealth_var=4.0, sampling_noise_var=0.0)

    assert result["noisy_label_gap_naive_pct"] == pytest.approx(0.0)
    assert result["noisy_label_gap_buffer_pct"] == pytest.approx(0.0)
    assert result["theoretical_max_noisy_label_gap_pct"] == pytest.approx(0.0)


def test_decompose_gaps_reports_jitter_mitigation_as_latent_r2_difference():
    metrics_naive = {"apparent": {"R2": 0.5}, "latent": {"R2": 0.6}}
    metrics_buffer = {"apparent": {"R2": 0.55}, "latent": {"R2": 0.65}}

    result = decompose_gaps(metrics_naive, metrics_buffer, true_latent_wealth_var=4.0, sampling_noise_var=0.1)

    assert result["jitter_mitigation_r2_gain"] == pytest.approx(0.05)


def test_theoretical_ceiling_matches_noise_to_signal_ratio():
    metrics_naive = {"apparent": {"R2": 0.5}, "latent": {"R2": 0.5}}
    metrics_buffer = {"apparent": {"R2": 0.5}, "latent": {"R2": 0.5}}

    result = decompose_gaps(metrics_naive, metrics_buffer, true_latent_wealth_var=9.0, sampling_noise_var=1.0)

    assert result["theoretical_max_noisy_label_gap_pct"] == pytest.approx(10.0)
