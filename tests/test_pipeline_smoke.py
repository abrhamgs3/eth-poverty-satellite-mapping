import os

from geo_poverty_mapping.config import PipelineConfig
from geo_poverty_mapping.pipeline import run_full_pipeline


def test_full_pipeline_runs_end_to_end(tmp_path):
    config = PipelineConfig(seed=1, n_clusters=200, output_dir=str(tmp_path))

    results = run_full_pipeline(config)

    assert set(results["metrics_naive"]) == {"apparent", "latent"}
    assert set(results["metrics_buffer"]) == {"apparent", "latent"}
    assert os.path.isfile(results["report_path"])
    assert os.path.isfile(results["chart_path"])

    with open(results["report_path"], encoding="utf-8") as f:
        report_text = f.read()
    assert "Naive (point)" in report_text
    assert "Buffer (pooled)" in report_text


def test_pipeline_is_reproducible_given_same_seed(tmp_path):
    kwargs = dict(seed=7, n_clusters=150)

    r1 = run_full_pipeline(PipelineConfig(output_dir=str(tmp_path / "a"), **kwargs))
    r2 = run_full_pipeline(PipelineConfig(output_dir=str(tmp_path / "b"), **kwargs))

    assert r1["metrics_buffer"]["latent"]["R2"] == r2["metrics_buffer"]["latent"]["R2"]


def test_short_correlation_length_favors_buffer_over_naive(tmp_path):
    config = PipelineConfig(seed=1, n_clusters=2000, field_correlation_length_km=2.0, output_dir=str(tmp_path / "short"))
    results = run_full_pipeline(config)
    assert results["decomposition"]["jitter_mitigation_r2_gain"] > 0


def test_long_correlation_length_favors_naive_over_buffer(tmp_path):
    config = PipelineConfig(seed=1, n_clusters=2000, field_correlation_length_km=25.0, output_dir=str(tmp_path / "long"))
    results = run_full_pipeline(config)
    assert results["decomposition"]["jitter_mitigation_r2_gain"] < 0
