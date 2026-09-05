"""Pipeline orchestrator and CLI.

Wires the stages together in order:

    data_extraction (clusters + spatial field)
        -> feature_extraction (naive point vs. buffer-pooled sampling, satellite bands)
        -> modeling (Ridge, naive vs. buffer)
        -> evaluation (apparent vs. latent)
        -> performance (honest decomposition)
        -> reports
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_extraction import DHSClusterSimulator
from .feature_extraction import (
    sample_buffer_wealth, sample_naive_wealth, simulate_nightlights, simulate_optical_bands,
)
from .modeling import SimpleRidgeRegression, train_test_split
from .evaluation import evaluate_apparent_vs_latent
from .performance import StageTimer, decompose_gaps
from .reports import build_markdown_report, plot_comparison

logger = logging.getLogger("geo_poverty_mapping")

BAND_KEYS = ["S2_red", "S2_green", "S2_blue", "S2_nir", "S2_swir", "NDVI"]


def _feature_frame(prefix: str, wealth: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    bands = simulate_optical_bands(wealth, rng)
    ntl = simulate_nightlights(wealth, rng)
    data = {f"{prefix}_{k}": v for k, v in bands.items()}
    data[f"{prefix}_ntl"] = ntl
    return pd.DataFrame(data)


def run_full_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    timer = StageTimer()
    out_dir = config.ensure_output_dir()

    with timer.time("data_extraction"):
        simulator = DHSClusterSimulator(
            rng, field_correlation_length_km=config.field_correlation_length_km,
            field_grid_resolution_km=config.field_grid_resolution_km,
            field_amplitude_sd=config.field_amplitude_sd,
        )
        df = simulator.generate_clusters(
            n_clusters=config.n_clusters,
            households_per_cluster_range=config.households_per_cluster_range,
            household_outcome_sd=config.household_outcome_sd,
        )
    logger.info("Simulated %d georeferenced clusters.", len(df))

    with timer.time("feature_extraction"):
        naive_wealth = sample_naive_wealth(simulator.field, df)
        buffer_wealth = sample_buffer_wealth(simulator.field, df, rng, n_samples=config.n_buffer_samples)

        naive_features = _feature_frame("naive", naive_wealth, rng)
        buffer_features = _feature_frame("buffer", buffer_wealth, rng)
        df = pd.concat([df, naive_features, buffer_features], axis=1)

    naive_cols = [f"naive_{k}" for k in BAND_KEYS] + ["naive_ntl"]
    buffer_cols = [f"buffer_{k}" for k in BAND_KEYS] + ["buffer_ntl"]

    with timer.time("modeling"):
        df_train, df_test = train_test_split(df, config.test_fraction, rng)

        naive_model = SimpleRidgeRegression(alpha=config.ridge_alpha).fit(
            df_train[naive_cols].to_numpy(), df_train["observed_wealth"].to_numpy(),
        )
        buffer_model = SimpleRidgeRegression(alpha=config.ridge_alpha).fit(
            df_train[buffer_cols].to_numpy(), df_train["observed_wealth"].to_numpy(),
        )
        pred_naive = naive_model.predict(df_test[naive_cols].to_numpy())
        pred_buffer = buffer_model.predict(df_test[buffer_cols].to_numpy())

    y_observed = df_test["observed_wealth"].to_numpy()
    y_latent = df_test["true_latent_wealth"].to_numpy()

    with timer.time("evaluation"):
        metrics_naive = evaluate_apparent_vs_latent(y_observed, y_latent, pred_naive)
        metrics_buffer = evaluate_apparent_vs_latent(y_observed, y_latent, pred_buffer)

    with timer.time("performance"):
        sampling_noise_var = float(
            np.var(df["observed_wealth"].to_numpy() - df["true_latent_wealth"].to_numpy())
        )
        decomposition = decompose_gaps(
            metrics_naive, metrics_buffer,
            true_latent_wealth_var=float(df["true_latent_wealth"].var()),
            sampling_noise_var=sampling_noise_var,
        )

    with timer.time("reports"):
        chart_path = plot_comparison(
            y_observed, y_latent, pred_naive, pred_buffer, metrics_naive, metrics_buffer,
            decomposition, os.path.join(out_dir, "poverty_mapping_diagnostics.png"),
        )
        report_context = {
            "n_clusters": config.n_clusters,
            "metrics_naive": metrics_naive,
            "metrics_buffer": metrics_buffer,
            "decomposition": decomposition,
            "timings": timer.as_dict(),
            "chart_path": chart_path,
        }
        report_path = build_markdown_report(report_context, os.path.join(out_dir, "report.md"))

    return {
        "df": df,
        "metrics_naive": metrics_naive,
        "metrics_buffer": metrics_buffer,
        "decomposition": decomposition,
        "timings": timer.as_dict(),
        "report_path": report_path,
        "chart_path": chart_path,
    }


def _print_summary(results: Dict[str, Any]) -> None:
    naive, buffer = results["metrics_naive"], results["metrics_buffer"]
    decomp = results["decomposition"]

    print("\n--- Naive Point Model ---")
    print(f"  Apparent R2 = {naive['apparent']['R2']:.4f} | Latent R2 = {naive['latent']['R2']:.4f}")

    print("\n--- Robust Buffer Model ---")
    print(f"  Apparent R2 = {buffer['apparent']['R2']:.4f} | Latent R2 = {buffer['latent']['R2']:.4f}")

    print("\n--- Decomposition ---")
    print(f"  Noisy-label gap (buffer model)   : {decomp['noisy_label_gap_buffer_pct']:.2f}%  (ceiling: {decomp['theoretical_max_noisy_label_gap_pct']:.2f}%)")
    print(f"  GPS-jitter mitigation R2 gain    : {decomp['jitter_mitigation_r2_gain']:.4f}")

    print(f"\nReport written to: {results['report_path']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo-poverty-mapping",
        description="GPS-jitter and noisy-label diagnostics for satellite poverty mapping (Ethiopia).",
    )
    parser.add_argument("--clusters", type=int, default=500, dest="n_clusters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-buffer-samples", type=int, default=30)
    parser.add_argument(
        "--correlation-length-km", type=float, default=2.0,
        help="Spatial correlation length of the wealth field. Short lengths "
             "comparable to the 2-5km jitter radii give buffer pooling real "
             "local mismatch to correct; long lengths make the field nearly "
             "flat across one jitter disk, so buffer pooling only adds "
             "smoothing loss and naive point extraction wins. Try 20.0 to "
             "see it flip.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    config = PipelineConfig(
        seed=args.seed, n_clusters=args.n_clusters, ridge_alpha=args.ridge_alpha,
        n_buffer_samples=args.n_buffer_samples, field_correlation_length_km=args.correlation_length_km,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    results = run_full_pipeline(config)
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
