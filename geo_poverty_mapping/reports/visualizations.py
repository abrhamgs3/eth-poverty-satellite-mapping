"""Stage 5: the two-panel diagnostic chart.

Same layout, colors, and panels as the original prototype (it read well) -
what changed is what feeds it: real sampled predictions instead of
predictions generated from a hardcoded recovery fraction, a guarded style
lookup instead of one that raises on some matplotlib versions, and a
caption that reports the actual decomposition (see performance/benchmarks.py)
instead of the "severely underestimates" claim its own numbers didn't support.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def plot_comparison(
    y_observed, y_latent, y_pred_naive, y_pred_buffer,
    metrics_naive: Dict[str, Any], metrics_buffer: Dict[str, Any],
    decomposition: Dict[str, float],
    out_path: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(y_latent, y_pred_buffer, alpha=0.6, color="#1b9e77", label="Robust Buffer Model", edgecolors="none", s=40)
    ax1.scatter(y_latent, y_pred_naive, alpha=0.4, color="#d95f02", label="Naive Point Model", edgecolors="none", s=40)
    min_val = min(np.min(y_latent), np.min(y_pred_buffer), np.min(y_pred_naive))
    max_val = max(np.max(y_latent), np.max(y_pred_buffer), np.max(y_pred_naive))
    ax1.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Alignment")
    ax1.set_xlabel("True Latent Wealth (Unobserved Ground Truth)", fontsize=12)
    ax1.set_ylabel("Spatially Predicted Wealth Index", fontsize=12)
    ax1.set_title("Predicted vs. True Latent Wealth (Ethiopia)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)

    categories = [
        "Apparent R2\n(Noisy Surveys)", "True R2\n(Latent Wealth)",
        "Apparent r\n(Noisy correlation)", "True r\n(Latent correlation)",
    ]
    naive_values = [
        metrics_naive["apparent"]["R2"], metrics_naive["latent"]["R2"],
        metrics_naive["apparent"]["correlation"], metrics_naive["latent"]["correlation"],
    ]
    buffer_values = [
        metrics_buffer["apparent"]["R2"], metrics_buffer["latent"]["R2"],
        metrics_buffer["apparent"]["correlation"], metrics_buffer["latent"]["correlation"],
    ]
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width / 2, naive_values, width, label="Naive Point Model (No Jitter Mitigation)", color="#d95f02", alpha=0.85)
    ax2.bar(x + width / 2, buffer_values, width, label="Robust Buffer Model (Jitter Mitigated)", color="#1b9e77", alpha=0.85)
    ax2.set_ylabel("Metric Value (0 to 1 scale)", fontsize=12)
    ax2.set_title("Apparent vs. Latent Performance (Burke et al., 2020)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(-0.1, 1.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=10)

    textstr = (
        "Key Takeaway - two separate effects, not one:\n"
        f"1. Noisy-label evaluation gap (apparent vs. latent, same model): {decomposition['noisy_label_gap_buffer_pct']:.1f}% for the "
        f"buffer model - modest, capped near {decomposition['theoretical_max_noisy_label_gap_pct']:.1f}% by the survey's noise-to-signal ratio.\n"
        f"2. GPS-jitter mitigation (naive vs. buffer, both against true wealth): buffer pooling recovers "
        f"{decomposition['jitter_mitigation_r2_gain']:.3f} more latent R2 - measured from actually sampling a spatial field, not assumed."
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
    fig.text(0.5, 0.02, textstr, fontsize=9.5, bbox=props, ha="center", va="bottom", style="italic")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
