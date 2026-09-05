"""Assemble a single Markdown report summarizing a pipeline run."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict


def build_markdown_report(context: Dict[str, Any], out_path: str) -> str:
    naive = context["metrics_naive"]
    buffer = context["metrics_buffer"]
    decomp = context["decomposition"]
    timings = context.get("timings", {})

    lines = [
        "# Pipeline Run Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Clusters: {context['n_clusters']}",
        "",
        "## Model Performance",
        "",
        "| Model | Apparent R2 | Latent R2 | Apparent r | Latent r |",
        "|---|---|---|---|---|",
        f"| Naive (point) | {naive['apparent']['R2']:.4f} | {naive['latent']['R2']:.4f} | {naive['apparent']['correlation']:.4f} | {naive['latent']['correlation']:.4f} |",
        f"| Buffer (pooled) | {buffer['apparent']['R2']:.4f} | {buffer['latent']['R2']:.4f} | {buffer['apparent']['correlation']:.4f} | {buffer['latent']['correlation']:.4f} |",
        "",
        "## Decomposition — two distinct effects",
        "",
        "### 1. Noisy-label evaluation gap (Burke et al., 2020)",
        "Apparent vs. latent R2 for the *same* model — how much evaluating "
        "against the noisy survey label (instead of the unobservable true "
        "wealth) understates real performance.",
        "",
        f"- Naive model gap: {decomp['noisy_label_gap_naive_pct']:.2f}%",
        f"- Buffer model gap: {decomp['noisy_label_gap_buffer_pct']:.2f}%",
        f"- Theoretical ceiling (from the simulated noise-to-signal variance ratio): {decomp['theoretical_max_noisy_label_gap_pct']:.2f}%",
        "",
        "This gap is *modest by construction* — averaging 15-25 households "
        "per cluster shrinks survey sampling noise a lot relative to "
        "between-cluster wealth variation. A dramatically larger gap would "
        "require a noisier survey instrument, not a different pipeline.",
        "",
        "### 2. GPS-jitter mitigation effect",
        "Naive-point vs. buffer-pooled *latent* R2 — how much real signal "
        "buffer pooling recovers once evaluated against true wealth (i.e., "
        "independent of the label-noise question above).",
        "",
        f"- Latent R2 gain from buffer pooling: {decomp['jitter_mitigation_r2_gain']:.4f}",
        "",
        "This is the effect that matters for satellite-based poverty "
        "mapping, and it's measured here from actually sampling a "
        "spatially-autocorrelated wealth field at the true, jittered, and "
        "buffer-pooled coordinates — not asserted as a parameter.",
        "",
    ]

    if "chart_path" in context:
        rel = os.path.relpath(context["chart_path"], os.path.dirname(out_path))
        lines += ["## Chart", "", f"![Comparison]({rel})", ""]

    if timings:
        lines += ["## Stage Timings", "", "| Stage | Seconds |", "|---|---|"]
        lines += [f"| {stage} | {duration:.3f} |" for stage, duration in timings.items()]
        lines.append("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return out_path
