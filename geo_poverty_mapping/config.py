"""Central configuration for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")


@dataclass
class PipelineConfig:
    """Runtime parameters for a pipeline run.

    ``seed`` drives a single ``numpy.random.Generator`` threaded through
    every stage, rather than the global ``numpy.random`` state the
    original prototype re-seeded in three different constructors.
    """

    seed: int = 42
    n_clusters: int = 500
    test_fraction: float = 0.2
    ridge_alpha: float = 1.0

    # The smooth wealth field (see data_extraction/spatial_field.py) is
    # what makes GPS jitter and buffer pooling have a *real*, geometry-
    # derived effect instead of a hardcoded one.
    #
    # The correlation length relative to the jitter radius (2-5km) is what
    # determines whether buffer pooling helps *or hurts* - verified by
    # sweeping it: short correlation lengths (~2km, real fine-grained
    # features like individual structures) mean the field has real local
    # variation for a displaced point sample to miss, and buffer averaging
    # nets out ahead by capturing it. Longer correlation lengths (>=5km,
    # roughly neighborhood-scale) make the field nearly flat across a
    # single jitter disk, at which point buffer averaging only adds
    # smoothing/regression-to-the-mean with no mismatch left to correct -
    # naive point extraction wins outright. This is a real bias-variance
    # tradeoff in the geospatial literature, not a modeling artifact - see
    # the README.
    field_correlation_length_km: float = 2.0
    field_grid_resolution_km: float = 1.0
    field_amplitude_sd: float = 2.5

    # Household-survey sampling noise (independent of GPS jitter -
    # contaminates the label, not the features). Real DHS/LSMS clusters
    # survey ~15-25 households; averaging shrinks this a lot, which is
    # exactly why the resulting apparent-vs-latent gap is modest, not the
    # "severe deception" the original prototype's narrative claimed.
    households_per_cluster_range: tuple[int, int] = (15, 25)
    household_outcome_sd: float = 1.0

    # Buffer-pooling Monte Carlo sample count per cluster.
    n_buffer_samples: int = 30

    output_dir: str = field(default=OUTPUT_DIR)

    def ensure_output_dir(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
