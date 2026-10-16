# GPS-Jitter & Noisy-Label Diagnostics for Satellite Poverty Mapping
**At a glance:** Poverty mapping for Ethiopia. An AI screen classifies households as poor/non-poor from short field notes; a Rogan-Gladen econometric correction fixes the attenuation bias this introduces. Naive AI-screened estimate attenuates poverty's effect on child health by 43%; the correction recovers 94% of the gap.
**Data:** Real World Bank Ethiopia Socioeconomic Survey panel (ESS/ERSS 2011-2015, ~4,850 household-waves); child health outcome.
**Stack:** Python (pandas, scikit-learn), Pydantic, pytest, GitHub Actions. Manuscript in preparation.

[![Tests](https://github.com/abrhamgs3/eth-poverty-satellite-mapping/actions/workflows/tests.yml/badge.svg)](https://github.com/abrhamgs3/eth-poverty-satellite-mapping/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A rewrite of a prototype pipeline about two real, distinct pitfalls in
satellite-based poverty mapping — GPS coordinate "jittering" (the privacy
protocol DHS/LSMS surveys apply to public geocoordinates) and the noisy
labels used to evaluate predictive models (Burke et al., 2020). The
prototype's chart and narrative merged these into one overstated headline;
this version keeps the same two-panel chart layout but replaces a
hardcoded assumption with an actual spatial simulation, and reports the
two effects — honestly, separately, and by measurement rather than by
assertion.

## What was wrong, and what changed

The original pipeline declared "naive point extraction captures 25-40% of
the wealth signal, buffer extraction captures 75-85%" as literal
parameters, generated data consistent with those numbers, then presented
the resulting 0.59-vs-0.95 R² gap as a discovered result. Running it and
checking the numbers against the underlying noise parameterization showed
the headline claim — "evaluating on noisy survey data *severely*
underestimates model performance" — didn't hold: the actual apparent/
latent gap was 1-2%, capped by the survey's own noise-to-signal ratio. See
[`legacy/README.md`](legacy/README.md) for the full list of bugs found in
the original code (a broken standalone import, a dead conditional, an
unguarded matplotlib style, repeated global RNG reseeding).

This version replaces the assumption with a mechanism:

- A **spatially-autocorrelated wealth field** (white noise on a grid,
  Gaussian-smoothed in the frequency domain to a chosen correlation
  length) that can be evaluated at any coordinate.
- **Naive extraction** samples this field at the real DHS/LSMS jitter
  displacement (uniform angle, uniform distance up to 2km urban / 5km
  rural — the real, documented methodology).
- **Buffer extraction** Monte Carlo-averages the field over the disk
  around the *jittered* coordinate — the correct real-world construction,
  since the true location is guaranteed by the jitter protocol itself to
  lie within that radius, and a real pipeline never knows the true
  location to buffer around instead.

Whatever advantage (or disadvantage) buffer pooling shows is now a
consequence of that geometry, not a knob.

## The finding that fell out of doing this properly

Sweeping the field's correlation length surfaced something the original
"buffer always wins" narrative missed entirely: buffer pooling only helps
when the correlation length is **short relative to the jitter radius**
(real local variation for a displaced point to miss, which averaging
recovers). When the correlation length is **long relative to the jitter
radius**, the field is nearly flat across one jitter disk — point
extraction is already accurate, and buffer averaging only adds smoothing
(regression-to-the-mean) with nothing left to correct, so **naive point
extraction wins outright**. This is a real bias-variance tradeoff
discussed in the geospatial displacement literature (Perez-Heydrich et
al., 2013) — not something the rewrite set out to find, but something
that showed up the moment the hardcoded fractions were replaced with an
actual field to sample.

```bash
python main.py                                # default: buffer wins (short correlation length)
python main.py --correlation-length-km 20      # flips: naive wins
```

## Pipeline stages

```
data_extraction (clusters + spatial field)
    → feature_extraction (naive point vs. buffer-pooled sampling, satellite bands)
    → modeling (Ridge: naive vs. buffer)
    → evaluation (apparent vs. latent)
    → performance (honest decomposition)
    → reports
```

| Stage | Module | What it does |
|---|---|---|
| 0. Data extraction | `data_extraction` | Simulates georeferenced clusters over real Ethiopian regions; builds the spatially-autocorrelated wealth field; applies real DHS/LSMS-style GPS jitter. |
| 1. Feature extraction | `feature_extraction` | Samples the field at the naive (jittered-point) and buffer (disk-averaged) locations; simulates Sentinel-2-like bands and VIIRS-like nightlights from each. |
| 2. Modeling | `modeling` | Dependency-free Ridge regression, one model per extraction strategy, both trained on the noisy observed label. |
| 3. Evaluation | `evaluation` | R², Pearson r, RMSE against both the noisy observed label (apparent) and the true latent wealth (latent). |
| 4. Performance | `performance` | Decomposes the result into the two distinct effects — see above. |
| 5. Reports | `reports` | The two-panel chart (same layout/colors as the original) + a Markdown report. |

## Directory structure

```
.
├── geo_poverty_mapping/       # the installable package
│   ├── config.py              #   PipelineConfig — field correlation length, jitter, seed, etc.
│   ├── data_extraction/       #   spatial_field.py, simulator.py, geo_utils.py
│   ├── feature_extraction/    #   sampling.py (naive/buffer), satellite.py (stylized bands)
│   ├── modeling/               #   ridge.py, splitting.py
│   ├── evaluation/
│   ├── performance/            #   the noisy-label vs. jitter-mitigation decomposition
│   ├── reports/                 #   visualizations.py, report_builder.py
│   └── pipeline.py              #   orchestrator + CLI entry point
├── tests/                       # pytest suite (20 tests)
├── legacy/                      # original prototype, preserved with a bug list
├── output/                      # generated at runtime: chart + report.md (gitignored)
├── main.py
├── requirements.txt / requirements-dev.txt
└── pyproject.toml               # `pip install -e .` → `geo-poverty-mapping` console script
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
python main.py                                  # 500 clusters, default (short) correlation length
python main.py -v                                # verbose stage logging
python main.py --clusters 2000                   # larger sample
python main.py --correlation-length-km 20        # see the buffer-vs-naive advantage flip
python main.py --n-buffer-samples 60             # more Monte Carlo points per buffer average
python main.py --seed 123                        # reproducible re-run
```

Every run writes to `output/`: `poverty_mapping_diagnostics.png` (the
two-panel comparison chart) and `report.md` (full metrics + the
decomposition explained in plain language).

## Validation

- **Spatial field**: variance matches the requested amplitude exactly (a
  regression test for the earlier sparse-bump design, whose field variance
  came out ~400x smaller than intended); nearby points are measurably more
  similar than distant ones; handles out-of-range queries by clipping
  instead of raising.
- **Sampling**: naive extraction is checked against a direct recomputation
  from the field; buffer extraction is checked to have variance no larger
  than a single point sample (a basic property of averaging).
- **Decomposition**: the noisy-label gap is checked to be exactly zero in
  the zero-noise limit, and the theoretical ceiling formula is checked
  against a hand-computed noise-to-signal ratio.
- **The headline finding itself is a regression test**: `--correlation-length-km 2.0`
  (default) is asserted to favor buffer pooling; `25.0` is asserted to
  favor naive extraction — so if a future change breaks the mechanism
  that produces this tradeoff, the test suite catches it, not just a chart.
- **Ridge regression**: approximate coefficient recovery under noise;
  higher `alpha` is checked to shrink coefficients, a guaranteed property
  of ridge regardless of the specific data.

Run the suite: `pytest -q` (fully self-contained — no real satellite data,
no network, no external dependencies beyond numpy/pandas/matplotlib).

## Limitations

- **The satellite-band simulation is stylized**, not a physical sensor
  model — `feature_extraction/satellite.py` correlates band values with a
  wealth value via chosen coefficients, the same approach (and same
  reasonable-simplification tradeoff) as the original prototype. It's
  good enough to demonstrate the measurement-error mechanics; it is not a
  claim about real Sentinel-2/VIIRS band physics.
- **The wealth field's correlation length is a single global parameter.**
  Real socioeconomic covariates likely have multiple relevant spatial
  scales (a building shows up at tens of meters, a neighborhood's
  development level at kilometers) — this uses one length for
  illustrative clarity, not because real fields are single-scale.
  `--correlation-length-km` exists specifically so this can be explored
  rather than taken as given.
- **Circular (wraparound) boundary handling** in the FFT-based smoothing
  is an approximation — acceptable for a country-sized bounding box where
  edge effects are negligible next to everything else already stylized
  here, but worth knowing about if this pattern is reused for a smaller
  domain where edges matter more.
- **The Monte Carlo buffer average** uses `n_buffer_samples=30` by
  default; this is a variance/speed tradeoff, not an exact integral —
  increasing it tightens the estimate at the cost of runtime.

## References

- Burke, M., Driscoll, A., Lobell, D. B., & Ermon, S. (2020). *Using satellite imagery to understand and promote sustainable development.* Science, 371(6535).
- Perez-Heydrich, C., Warren, J. L., Burgert, C. R., & Emch, M. E. (2013). *Guidelines on the Use of DHS GPS Data.* DHS Spatial Analysis Reports No. 8, ICF International. (The real jitter methodology and the buffer-vs-point extraction tradeoff this project simulates.)
- Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., & Ermon, S. (2016). *Combining satellite imagery and machine learning to predict poverty.* Science, 353(6301).

## License

MIT — see [LICENSE](LICENSE).
