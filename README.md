# TherasyncAnalysis

A pipeline for analyzing physiological synchrony in family therapy sessions. Processes wearable sensor data (Empatica E4), computes dyadic synchrony via Poincare-based Inter-Centroid Distance (ICD), and correlates it with therapeutic alliance annotations.

## Overview

TherasyncAnalysis links physiological signals to therapeutic process through three stages:

1. **Preprocessing** -- Load, clean, and extract metrics from BVP, EDA, HR, and TEMP signals
2. **DPPA (Dyadic Physiological Profile Analysis)** -- Compute Poincare centroids per epoch, then Inter-Centroid Distance between dyad pairs (real and pseudo)
3. **Alliance correlation** -- Merge ICD time series with coded Moments of Interest (MOI) annotations and test whether alliance states predict synchrony

All outputs follow [BIDS](https://bids.neuroimaging.io/) conventions.

## Project structure

```
TherasyncAnalysis/
├── src/
│   ├── core/               # Config loader, BIDS utilities, logging
│   ├── physio/
│   │   ├── preprocessing/  # BVP, EDA, HR, TEMP pipelines (load → clean → metrics → BIDS)
│   │   ├── dppa/           # Poincare centroids, ICD calculation, dyad config
│   │   └── epoching/       # Fixed, n-split, and sliding window epoch assignment
│   ├── alliance/           # MOI loading/epoching, alliance-ICD analysis and plotting
│   └── visualization/      # Signal, HRV, EDA, TEMP, and composite plotters
├── scripts/
│   ├── batch/              # run_all_preprocessing, run_all_visualizations
│   ├── physio/             # Per-modality preprocessing and DPPA scripts
│   ├── alliance/           # MOI validation, epoching, visualization
│   ├── analysis/           # Statistics, quality reports, outlier investigation
│   └── utils/              # Cleanup, docs generation
├── config/                 # config.yaml + dyad pair mappings
├── tests/                  # 192 tests across 11 files
└── data/
    ├── raw/                # Original Empatica E4 outputs (BIDS layout)
    └── derivatives/        # All processed outputs
```

## Setup

Requires Python >= 3.8 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

To include dev dependencies (testing, linting):

```bash
uv sync --all-extras
```

## Usage

### Full pipeline

```bash
# 1. Preprocess all subjects and modalities
uv run python scripts/batch/run_all_preprocessing.py

# 2. Generate visualizations
uv run python scripts/batch/run_all_visualizations.py

# 3. Compute Poincare centroids
uv run python scripts/physio/dppa/compute_poincare.py --batch

# 4. Compute DPPA (inter-session + intra-family)
uv run python scripts/physio/dppa/compute_dppa.py --mode both --batch

# 5. ICD statistics
uv run python scripts/physio/dppa/analyze_icd_statistics.py --all

# 6. Alliance-ICD analysis
uv run python scripts/analysis/run_alliance_icd_analysis.py

# 7. Reports
uv run python scripts/analysis/compute_preprocessing_stats.py
uv run python scripts/analysis/generate_quality_report.py
uv run python scripts/analysis/investigate_outliers.py
uv run python scripts/analysis/count_valid_dyads.py
```

### Single subject preprocessing

```bash
uv run python scripts/physio/preprocessing/preprocess_bvp.py --subject g01p01 --session ses-01
uv run python scripts/physio/preprocessing/preprocess_eda.py --subject g01p01 --session ses-01
uv run python scripts/physio/preprocessing/preprocess_hr.py  --subject g01p01 --session ses-01
uv run python scripts/physio/preprocessing/preprocess_temp.py --subject g01p01 --session ses-01
```

## Configuration

All parameters live in `config/config.yaml`:

- **Paths** -- raw data, derivatives, logs
- **Modality settings** -- sampling rates, cleaning thresholds, metric extraction
- **Epoching** -- fixed (30s/5s overlap), n-split (120 epochs), sliding (30s/5s step)
- **DPPA** -- dyad pair definitions, ICD computation settings
- **Visualization** -- figure sizes, colormaps, output formats

## Testing

```bash
uv run pytest                                  # All tests
uv run pytest tests/test_bvp_pipeline.py -v    # Single file
uv run pytest tests/test_dppa.py::test_fn      # Single test
```

## Code quality

```bash
uv run ruff check .    # Lint
uv run ruff format .   # Format
```

## Key concepts

**DPPA** -- Each participant's RR intervals are plotted on a Poincare diagram (RR_n vs RR_{n+1}). The centroid of each epoch's cloud summarizes that epoch's cardiac dynamics.

**ICD** -- The Euclidean distance between two participants' Poincare centroids at each epoch. Lower ICD = higher physiological synchrony.

**Real vs pseudo dyads** -- Real dyads are family members recorded in the same session. Pseudo dyads pair participants from different sessions as a baseline comparison.

**Alliance states** -- Coded from video by trained annotators as neutral (0), positive (1), negative (-1), or split (2). These map onto ICD epochs to test whether alliance quality predicts synchrony.

## Data format

Input data follows BIDS layout:

```
data/raw/sub-{id}/ses-{id}/{modality}/{files}.csv
```

Derivatives mirror this structure under `data/derivatives/` with additional directories for preprocessing outputs, DPPA results, and alliance analyses.

## Authors

- Guillaume Dumas
- Lena Adel
- Remy Ramadour

## License

MIT
