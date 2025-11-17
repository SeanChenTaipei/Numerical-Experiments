# Temporal Group Regression Lab

A fully-operational Python + Streamlit toolkit for tabular regression problems that simultaneously exhibit **temporal structure** (timestamps) and **group structure** (machines, batches, SKUs). The app loads CSV/Feather files (or auto-generates realistic synthetic data), performs time-aware preprocessing, builds LightGBM/CatBoost/EBM models, evaluates drift & ICC stability, scores single features with shallow decision trees, and exposes an extensible probe-based EDA playbook.

## Key Capabilities

- **Unified configuration** via configs/default.yaml + optional overrides. Centralizes timestamp/group/factor columns, preprocessing strategy, drift & ICC settings, and Streamlit defaults.
- **Factory-driven architecture** (DataLoaderFactory, ModelTrainerFactory, ProbeRegistry) wrapped under src/timeseries_lab/ modules (data_io, split, preprocess, drift, icc, 	ree_score, probe, iz, utils).
- **Caching everywhere it matters**: Streamlit @st.cache_data/@st.cache_resource guard dataset loading, splits, preprocessing, drift tables, ICC summaries, tree scores, and probe payloads (cache keys include hashes of config + data).
- **Temporal splits & coverage reports** supporting ratio/date modes plus group coverage sanity checks.
- **Modeling suite** with expanding-window CV, RMSE/MAE/R2/MedAE metrics, permutation importances, and Top-K Jaccard stability tracking.
- **Comprehensive monitoring**: PSI/KS/CVM/JS/TV/Wasserstein drift metrics, ICC(1/2/3) stability, residual heatmaps, and Tree Score radar/bars. High-drift/high-impact features are auto-flagged.
- **Probe-based EDA Playbook** (6 pre-built probes) returning titles, markdown summaries, tables, Plotly figs, and tags that can be exported to Markdown/HTML reports.
- **Artifacts**: 10+ Plotly visualizations auto-saved under rtifacts/plots/ on import, plus report downloads (JSON + Markdown) from the Streamlit UI.

## Project Layout

`
app.py                              # Streamlit entrypoint with 8 tabs + sidebar controls
configs/default.yaml                # Central configuration (overridable via configs/experiment.yaml)
src/timeseries_lab/
  ¢u¢w¢w data_io.py                    # File loaders + synthetic generator
  ¢u¢w¢w split.py                      # TimeSplitService + SplitResult dataclass
  ¢u¢w¢w preprocess.py                 # Target encoding, VIF filtering, RobustScaler pipeline
  ¢u¢w¢w modeling.py                   # ModelTrainerFactory + expanding-window CV orchestration
  ¢u¢w¢w drift.py                      # DriftAnalyzer for PSI/KS/CVM/Wasserstein/JS/TV
  ¢u¢w¢w icc.py                        # ICCAnalyzer with pingouin fallback + summaries
  ¢u¢w¢w tree_score.py                 # Shallow decision-tree feature scoring
  ¢u¢w¢w probe.py                      # Probe registry/decorator + 6 built-in probes
  ¢u¢w¢w viz.py                        # Plotly helpers + artifact bootstrapper (10+ plots saved)
  ¢u¢w¢w utils.py                      # Hashing, sampling, coverage helpers
  ¢|¢w¢w settings.py                   # Config loading facade + artifact path helpers
artifacts/plots/                    # Auto-generated Plotly HTML figures
requirements.txt                    # Version-bounded dependencies (Streamlit, Plotly, LGBM, CatBoost, EBM, etc.)
tests/                              # Split/Drift/ICC/Tree Score unit tests
`

## Getting Started

1. **Environment**

   `ash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   `

2. **Run the Streamlit Lab**

   `ash
   streamlit run app.py
   `

   - Use the sidebar to upload df_data / df_factors or rely on the built-in synthetic generator.
   - Select target/group/timestamp columns, tweak split mode (ratio/date), drift thresholds, model choice, ICC/drift settings, and probe parameters.
   - Every heavy computation is cached; use the ¡§Clear cache¡¨ button if you change upstream files/configs.

3. **Artifacts & Reports**

   - All charts have associated download buttons; HTML copies also live under rtifacts/plots/.
   - The ¡§Reports & Artifacts¡¨ tab exports a consolidated JSON + Markdown summary (metrics, drift alerts, ICC status, probe highlights).

4. **Unit Tests**

   `ash
   pytest
   `

## Configuration Notes

- Override defaults by editing configs/experiment.yaml or pointing EXPERIMENT_CONFIG to another YAML.
- Key sections:
  - data.defaults: timestamp/group/target/factor column names.
  - preprocess: target encoding folds, VIF threshold, missingness cap.
  - modeling: per-model hyperparameters + CV settings.
  - drift / icc / 	ree_score: risk thresholds and scoring knobs.
  - probes: binning cadence + Top-N controls for the EDA playbook.

## Generating Synthetic Data on Disk

Run the helper below to persist the default synthetic dataset (optional):

`ash
python - <<"PY"
from pathlib import Path
from timeseries_lab.data_io import DataLoaderFactory
bundle = DataLoaderFactory.fallback_synthetic(rows=5000, groups=12)
bundle.df_data.to_csv("data/synthetic_data.csv", index=False)
bundle.df_factors.to_csv("data/synthetic_factors.csv", index=False)
PY
`

## Testing & Quality

- 	ests/test_split.py: verifies chronological ratio splitting & coverage integrity.
- 	ests/test_drift.py: ensures drift metrics & risk annotations are present.
- 	ests/test_icc.py: validates ICC table construction.
- 	ests/test_tree_score.py: checks shallow-tree scoring outputs.

## Design Patterns & Extensibility

- **Factory Pattern**: DataLoaderFactory picks CSV/Feather/Synthetic loaders; ModelTrainerFactory instantiates LightGBM/CatBoost/EBM trainers.
- **Registry Pattern**: ProbeRegistry uses decorators to register probes, making it trivial to add new investigative recipes.
- **Dependency Injection**: Config dictionaries flow through each service (split, preprocessing, drift, ICC, tree score) enabling reproducible overrides and straightforward unit tests.
- **Caching Strategy**: hash_pandas_frame + config hashes guarantee cache keys reflect both data and settings; clearing caches is a single click from the UI.

Happy analyzing! ??
