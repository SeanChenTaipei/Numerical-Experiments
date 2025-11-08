# Machine Learning Pipeline Starter

This repository provides a lightweight, opinionated starting point for applied machine learning research. It includes:

- Structured source layout under `src/` with data ingestion, preprocessing, model definition, and training orchestration modules.
- Configuration-first workflow using YAML files for experiment reproducibility.
- Experiment tracking stubs and testing harness to keep research iterations reliable.

## Getting Started

1. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Train a model with the default configuration**

   ```bash
   python -m src.pipeline.train --config configs/default.yaml
   ```

4. **Run the unit test suite**

   ```bash
   pytest
   ```

## Repository Layout

```
├── configs/            # Experiment and data settings
├── data/               # Local datasets (ignored by git)
├── notebooks/          # Research notebooks
├── src/                # Reusable pipeline code
├── tests/              # Unit tests for the pipeline
└── requirements.txt    # Python dependencies
```

## Configuration System

Configurations are stored as YAML files inside `configs/` and loaded via `OmegaConf`. Each configuration file should define the dataset, preprocessing choices, model hyperparameters, and training options. Copy `configs/default.yaml` to create new experiments.

## Next Steps

- Add new datasets by extending `src/pipeline/data.py`.
- Introduce experiment logging (e.g., MLflow, Weights & Biases) via the hook in `src/pipeline/train.py`.
- Replace the baseline scikit-learn model with bespoke architectures or AutoML tools as needed.

## MixedGAM Utilities

### Ray Tune Auto Search

`MixedGAMRegressor` now defers to Ray Tune when `auto_search=True`. You can let the built-in search space run or provide a Ray-compatible `search_space` dictionary (e.g., `{"learning_rate": tune.loguniform(...), "n_epochs": tune.choice([...])}`). The search summary exposes the best config/metric, while the actual model instance is updated in-place with the chosen hyper-parameters.

### Validation Splits, Callbacks, and Progress

The `fit` signature accepts optional validation arrays or a `val_ratio`:

```python
model.fit(
    X_train,
    y_train,
    groups=group_labels,
    X_val=X_valid,
    y_val=y_valid,
    groups_val=g_valid,
    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss")],
)
```

If no validation data is supplied, the trainer performs a stratified split using `val_ratio` (default `0.2`). During training the Lightning progress bar is always visible and now highlights the current mode, running train/val losses, and optimizer learning rate so long runs stay informative.

### Inspecting Feature Contributions

`predict_and_contrib` returns both predictions and a dictionary of additive pieces:

```python
preds, contribs = model.predict_and_contrib(X_test, groups=g_test)
base = contribs["base_per_feature"]      # per-feature global trunk output
bias = contribs["bias"]                  # broadcast scalar bias

if model.mode == "group_residual":
    residual = contribs["residual_per_feature"]
    per_feature = contribs["per_feature"]    # base + residual per feature
    residual_sum = contribs["residual_sum"]  # per-sample adapter total
else:  # group_affine
    before_affine = contribs["base_per_feature"]
    after_affine = contribs["per_feature"]   # values after affine adapters
```

Use these arrays to plot shape functions, audit per-group adjustments, or feed into downstream explanation tooling.
