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

