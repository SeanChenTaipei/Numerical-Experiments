"""Model construction utilities."""

from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig, PreprocessingConfig, TrainingConfig


def build_model(
    model_cfg: ModelConfig,
    preprocessing_cfg: PreprocessingConfig,
    training_cfg: TrainingConfig,
    feature_names: list[str],
) -> Pipeline:
    """Create a scikit-learn pipeline given model and preprocessing configs."""
    numeric_features = feature_names

    transformers = []
    if preprocessing_cfg.impute_strategy:
        transformers.append(
            ("imputer",
             SimpleImputer(strategy=preprocessing_cfg.impute_strategy)))
    if preprocessing_cfg.scale:
        transformers.append(("scaler", StandardScaler()))

    preprocessing_steps = transformers or "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[("numeric", Pipeline(preprocessing_steps),
                       numeric_features)],
        remainder="drop",
    )

    estimator = _create_estimator(model_cfg, training_cfg)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _create_estimator(model_cfg: ModelConfig,
                      training_cfg: TrainingConfig) -> BaseEstimator:
    params: Dict[str, object] = dict(model_cfg.params)
    if training_cfg.use_class_weight and "class_weight" not in params:
        params["class_weight"] = "balanced"

    if model_cfg.type == "logistic_regression":
        return LogisticRegression(**params)

    raise ValueError(
        f"Unsupported model type '{model_cfg.type}'. Extend `_create_estimator` to add more."
    )
