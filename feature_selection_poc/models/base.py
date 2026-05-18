"""Model adapter factory and baseline implementations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from feature_selection_poc.config import ModelConfig


class SklearnRegressorAdapter:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> "SklearnRegressorAdapter":
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        values = getattr(self.estimator, "feature_importances_", np.zeros(len(feature_names)))
        s = pd.Series(values, index=feature_names, dtype=float)
        total = s.abs().sum()
        return s.abs() / total if total else s


def create_model_adapter(config: ModelConfig) -> SklearnRegressorAdapter:
    name = config.name.lower()
    params = dict(config.params)
    if name in {"lightgbm", "lgbm", "lightgbmregressor"}:
        try:
            from lightgbm import LGBMRegressor
            return SklearnRegressorAdapter(LGBMRegressor(**params))
        except Exception:
            pass
    if name in {"catboost", "catboostregressor"}:
        try:
            from catboost import CatBoostRegressor
            params.setdefault("verbose", False)
            return SklearnRegressorAdapter(CatBoostRegressor(**params))
        except Exception:
            pass
    if name in {"ebm", "explainable_boosting_machine"}:
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
            return SklearnRegressorAdapter(ExplainableBoostingRegressor(**params))
        except Exception:
            pass
    params.setdefault("random_state", 42)
    params.setdefault("n_estimators", 80)
    return SklearnRegressorAdapter(RandomForestRegressor(**params))
