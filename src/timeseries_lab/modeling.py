from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .utils import top_k_jaccard


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(median_absolute_error(y_true, y_pred))


METRIC_FNS = {"rmse": rmse, "mae": mae, "r2": r2, "medae": medae}


class BaseModelTrainer:
    """Interface for concrete trainers."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit")
        return self.model.predict(X)

    def feature_importances(self) -> pd.Series:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=self.model.feature_name_)
        return pd.Series(dtype=float)


class LightGBMTrainer(BaseModelTrainer):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from lightgbm import LGBMRegressor

        params = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": -1, **self.params}
        self.model = LGBMRegressor(**params)
        self.model.fit(X, y)


class CatBoostTrainer(BaseModelTrainer):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from catboost import CatBoostRegressor

        params = {
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "verbose": False,
            **self.params,
        }
        self.model = CatBoostRegressor(**params)
        self.model.fit(X, y)


class EBMTrainer(BaseModelTrainer):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from interpret.glassbox import ExplainableBoostingRegressor

        params = {"learning_rate": 0.05, "max_leaves": 5, "interactions": 0, **self.params}
        self.model = ExplainableBoostingRegressor(**params)
        self.model.fit(X, y)


class ModelTrainerFactory:
    registry = {
        "lightgbm": LightGBMTrainer,
        "catboost": CatBoostTrainer,
        "ebm": EBMTrainer,
    }

    @classmethod
    def create(cls, name: str, params: Dict[str, Any]) -> BaseModelTrainer:
        trainer_cls = cls.registry.get(name.lower())
        if not trainer_cls:
            raise ValueError(f"Unknown model {name}")
        return trainer_cls(params=params)


@dataclass(slots=True)
class ModelingResult:
    cv_metrics: List[Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    feature_importance: pd.Series
    permutation_importance: pd.Series
    topk_stability: float
    oof_predictions: pd.Series
    final_model: BaseModelTrainer


def run_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cfg: Dict[str, Any],
    model_name: str,
    model_params: Dict[str, Any],
) -> ModelingResult:
    """Execute expanding window cross-validation."""

    n_splits = cfg.get("n_folds", 4)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_per_fold = []
    importances = []
    permutation_imps = []
    top_features = []
    oof = pd.Series(index=y.index, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        trainer = ModelTrainerFactory.create(model_name, model_params)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        trainer.fit(X_train, y_train)
        preds = trainer.predict(X_val)
        fold_metrics = {name: fn(y_val, preds) for name, fn in METRIC_FNS.items()}
        metrics_per_fold.append(fold_metrics)
        oof.iloc[val_idx] = preds

        fi = getattr(trainer.model, "feature_importances_", None)
        if fi is not None:
            importances.append(pd.Series(fi, index=X.columns))
            top_features.append(list(pd.Series(fi, index=X.columns).sort_values(ascending=False).index))
        p_imp = permutation_importance(trainer.model, X_val, y_val, n_repeats=5, random_state=42)
        permutation_imps.append(pd.Series(p_imp.importances_mean, index=X.columns))

    agg_metrics = {name: float(np.mean([m[name] for m in metrics_per_fold])) for name in METRIC_FNS.keys()}
    mean_fi = (
        pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        if importances
        else pd.Series(dtype=float)
    )
    mean_perm = (
        pd.concat(permutation_imps, axis=1).mean(axis=1).sort_values(ascending=False)
        if permutation_imps
        else pd.Series(dtype=float)
    )
    k = cfg.get("stability_top_k", 10)
    topk_stability = top_k_jaccard(top_features, k) if top_features else 0.0

    final_trainer = ModelTrainerFactory.create(model_name, model_params)
    final_trainer.fit(X, y)

    return ModelingResult(
        cv_metrics=metrics_per_fold,
        aggregate_metrics=agg_metrics,
        feature_importance=mean_fi,
        permutation_importance=mean_perm,
        topk_stability=topk_stability,
        oof_predictions=oof,
        final_model=final_trainer,
    )
