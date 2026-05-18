"""Tail-aware scoring utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_tail_sample_weight(y: pd.Series, tail_quantile: float = 0.9, tail_weight: float = 5.0) -> np.ndarray:
    threshold = y.quantile(tail_quantile)
    return np.where(y >= threshold, tail_weight, 1.0).astype(float)


def tail_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, tail_quantile: float = 0.9) -> dict[str, float]:
    y_series = pd.Series(y_true)
    mask = y_series >= y_series.quantile(tail_quantile)
    if mask.sum() == 0:
        return {"tail_rmse": 0.0, "tail_mae": 0.0, "tail_sample_count": 0.0}
    err_rmse = mean_squared_error(y_series[mask], y_pred[mask], squared=False)
    err_mae = mean_absolute_error(y_series[mask], y_pred[mask])
    return {"tail_rmse": float(err_rmse), "tail_mae": float(err_mae), "tail_sample_count": float(mask.sum())}
