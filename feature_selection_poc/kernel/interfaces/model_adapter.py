"""Unified model adapter protocol."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class RegressionModelAdapter(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> "RegressionModelAdapter": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def feature_importance(self, feature_names: list[str]) -> pd.Series: ...
