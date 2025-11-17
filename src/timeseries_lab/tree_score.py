from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


@dataclass(slots=True)
class TreeScoreResult:
    table: pd.DataFrame
    split_details: Dict[str, pd.DataFrame]
    radar_payload: pd.DataFrame


class TreeScoreAnalyzer:
    """Evaluate per-feature predictive strength via shallow trees."""

    def __init__(self, cfg: Dict[str, float]):
        self.cfg = cfg

    def score(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        drift_table: Optional[pd.DataFrame] = None,
    ) -> TreeScoreResult:
        max_depth = self.cfg.get("max_depth", 3)
        min_leaf = self.cfg.get("min_samples_leaf", 30)
        baseline = mean_squared_error(y_val, np.full_like(y_val, y_train.mean()))
        rows = []
        split_details: Dict[str, pd.DataFrame] = {}
        for feature in X_train.columns:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=42,
            )
            model.fit(X_train[[feature]], y_train)
            preds = model.predict(X_val[[feature]])
            mse = mean_squared_error(y_val, preds)
            delta = baseline - mse
            r2 = r2_score(y_val, preds)
            gain = model.tree_.impurity[0] - np.sum(model.tree_.impurity[model.tree_.children_left != -1])
            rows.append(
                {
                    "feature": feature,
                    "delta_mse": float(delta),
                    "r2_single": float(r2),
                    "split_gain": float(gain),
                }
            )
            details = pd.DataFrame(
                {
                    "thresholds": model.tree_.threshold,
                    "impurity": model.tree_.impurity,
                    "n_node_samples": model.tree_.n_node_samples,
                }
            )
            split_details[feature] = details

        table = pd.DataFrame(rows).sort_values("delta_mse", ascending=False)
        if drift_table is not None and not drift_table.empty:
            high_drift = set(drift_table[drift_table["risk_level"] == "high"]["feature"])
            table["high_drift_flag"] = table["feature"].isin(high_drift)
        else:
            table["high_drift_flag"] = False
        radar_payload = table.set_index("feature")[["delta_mse", "r2_single", "split_gain"]]
        return TreeScoreResult(table=table, split_details=split_details, radar_payload=radar_payload)
