from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass(slots=True)
class PreprocessArtifacts:
    numeric_cols: List[str]
    categorical_cols: List[str]
    dropped_cols: List[str]
    target_name: str


class Preprocessor:
    """Handle imputing, encoding, scaling, and filtering."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = RobustScaler()
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.dropped_cols: List[str] = []
        self.target_encoder_: Dict[str, Dict[str, float]] = {}
        self.global_target_mean: float = 0.0
        self.artifacts: PreprocessArtifacts | None = None

    def fit(self, df: pd.DataFrame, target_col: str) -> PreprocessArtifacts:
        """Fit preprocessing assets on training dataframe."""

        self.global_target_mean = float(df[target_col].mean())
        features = df.drop(columns=[target_col])
        self.numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

        self._filter_missing(features)
        self._filter_constant(features)
        self._filter_vif(features)

        if self.categorical_cols:
            self._fit_target_encoders(df, target_col)

        if self.numeric_cols:
            self.scaler.fit(df[self.numeric_cols].fillna(df[self.numeric_cols].median()))

        self.artifacts = PreprocessArtifacts(
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            dropped_cols=self.dropped_cols,
            target_name=target_col,
        )
        return self.artifacts

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe using fitted assets."""

        if not self.artifacts:
            raise RuntimeError("Call fit before transform")

        numeric = df[self.numeric_cols].copy() if self.numeric_cols else pd.DataFrame(index=df.index)
        if not numeric.empty:
            numeric = numeric.fillna(numeric.median())
            numeric.loc[:, :] = self.scaler.transform(numeric)

        cat_frames = []
        for col in self.categorical_cols:
            encoded = df[col].fillna("__MISSING__").astype(str).map(self.target_encoder_.get(col, {}))
            encoded = encoded.fillna(self.global_target_mean)
            cat_frames.append(encoded.rename(f"{col}_te"))
        categorical = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=df.index)

        return pd.concat([numeric, categorical], axis=1)

    # Internal helpers -----------------------------------------------------

    def _filter_missing(self, features: pd.DataFrame) -> None:
        threshold = self.config.get("missing_ratio_threshold", 0.5)
        to_drop = [col for col in features.columns if features[col].isna().mean() > threshold]
        self._drop_columns(to_drop)

    def _filter_constant(self, features: pd.DataFrame) -> None:
        to_drop = [col for col in features.columns if features[col].nunique(dropna=False) <= 1]
        self._drop_columns(to_drop)

    def _filter_vif(self, features: pd.DataFrame) -> None:
        vif_threshold = self.config.get("vif_threshold", 15.0)
        cols = list(self.numeric_cols)
        if len(cols) <= 1:
            return
        working = features[cols].dropna()
        if working.empty:
            return
        while True:
            vifs = {
                col: variance_inflation_factor(working.values, idx)
                for idx, col in enumerate(working.columns)
            }
            max_col = max(vifs, key=vifs.get)
            if vifs[max_col] <= vif_threshold:
                break
            self.dropped_cols.append(max_col)
            cols.remove(max_col)
            if len(cols) <= 1:
                break
            working = working[cols]
        self.numeric_cols = cols

    def _drop_columns(self, cols: List[str]) -> None:
        if not cols:
            return
        self.dropped_cols.extend(cols)
        self.numeric_cols = [c for c in self.numeric_cols if c not in cols]
        self.categorical_cols = [c for c in self.categorical_cols if c not in cols]

    def _fit_target_encoders(self, df: pd.DataFrame, target_col: str) -> None:
        folds = self.config.get("encoding_folds", 5)
        time_col = self.config.get("timestamp_col")
        ordered_df = df.sort_values(time_col) if time_col in df.columns else df
        kfold = KFold(n_splits=folds, shuffle=False)
        for col in self.categorical_cols:
            encoded = pd.Series(index=ordered_df.index, dtype=float)
            for train_idx, val_idx in kfold.split(ordered_df):
                train_subset = ordered_df.iloc[train_idx]
                mapping = train_subset.groupby(col)[target_col].mean()
                encoded.iloc[val_idx] = ordered_df.iloc[val_idx][col].map(mapping)
            encoded = encoded.fillna(self.global_target_mean)
            prep = (
                ordered_df.assign(
                    __cat=ordered_df[col].fillna("__MISSING__").astype(str),
                    __enc=encoded,
                )
                .groupby("__cat")["__enc"]
                .mean()
                .to_dict()
            )
            self.target_encoder_[col] = prep
