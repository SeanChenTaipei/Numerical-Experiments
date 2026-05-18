"""Quality filters and scorebook construction for high-dimensional regression features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from feature_selection_poc.config import QualityFilterConfig, ScoringConfig


def _numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=[np.number]).copy()


def minmax(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    lo, hi = float(s.min()), float(s.max())
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class FeatureQualityFilter:
    """Computes feature health metrics and marks low-quality columns for removal."""

    def __init__(self, config: QualityFilterConfig) -> None:
        self.config = config

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        Xn = _numeric_frame(X)
        rows: list[dict[str, object]] = []
        y_numeric = pd.Series(y).astype(float)
        for col in Xn.columns:
            s = Xn[col]
            missing = float(s.isna().mean())
            nunique = int(s.nunique(dropna=True))
            non_null = max(int(s.notna().sum()), 1)
            unique_ratio = float(nunique / non_null)
            top_freq = float(s.value_counts(normalize=True, dropna=True).iloc[0]) if nunique else 1.0
            variance = float(s.var(skipna=True) or 0.0)
            corr = float(s.corr(y_numeric)) if variance > 0 else 0.0
            if not np.isfinite(corr):
                corr = 0.0
            reason = ""
            if nunique <= 1 or top_freq >= self.config.near_constant_threshold:
                reason = "constant_or_near_constant"
            elif missing > self.config.missing_rate_threshold:
                reason = "high_missing_rate"
            elif unique_ratio < self.config.unique_ratio_min or unique_ratio > self.config.unique_ratio_max:
                reason = "unique_ratio_out_of_range"
            elif variance <= self.config.variance_threshold:
                reason = "low_variance"
            elif abs(corr) < self.config.min_abs_target_corr:
                reason = "low_target_correlation"
            rows.append(
                {
                    "feature_name": col,
                    "missing_rate": missing,
                    "unique_ratio": unique_ratio,
                    "variance": variance,
                    "target_correlation": corr,
                    "top_frequency": top_freq,
                    "quality_removed": bool(reason),
                    "removal_reason": reason,
                }
            )
        return pd.DataFrame(rows)

    def filter(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        metrics = self.evaluate(X, y)
        keep = metrics.loc[~metrics["quality_removed"], "feature_name"].tolist()
        if self.config.max_features_after_quality and len(keep) > self.config.max_features_after_quality:
            rank = metrics[metrics["feature_name"].isin(keep)].assign(abs_corr=lambda d: d.target_correlation.abs())
            keep = rank.nlargest(self.config.max_features_after_quality, "abs_corr")["feature_name"].tolist()
            metrics.loc[~metrics["feature_name"].isin(keep) & ~metrics["quality_removed"], "quality_removed"] = True
            metrics.loc[~metrics["feature_name"].isin(keep) & (metrics["removal_reason"] == ""), "removal_reason"] = "quality_top_k_cap"
        return X[keep].copy(), metrics


class ScorebookBuilder:
    """Builds a domain-readable multi-angle feature scorebook."""

    def __init__(self, scoring_config: ScoringConfig) -> None:
        self.config = scoring_config

    def build(self, X: pd.DataFrame, y: pd.Series, quality_metrics: pd.DataFrame) -> pd.DataFrame:
        scorebook = quality_metrics.copy()
        feature_names = scorebook["feature_name"].tolist()
        X_fill = X.reindex(columns=feature_names).select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        mi = pd.Series(0.0, index=feature_names)
        if len(feature_names) and len(X_fill):
            try:
                mi_values = mutual_info_regression(X_fill, y, random_state=42)
                mi = pd.Series(mi_values, index=X_fill.columns).reindex(feature_names).fillna(0.0)
            except Exception:
                mi = pd.Series(0.0, index=feature_names)
        tail_mask = y >= y.quantile(self.config.tail_quantile)
        tail_corr = {}
        for col in feature_names:
            if col in X.columns and tail_mask.sum() > 2:
                val = X.loc[tail_mask, col].corr(y.loc[tail_mask])
                tail_corr[col] = 0.0 if not np.isfinite(val) else abs(float(val))
            else:
                tail_corr[col] = 0.0
        scorebook["mutual_info"] = scorebook["feature_name"].map(mi).fillna(0.0)
        scorebook["missing_penalty"] = scorebook["missing_rate"].clip(0, 1)
        scorebook["unique_ratio_score"] = 1 - (scorebook["unique_ratio"] - 0.5).abs().clip(0, 0.5) * 2
        scorebook["variance_score"] = minmax(np.log1p(scorebook["variance"].clip(lower=0)))
        scorebook["target_correlation_score"] = scorebook["target_correlation"].abs().fillna(0.0)
        scorebook["mutual_info_score"] = minmax(scorebook["mutual_info"])
        scorebook["univariate_contribution_score"] = minmax(scorebook["target_correlation_score"] + scorebook["mutual_info_score"])
        scorebook["quality_score"] = (1 - scorebook["missing_penalty"]) * 0.5 + scorebook["unique_ratio_score"] * 0.25 + scorebook["variance_score"] * 0.25
        scorebook["model_importance_score"] = 0.0
        scorebook["shap_attribution_score"] = 0.0
        scorebook["tail_importance_score"] = scorebook["feature_name"].map(tail_corr).fillna(0.0)
        scorebook["outlier_sensitivity_score"] = scorebook["tail_importance_score"]
        scorebook["stability_score"] = 0.0
        scorebook["feature_family_representative_score"] = 0.0
        scorebook["redundancy_score"] = 0.0
        scorebook["domain_rescue_score"] = scorebook["tail_importance_score"] * scorebook["missing_penalty"].rsub(1)
        scorebook["status"] = np.where(scorebook["quality_removed"], "removed", "candidate")
        scorebook["selection_round"] = np.where(scorebook["quality_removed"], "quality", "candidate")
        scorebook["family_id"] = "unassigned"
        final = pd.Series(0.0, index=scorebook.index)
        for col, weight in self.config.final_score_weights.items():
            if col in scorebook:
                final += weight * minmax(scorebook[col])
        scorebook["final_feature_score"] = final
        return scorebook.sort_values("final_feature_score", ascending=False).reset_index(drop=True)
