"""Model-importance pruning strategies."""
from __future__ import annotations

import pandas as pd

from feature_selection_poc.config import PruningConfig


class IterativeImportancePruningStrategy:
    """Coarse-to-fine pruning by scorebook/model importance columns."""

    def __init__(self, config: PruningConfig) -> None:
        self.config = config

    def choose_features(self, scorebook: pd.DataFrame, candidate_features: list[str], round_index: int) -> list[str]:
        candidates = scorebook[scorebook["feature_name"].isin(candidate_features)].copy()
        score_col = "model_importance_score" if candidates["model_importance_score"].sum() > 0 else "final_feature_score"
        target_count = max(self.config.min_features, min(self.config.keep_top_k, int(len(candidates) * self.config.keep_fraction)))
        if round_index >= self.config.max_rounds - 1:
            target_count = max(self.config.min_features, min(self.config.keep_top_k, len(candidates)))
        return candidates.nlargest(target_count, score_col)["feature_name"].tolist()
