"""Domain rescue candidate rules."""
from __future__ import annotations

import pandas as pd


def propose_rescue_candidates(scorebook: pd.DataFrame, tail_threshold: float = 0.4, limit: int = 10) -> list[str]:
    removed = scorebook[scorebook["status"].eq("removed")].copy()
    if removed.empty:
        return []
    return removed[removed["tail_importance_score"] >= tail_threshold].nlargest(limit, "domain_rescue_score")["feature_name"].tolist()
