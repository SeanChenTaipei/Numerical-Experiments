"""Pipeline result entities."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .selection_history import SelectionRound


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    removed_features: list[str]
    scorebook: pd.DataFrame
    feature_metadata: pd.DataFrame
    feature_family: pd.DataFrame
    graph_nodes: pd.DataFrame
    graph_edges: pd.DataFrame
    selection_history: list[SelectionRound] = field(default_factory=list)
    model_evaluation: dict[str, float] = field(default_factory=dict)

    @property
    def feature_graph(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.graph_nodes, self.graph_edges
