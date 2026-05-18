"""In-memory NetworkX graph repository."""
from __future__ import annotations

import pandas as pd


class NetworkXFeatureGraphRepository:
    def __init__(self) -> None:
        import networkx as nx
        self.nx = nx
        self.graph = nx.Graph()
        self._nodes = pd.DataFrame()
        self._edges = pd.DataFrame()

    def save_nodes(self, nodes: pd.DataFrame) -> None:
        self._nodes = nodes.copy()
        for _, row in nodes.iterrows():
            self.graph.add_node(row["name"], **row.to_dict())

    def save_edges(self, edges: pd.DataFrame) -> None:
        self._edges = edges.copy()
        for _, row in edges.iterrows():
            self.graph.add_edge(row["source"], row["target"], **row.to_dict())

    def load_nodes(self) -> pd.DataFrame: return self._nodes.copy()
    def load_edges(self) -> pd.DataFrame: return self._edges.copy()

    def query_feature_neighbors(self, feature_name: str, max_depth: int = 2) -> pd.DataFrame:
        if feature_name not in self.graph:
            return pd.DataFrame(columns=["feature_name", "depth"])
        lengths = self.nx.single_source_shortest_path_length(self.graph, feature_name, cutoff=max_depth)
        return pd.DataFrame([{"feature_name": k, "depth": v} for k, v in lengths.items() if k != feature_name])

    def query_family_members(self, family_id: str) -> pd.DataFrame:
        nodes = self.load_nodes()
        return nodes[nodes["family_id"].eq(family_id)].copy() if "family_id" in nodes else pd.DataFrame()

    def sync(self) -> None: pass
