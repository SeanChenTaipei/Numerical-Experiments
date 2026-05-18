"""Local file implementation of FeatureGraphRepository."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class LocalFeatureGraphRepository:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: pd.DataFrame | None = None
        self.edges: pd.DataFrame | None = None

    def save_nodes(self, nodes: pd.DataFrame) -> None:
        self.nodes = nodes.copy()
        nodes.to_csv(self.output_dir / "feature_graph_nodes.csv", index=False)

    def save_edges(self, edges: pd.DataFrame) -> None:
        self.edges = edges.copy()
        edges.to_csv(self.output_dir / "feature_graph_edges.csv", index=False)

    def load_nodes(self) -> pd.DataFrame:
        if self.nodes is not None:
            return self.nodes.copy()
        path = self.output_dir / "feature_graph_nodes.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def load_edges(self) -> pd.DataFrame:
        if self.edges is not None:
            return self.edges.copy()
        path = self.output_dir / "feature_graph_edges.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def query_feature_neighbors(self, feature_name: str, max_depth: int = 2) -> pd.DataFrame:
        edges = self.load_edges()
        if edges.empty:
            return pd.DataFrame(columns=["feature_name", "depth", "via"])
        seen = {feature_name}
        frontier = {feature_name}
        rows = []
        for depth in range(1, max_depth + 1):
            next_frontier = set()
            mask = edges["source"].isin(frontier) | edges["target"].isin(frontier)
            for _, e in edges[mask].iterrows():
                for node in [e["source"], e["target"]]:
                    if node not in seen:
                        seen.add(node); next_frontier.add(node)
                        rows.append({"feature_name": node, "depth": depth, "via": e["edge_type"]})
            frontier = next_frontier
        return pd.DataFrame(rows)

    def query_family_members(self, family_id: str) -> pd.DataFrame:
        nodes = self.load_nodes()
        if nodes.empty or "family_id" not in nodes:
            return pd.DataFrame()
        return nodes[nodes["family_id"].eq(family_id)].copy()

    def sync(self) -> None:
        if self.nodes is not None:
            self.save_nodes(self.nodes)
        if self.edges is not None:
            self.save_edges(self.edges)
