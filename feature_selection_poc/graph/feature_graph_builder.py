"""Build feature families and graph node/edge tables."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


class FeatureGraphBuilder:
    def __init__(self, corr_threshold: float = 0.85, max_edges: int = 10000) -> None:
        self.corr_threshold = corr_threshold
        self.max_edges = max_edges

    def build(self, X: pd.DataFrame, scorebook: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features = scorebook["feature_name"].tolist()
        nodes = scorebook.merge(metadata, on="feature_name", how="left")
        nodes = nodes.rename(columns={"feature_name": "name"})
        nodes["node_type"] = "Feature"
        edges = self._correlation_edges(X.reindex(columns=[c for c in features if c in X.columns]))
        family = self._family_table(features, edges, nodes)
        nodes = nodes.merge(family[["feature_name", "family_id"]].rename(columns={"feature_name": "name"}), on="name", how="left", suffixes=("", "_graph"))
        nodes["family_id"] = nodes.get("family_id_graph", nodes.get("family_id", "unassigned")).fillna(nodes.get("family_id", "unassigned"))
        nodes = nodes.drop(columns=[c for c in ["family_id_graph"] if c in nodes.columns])
        return nodes, edges, family

    def _correlation_edges(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.empty or X.shape[1] < 2:
            return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "correlation", "family_method"])
        corr = X.corr(numeric_only=True).abs()
        rows = []
        for a, b in itertools.combinations(corr.columns, 2):
            value = corr.loc[a, b]
            if np.isfinite(value) and value >= self.corr_threshold:
                rows.append({"source": a, "target": b, "edge_type": "CORRELATED_WITH", "weight": float(value), "correlation": float(value), "family_method": "pearson_threshold"})
                if len(rows) >= self.max_edges:
                    break
        return pd.DataFrame(rows, columns=["source", "target", "edge_type", "weight", "correlation", "family_method"])

    def _family_table(self, features: list[str], edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        parent = {f: f for f in features}
        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a: str, b: str) -> None:
            if a in parent and b in parent:
                parent[find(b)] = find(a)
        for _, row in edges.iterrows():
            union(row["source"], row["target"])
        raw = {f: find(f) for f in features}
        family_ids = {root: f"corr_family_{i:04d}" for i, root in enumerate(sorted(set(raw.values())), 1)}
        df = pd.DataFrame({"feature_name": features, "family_id": [family_ids[raw[f]] for f in features]})
        score = nodes.rename(columns={"name": "feature_name"})[["feature_name", "final_feature_score"]]
        df = df.merge(score, on="feature_name", how="left")
        reps = df.sort_values("final_feature_score", ascending=False).groupby("family_id").first()["feature_name"].to_dict()
        sizes = df.groupby("family_id").size().to_dict()
        df["family_method"] = "correlation_components"
        df["representative_feature"] = df["family_id"].map(reps)
        df["family_size"] = df["family_id"].map(sizes).astype(int)
        return df.drop(columns=["final_feature_score"])
