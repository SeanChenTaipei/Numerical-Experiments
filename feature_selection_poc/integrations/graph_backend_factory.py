"""Factory for graph backend adapters."""
from __future__ import annotations

from feature_selection_poc.config import GraphBackendConfig
from feature_selection_poc.graph import LocalFeatureGraphRepository, Neo4jFeatureGraphRepository, NetworkXFeatureGraphRepository


def create_graph_repository(config: GraphBackendConfig):
    backend = config.backend.lower()
    if backend == "local":
        return LocalFeatureGraphRepository(config.output_dir)
    if backend == "networkx":
        return NetworkXFeatureGraphRepository()
    if backend == "neo4j":
        if not (config.neo4j_uri and config.neo4j_user and config.neo4j_password):
            raise ValueError("Neo4j backend requires uri, user, and password")
        return Neo4jFeatureGraphRepository(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    raise ValueError(f"Unsupported graph backend: {config.backend}")
