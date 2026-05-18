"""Neo4j repository skeleton kept behind the graph repository abstraction."""
from __future__ import annotations

import pandas as pd

from . import cypher_templates
from .graph_schema import NEO4J_CONSTRAINTS


class Neo4jFeatureGraphRepository:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("Install neo4j to use Neo4jFeatureGraphRepository") from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._nodes = pd.DataFrame()
        self._edges = pd.DataFrame()

    def _execute(self, query: str, **params):
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, **params))

    def save_nodes(self, nodes: pd.DataFrame) -> None:
        self._nodes = nodes.copy()
        query = """
        UNWIND $rows AS row
        MERGE (f:Feature {name: row.name})
        SET f += row
        """
        self._execute(query, rows=nodes.where(pd.notnull(nodes), None).to_dict("records"))

    def save_edges(self, edges: pd.DataFrame) -> None:
        self._edges = edges.copy()
        query = """
        UNWIND $rows AS row
        MATCH (a:Feature {name: row.source}), (b:Feature {name: row.target})
        MERGE (a)-[r:CORRELATED_WITH]-(b)
        SET r.weight = row.weight, r.correlation = row.correlation, r.family_method = row.family_method
        """
        self._execute(query, rows=edges.where(pd.notnull(edges), None).to_dict("records"))

    def load_nodes(self) -> pd.DataFrame: return self._nodes.copy()
    def load_edges(self) -> pd.DataFrame: return self._edges.copy()

    def query_feature_neighbors(self, feature_name: str, max_depth: int = 2) -> pd.DataFrame:
        records = self._execute(cypher_templates.FEATURE_NEIGHBORS, feature_name=feature_name, max_depth=max_depth)
        return pd.DataFrame([dict(r) for r in records])

    def query_family_members(self, family_id: str) -> pd.DataFrame:
        query = "MATCH (f:Feature {family_id: $family_id}) RETURN f.name AS feature_name, f AS properties"
        return pd.DataFrame([dict(r) for r in self._execute(query, family_id=family_id)])

    def sync(self) -> None:
        for constraint in NEO4J_CONSTRAINTS:
            self._execute(constraint)
