from feature_selection_poc.graph import cypher_templates
from feature_selection_poc.graph.graph_schema import NODE_LABELS, RELATIONSHIP_TYPES


def test_neo4j_schema_and_templates_are_available():
    assert "Feature" in NODE_LABELS
    assert "CORRELATED_WITH" in RELATIONSHIP_TYPES
    assert "MATCH" in cypher_templates.FEATURE_NEIGHBORS
