"""Neo4j-oriented feature graph schema and Cypher examples."""
NODE_LABELS = ["Feature", "FeatureFamily", "DataSource", "Station", "ProcessStep", "ModelRun", "SelectionRound"]
RELATIONSHIP_TYPES = [
    "BELONGS_TO_FAMILY", "FROM_DATA_SOURCE", "MEASURED_AT_STATION", "IN_PROCESS_STEP",
    "CORRELATED_WITH", "SELECTED_IN", "REMOVED_IN", "RESCUED_IN", "IMPORTANT_IN_MODEL",
]
NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT feature_name IF NOT EXISTS FOR (f:Feature) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT family_id IF NOT EXISTS FOR (ff:FeatureFamily) REQUIRE ff.family_id IS UNIQUE",
]
