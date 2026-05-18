"""Reusable Cypher snippets for graph users and agents."""
FEATURE_NEIGHBORS = """
MATCH (f:Feature {name: $feature_name})-[r:CORRELATED_WITH*1..$max_depth]-(n:Feature)
RETURN DISTINCT n.name AS feature_name, length(r) AS depth
ORDER BY depth, feature_name
"""
STATION_SELECTED = """
MATCH (s:Station {name: $station})<-[:MEASURED_AT_STATION]-(f:Feature)
WHERE f.status IN ['selected', 'rescued']
RETURN f.name AS feature_name, f.final_feature_score AS score
ORDER BY score DESC
"""
FAMILY_REPRESENTATIVE = """
MATCH (:FeatureFamily {family_id: $family_id})<-[:BELONGS_TO_FAMILY]-(f:Feature)
RETURN f.name AS feature_name, f.final_feature_score AS score
ORDER BY score DESC LIMIT 1
"""
RESCUE_CANDIDATES = """
MATCH (f:Feature)
WHERE f.status = 'removed' AND coalesce(f.tail_importance_score, 0) >= $tail_threshold
RETURN f.name AS feature_name, f.tail_importance_score AS tail_score, f.removal_reason AS removal_reason
ORDER BY tail_score DESC
"""
MODEL_RUN_SELECTIONS = """
MATCH (m:ModelRun {run_id: $run_id})<-[:SELECTED_IN|REMOVED_IN|RESCUED_IN]-(f:Feature)
RETURN f.name AS feature_name, f.status AS status, f.selection_round AS selection_round
"""
COMPARE_RUNS = """
MATCH (f:Feature)-[r]->(m:ModelRun)
WHERE m.run_id IN [$left_run_id, $right_run_id]
RETURN f.name AS feature_name, collect({run: m.run_id, rel: type(r)}) AS run_membership
"""
