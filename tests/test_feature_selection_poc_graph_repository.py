from feature_selection_poc.data import make_regression_poc_dataset
from feature_selection_poc.graph import FeatureGraphBuilder, LocalFeatureGraphRepository
from feature_selection_poc.kernel.strategies import FeatureQualityFilter, ScorebookBuilder
from feature_selection_poc.config import QualityFilterConfig, ScoringConfig


def test_local_graph_repository_queries_neighbors_and_family(tmp_path):
    X, y, metadata = make_regression_poc_dataset(n_samples=80, n_features=8, random_state=3)
    _, metrics = FeatureQualityFilter(QualityFilterConfig()).filter(X, y)
    scorebook = ScorebookBuilder(ScoringConfig()).build(X, y, metrics)
    nodes, edges, family = FeatureGraphBuilder(corr_threshold=0.8).build(X, scorebook, metadata)
    repo = LocalFeatureGraphRepository(tmp_path)
    repo.save_nodes(nodes)
    repo.save_edges(edges)
    members = repo.query_family_members(family.iloc[0]["family_id"])
    assert not members.empty
    if not edges.empty:
        neighbors = repo.query_feature_neighbors(edges.iloc[0]["source"], max_depth=1)
        assert not neighbors.empty
