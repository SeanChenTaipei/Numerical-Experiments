from feature_selection_poc.config import FeatureSelectionConfig, ModelConfig, PruningConfig
from feature_selection_poc.data import make_regression_poc_dataset
from feature_selection_poc.pipeline import FeatureSelectionPipeline


def test_pipeline_fit_select_and_save_artifacts(tmp_path):
    X, y, metadata = make_regression_poc_dataset(n_samples=90, n_features=10, random_state=4)
    config = FeatureSelectionConfig(
        model=ModelConfig(name="random_forest", params={"n_estimators": 10, "random_state": 4}, cv_folds=3),
        pruning=PruningConfig(max_rounds=2, keep_top_k=5, keep_fraction=0.6, min_features=3),
    )
    pipeline = FeatureSelectionPipeline(config)
    result = pipeline.fit_select(X, y, metadata)
    assert len(result.selected_features) >= 3
    assert not result.scorebook.empty
    pipeline.save_artifacts(result, tmp_path)
    assert (tmp_path / "selected_features.csv").exists()
    assert (tmp_path / "feature_graph_nodes.csv").exists()
