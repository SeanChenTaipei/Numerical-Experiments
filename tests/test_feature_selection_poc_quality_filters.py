import pandas as pd

from feature_selection_poc.config import QualityFilterConfig, ScoringConfig
from feature_selection_poc.data import make_regression_poc_dataset
from feature_selection_poc.kernel.strategies import FeatureQualityFilter, ScorebookBuilder


def test_quality_filter_removes_near_constant_feature():
    X, y, _ = make_regression_poc_dataset(n_samples=80, n_features=8, random_state=1)
    _, metrics = FeatureQualityFilter(QualityFilterConfig()).filter(X, y)
    row = metrics[metrics["feature_name"].eq("near_constant")].iloc[0]
    assert row["quality_removed"]
    assert row["removal_reason"] == "constant_or_near_constant"


def test_scorebook_contains_required_scores():
    X, y, _ = make_regression_poc_dataset(n_samples=80, n_features=8, random_state=2)
    _, metrics = FeatureQualityFilter(QualityFilterConfig()).filter(X, y)
    scorebook = ScorebookBuilder(ScoringConfig()).build(X, y, metrics)
    assert {"final_feature_score", "tail_importance_score", "domain_rescue_score"}.issubset(scorebook.columns)
    assert scorebook["feature_name"].is_unique
