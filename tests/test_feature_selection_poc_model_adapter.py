from feature_selection_poc.config import ModelConfig
from feature_selection_poc.data import make_regression_poc_dataset
from feature_selection_poc.models import create_model_adapter


def test_model_adapter_fit_predict_importance():
    X, y, _ = make_regression_poc_dataset(n_samples=60, n_features=6, random_state=5)
    adapter = create_model_adapter(ModelConfig(name="random_forest", params={"n_estimators": 5, "random_state": 5}))
    adapter.fit(X, y)
    preds = adapter.predict(X)
    imp = adapter.feature_importance(list(X.columns))
    assert len(preds) == len(y)
    assert set(imp.index) == set(X.columns)
