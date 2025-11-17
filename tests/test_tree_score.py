from timeseries_lab.data_io import generate_synthetic_dataset
from timeseries_lab.tree_score import TreeScoreAnalyzer


def test_tree_score_generates_table():
    df = generate_synthetic_dataset(rows=400).df_data
    features = ["runtime_hours", "temperature", "pressure"]
    X_train = df.iloc[:200][features]
    y_train = df.iloc[:200]["target_main"]
    X_val = df.iloc[200:300][features]
    y_val = df.iloc[200:300]["target_main"]
    analyzer = TreeScoreAnalyzer({"max_depth": 2, "min_samples_leaf": 10})
    result = analyzer.score(X_train, y_train, X_val, y_val)
    assert not result.table.empty
    assert set(["feature", "delta_mse", "r2_single"]).issubset(result.table.columns)
