from timeseries_lab.data_io import generate_synthetic_dataset
from timeseries_lab.drift import DriftAnalyzer


def test_drift_metrics_presence():
    df = generate_synthetic_dataset(rows=600).df_data
    analyzer = DriftAnalyzer({"thresholds": {"psi": 0.1, "ks": 0.1}})
    report = analyzer.compare(
        reference=df.iloc[:300],
        comparison=df.iloc[300:600],
        features=["runtime_hours", "temperature"],
        label="train_vs_test",
    )
    assert {"feature", "psi", "risk_level"}.issubset(report.table.columns)
    assert len(report.table) == 2
