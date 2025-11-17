from timeseries_lab.data_io import generate_synthetic_dataset
from timeseries_lab.icc import ICCAnalyzer


def test_icc_result_not_empty():
    df = generate_synthetic_dataset(rows=500).df_data
    df["pred"] = df["target_main"] * 0.9
    analyzer = ICCAnalyzer({"default_metric": "ICC2"})
    result = analyzer.evaluate(
        df,
        y_col="target_main",
        yhat_col="pred",
        group_col="machine_id",
        time_col="timestamp",
        use_residuals=False,
    )
    assert not result.icc_table.empty
    assert "ICC" in result.icc_table.columns
