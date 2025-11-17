import pandas as pd

from timeseries_lab.data_io import generate_synthetic_dataset
from timeseries_lab.split import TimeSplitService


def test_time_split_ratio_sizes():
    bundle = generate_synthetic_dataset(rows=720)
    df = bundle.df_data
    service = TimeSplitService(
        timestamp_col="timestamp",
        mode="by_ratio",
        cfg={"ratios": {"train": 0.5, "val": 0.3, "test": 0.2}, "group_col": "machine_id"},
    )
    result = service.split(df)
    assert len(result.train) + len(result.val) + len(result.test) == len(df)
    assert len(result.train) > len(result.val) > 0
