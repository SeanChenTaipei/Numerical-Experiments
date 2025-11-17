from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import pandas as pd

from .utils import compute_group_coverage, hash_pandas_frame


@dataclass(slots=True)
class SplitResult:
    """Structured output of the time split service."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    coverage: pd.DataFrame
    meta: Dict[str, str]


class TimeSplitService:
    """Split dataframe into chronological train/val/test segments."""

    def __init__(self, *, timestamp_col: str, mode: Literal["by_ratio", "by_date"], cfg: Dict[str, Any]):
        self.timestamp_col = timestamp_col
        self.mode = mode
        self.cfg = cfg

    def split(self, df: pd.DataFrame) -> SplitResult:
        """Split dataframe according to configured mode."""

        df_sorted = df.sort_values(self.timestamp_col).reset_index(drop=True)
        if self.mode == "by_date":
            train, val, test = self._split_by_date(df_sorted)
        else:
            train, val, test = self._split_by_ratio(df_sorted)

        coverage = compute_group_coverage(
            train,
            val,
            test,
            group_col=self.cfg.get("group_col", "group"),
        )
        meta = {
            "mode": self.mode,
            "train_rows": str(len(train)),
            "val_rows": str(len(val)),
            "test_rows": str(len(test)),
            "hash": hash_pandas_frame(df_sorted),
        }
        return SplitResult(train=train, val=val, test=test, coverage=coverage, meta=meta)

    def _split_by_ratio(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ratios = self.cfg.get("ratios", {"train": 0.6, "val": 0.2, "test": 0.2})
        n = len(df)
        train_end = int(n * ratios["train"])
        val_end = train_end + int(n * ratios["val"])
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        return train, val, test

    def _split_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dates = self.cfg.get("dates", {})
        ts_col = self.timestamp_col
        val_start = pd.to_datetime(dates.get("val_start"))
        test_start = pd.to_datetime(dates.get("test_start"))
        train = df[df[ts_col] < val_start]
        val = df[(df[ts_col] >= val_start) & (df[ts_col] < test_start)]
        test = df[df[ts_col] >= test_start]
        return train, val, test
