from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ICCResult:
    icc_table: pd.DataFrame
    group_summary: pd.DataFrame
    residual_frame: pd.DataFrame
    default_metric: str


class ICCAnalyzer:
    """Compute ICC metrics and supporting tables."""

    def __init__(self, cfg: Dict[str, float]):
        self.cfg = cfg

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        y_col: str,
        yhat_col: str,
        group_col: str,
        time_col: str,
        use_residuals: bool = False,
    ) -> ICCResult:
        eval_df = df[[time_col, group_col, y_col]].copy()
        eval_df["value"] = eval_df[y_col]
        if use_residuals and yhat_col in df:
            eval_df["value"] = df[y_col] - df[yhat_col]
        else:
            eval_df["value"] = df[y_col]
        eval_df["yhat"] = df.get(yhat_col, np.nan)
        icc_table = _compute_icc(eval_df, value_col="value", group_col=group_col, time_col=time_col)
        grouped = (
            eval_df.groupby(group_col)
            .agg(
                mean_value=("value", "mean"),
                std_value=("value", "std"),
                count=("value", "count"),
            )
            .reset_index()
        )
        grouped["ci95"] = 1.96 * grouped["std_value"] / np.sqrt(grouped["count"].clip(lower=1))
        residual_frame = eval_df[[time_col, group_col, "value", "yhat"]].rename(
            columns={time_col: "timestamp", "value": "metric"}
        )
        default_metric = self.cfg.get("default_metric", "ICC2")
        return ICCResult(
            icc_table=icc_table,
            group_summary=grouped,
            residual_frame=residual_frame,
            default_metric=default_metric,
        )


def _compute_icc(df: pd.DataFrame, *, value_col: str, group_col: str, time_col: str) -> pd.DataFrame:
    try:
        import pingouin as pg

        tidy = df[[group_col, time_col, value_col]].dropna()
        tidy = tidy.rename(columns={group_col: "targets", time_col: "raters", value_col: "ratings"})
        tbl = pg.intraclass_corr(data=tidy, targets="targets", raters="raters", ratings="ratings")
        return tbl
    except Exception:
        return _manual_icc(df, value_col=value_col, group_col=group_col)


def _manual_icc(df: pd.DataFrame, *, value_col: str, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col)
    grand_mean = df[value_col].mean()
    n_groups = grouped.ngroups
    n_obs = len(df)
    group_sizes = grouped.size()
    ss_between = np.sum(group_sizes * (grouped[value_col].mean() - grand_mean) ** 2)
    ss_within = np.sum(grouped.apply(lambda g: ((g[value_col] - g[value_col].mean()) ** 2).sum()))
    ms_between = ss_between / max(n_groups - 1, 1)
    ms_within = ss_within / max(n_obs - n_groups, 1)
    n_bar = group_sizes.mean()
    icc1 = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within + 1e-6)
    icc2 = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within) if ms_between > ms_within else 0.0
    icc3 = (ms_between - ms_within) / ms_between if ms_between else 0.0
    return pd.DataFrame(
        {
            "Type": ["ICC1", "ICC2", "ICC3"],
            "ICC": [icc1, icc2, icc3],
            "Description": [
                "Single-rater absolute agreement",
                "Average-rater random effects",
                "Average-rater consistency",
            ],
        }
    )
