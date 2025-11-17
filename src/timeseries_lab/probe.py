from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import pandas as pd
import plotly.express as px


@dataclass(slots=True)
class ProbeResult:
    title: str
    summary_md: str
    figs: List[Any]
    tables: Dict[str, pd.DataFrame]
    tags: List[str]


class ProbeRegistry:
    """Simple decorator-based registry for probe functions."""

    registry: Dict[str, Callable[..., ProbeResult]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable[..., ProbeResult]):
            cls.registry[name] = func
            return func

        return decorator

    @classmethod
    def run(cls, name: str, **kwargs) -> ProbeResult:
        if name not in cls.registry:
            raise KeyError(f"Probe {name} not registered")
        return cls.registry[name](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls.registry.keys())


@ProbeRegistry.register("residual_by_time")
def probe_residual_by_time(df: pd.DataFrame, *, time_col: str, residual_col: str, bins: str = "7D") -> ProbeResult:
    grouped = (
        df.set_index(time_col)[residual_col]
        .resample(bins)
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig = px.line(grouped, x=time_col, y="mean", error_y="std", title="Residual mean by time bin")
    return ProbeResult(
        title="Residual by Time Bin",
        summary_md="觀察不同時間窗的殘差平均與變異以檢測時間漂移。",
        figs=[fig],
        tables={"residual_time": grouped},
        tags=["residual", "time"],
    )


@ProbeRegistry.register("residual_by_group")
def probe_residual_by_group(df: pd.DataFrame, *, group_col: str, residual_col: str, top_n: int = 10) -> ProbeResult:
    grouped = (
        df.groupby(group_col)[residual_col]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.bar(grouped, x=group_col, y="mean", error_y="std", title="Top residual groups")
    return ProbeResult(
        title="Residual by Group",
        summary_md="挑選殘差偏移最大的群組求進一步檢查。",
        figs=[fig],
        tables={"residual_group": grouped},
        tags=["residual", "group"],
    )


@ProbeRegistry.register("importance_stability")
def probe_importance_stability(feature_importance: pd.Series, permutation: pd.Series, jaccard: float) -> ProbeResult:
    fi = pd.DataFrame(
        {
            "feature": feature_importance.index,
            "gain_importance": feature_importance.values,
            "perm_importance": permutation.reindex(feature_importance.index).values,
        }
    )
    fig = px.scatter(
        fi,
        x="gain_importance",
        y="perm_importance",
        text="feature",
        title=f"Importance Alignment (Top-K Jaccard={jaccard:.2f})",
    )
    return ProbeResult(
        title="重要度穩定性",
        summary_md=f"特徵重要度一致性（Top-K Jaccard={jaccard:.2f}）。",
        figs=[fig],
        tables={"importance_alignment": fi},
        tags=["importance", "stability"],
    )


@ProbeRegistry.register("drift_importance_matrix")
def probe_drift_importance_matrix(drift_table: pd.DataFrame, importance: pd.Series) -> ProbeResult:
    merged = drift_table.merge(
        importance.rename("importance"),
        left_on="feature",
        right_index=True,
        how="left",
    )
    fig = px.scatter(
        merged,
        x="importance",
        y="psi",
        color="risk_level",
        hover_name="feature",
        title="High drift & impact matrix",
    )
    merged["risk_flag"] = merged["risk_level"].eq("high") & merged["importance"].gt(0)
    return ProbeResult(
        title="高 Drift × 高重要度",
        summary_md="同時具有高影響與高漂移的特徵需優先處理。",
        figs=[fig],
        tables={"drift_importance": merged},
        tags=["drift", "importance"],
    )


@ProbeRegistry.register("icc_group_stability")
def probe_icc_group_stability(group_summary: pd.DataFrame, icc_table: pd.DataFrame) -> ProbeResult:
    fig = px.bar(group_summary, x=group_summary.columns[0], y="mean_value", error_y="ci95", title="Group means ± CI")
    icc_row = icc_table.sort_values("ICC", ascending=False).head(1).iloc[0]
    summary = f"預設 ICC 指標 {icc_row.get('Type', '')}: {icc_row.get('ICC', 0):.3f}"
    return ProbeResult(
        title="ICC Group Stability",
        summary_md=summary,
        figs=[fig],
        tables={"icc_table": icc_table, "group_summary": group_summary},
        tags=["icc", "group"],
    )


@ProbeRegistry.register("target_feature_curve")
def probe_target_feature_curve(df: pd.DataFrame, feature: str, target: str) -> ProbeResult:
    sampled = df[[feature, target]].dropna()
    fig = px.scatter(sampled, x=feature, y=target, trendline="lowess", title=f"{target} vs {feature}")
    return ProbeResult(
        title="特徵關聯曲線",
        summary_md="檢查目標與指定特徵之局部單調性與相關性。",
        figs=[fig],
        tables={f"{feature}_curve": sampled},
        tags=["eda", "feature-insight"],
    )
