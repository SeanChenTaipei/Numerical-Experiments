from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .settings import resolve_artifact_path

SAVED_FIGURES: Dict[str, Path] = {}


def save_fig(fig: go.Figure, name: str) -> Path:
    """Persist a plotly figure as HTML for downstream consumption."""

    path = resolve_artifact_path("plots", f"{name}.html")
    fig.write_html(str(path))
    SAVED_FIGURES[name] = path
    return path


def plot_time_series(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: str = "") -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, title=title)
    save_fig(fig, f"time_series_{y}")
    return fig


def plot_distribution(df: pd.DataFrame, column: str, split_col: str, title: str = "") -> go.Figure:
    fig = px.histogram(df, x=column, color=split_col, marginal="box", nbins=40, title=title, opacity=0.6)
    save_fig(fig, f"distribution_{column}")
    return fig


def plot_box(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: str = "") -> go.Figure:
    fig = px.box(df, x=x, y=y, color=color, title=title)
    save_fig(fig, f"box_{x}_{y}")
    return fig


def plot_tree_score(table: pd.DataFrame, metric: str, top_k: int = 15) -> go.Figure:
    subset = table.head(top_k)
    fig = px.bar(subset, x="feature", y=metric, color="high_drift_flag", title=f"Tree Score - {metric}")
    save_fig(fig, f"tree_score_{metric}")
    return fig


def plot_radar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for feature, row in df.iterrows():
        fig.add_trace(
            go.Scatterpolar(r=row.values.tolist() + [row.values[0]], theta=list(df.columns) + [df.columns[0]], name=feature)
        )
    fig.update_layout(title="Tree Score Radar", polar=dict(radialaxis=dict(visible=True)))
    save_fig(fig, "tree_score_radar")
    return fig


def plot_icc_bars(icc_table: pd.DataFrame) -> go.Figure:
    fig = px.bar(icc_table, x="Type", y="ICC", color="ICC", title="ICC Summary")
    save_fig(fig, "icc_summary")
    return fig


def plot_drift_table(drift_table: pd.DataFrame, metric: str = "psi") -> go.Figure:
    fig = px.bar(
        drift_table.sort_values(metric, ascending=False).head(20),
        x="feature",
        y=metric,
        color="risk_level",
        title=f"Drift ranking ({metric})",
    )
    save_fig(fig, f"drift_{metric}")
    return fig


def _bootstrap_demo_plots() -> None:
    if SAVED_FIGURES:
        return
    from .data_io import generate_synthetic_dataset

    bundle = generate_synthetic_dataset(rows=500, groups=5)
    df = bundle.df_data.copy()
    df["split"] = pd.cut(
        df.index,
        bins=[-1, int(len(df) * 0.6), int(len(df) * 0.8), len(df)],
        labels=["train", "val", "test"],
    )
    daily = df.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()
    plot_time_series(daily, x="timestamp", y="target_main", title="Synthetic target trend")
    plot_distribution(df, "runtime_hours", "split", "Runtime hours distribution")
    plot_distribution(df, "temperature", "split", "Temperature distribution")
    plot_box(df, "split", "target_main", title="Target by split")

    sample_table = pd.DataFrame(
        {
            "feature": ["runtime_hours", "temperature", "pressure", "vibration", "operator_te"],
            "delta_mse": [0.5, 0.4, 0.2, 0.1, 0.05],
            "r2_single": [0.6, 0.55, 0.22, 0.1, 0.08],
            "split_gain": [0.3, 0.25, 0.12, 0.08, 0.02],
            "high_drift_flag": [True, True, False, False, False],
        }
    )
    plot_tree_score(sample_table, "delta_mse")
    plot_tree_score(sample_table, "r2_single")
    plot_tree_score(sample_table, "split_gain")
    plot_radar(sample_table.set_index("feature")[["delta_mse", "r2_single", "split_gain"]])
    icc_tbl = pd.DataFrame({"Type": ["ICC1", "ICC2", "ICC3"], "ICC": [0.7, 0.82, 0.85]})
    plot_icc_bars(icc_tbl)
    drift_tbl = pd.DataFrame(
        {
            "feature": ["runtime_hours", "temperature", "pressure", "vibration", "operator_te"],
            "psi": [0.3, 0.2, 0.1, 0.05, 0.01],
            "risk_level": ["high", "medium", "low", "low", "low"],
        }
    )
    plot_drift_table(drift_tbl, "psi")


try:
    _bootstrap_demo_plots()
except Exception:
    pass
