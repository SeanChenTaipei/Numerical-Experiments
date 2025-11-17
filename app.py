from __future__ import annotations

import json
import copy
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf
import plotly.express as px

from timeseries_lab import (
    DataLoaderFactory,
    DriftAnalyzer,
    ICCAnalyzer,
    TreeScoreAnalyzer,
    TimeSplitService,
    load_config,
    viz,
)
from timeseries_lab.modeling import ModelTrainerFactory, run_time_series_cv, METRIC_FNS
from timeseries_lab.preprocess import Preprocessor
from timeseries_lab.probe import ProbeRegistry
from timeseries_lab.utils import hash_pandas_frame, stable_hash, maybe_sample_df


st.set_page_config(page_title="Temporal Regression Lab", layout="wide")


def _default_index(options: List[str], target: str) -> int:
    try:
        return options.index(target)
    except (ValueError, AttributeError):
        return 0


def figure_block(fig, name: str) -> None:
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        label=f"Download {name}",
        data=fig.to_html().encode("utf-8"),
        file_name=f"{name}.html",
        mime="text/html",
    )


@st.cache_data(show_spinner=False)
def load_sources(
    data_bytes: bytes | None,
    data_name: str | None,
    factor_bytes: bytes | None,
    factor_name: str | None,
    cfg_hash: str,
    synthetic_rows: int,
    synthetic_groups: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    factor_default = None
    if data_bytes:
        df_data = DataLoaderFactory.from_bytes(data_bytes, data_name or "uploaded.csv")
        source = "uploaded"
    else:
        bundle = DataLoaderFactory.fallback_synthetic(rows=synthetic_rows, groups=synthetic_groups)
        df_data = bundle.df_data
        factor_default = bundle.df_factors
        source = "synthetic"
    if factor_bytes:
        df_factors = DataLoaderFactory.from_bytes(factor_bytes, factor_name or "factors.csv")
    else:
        df_factors = factor_default if factor_default is not None else pd.DataFrame({"FactorName": ["target_main"]})
    meta = {"hash": hash_pandas_frame(df_data), "source": source}
    return df_data, df_factors, meta


@st.cache_data(show_spinner=False)
def cache_split(
    df: pd.DataFrame,
    timestamp_col: str,
    split_cfg: Dict,
    mode: str,
    group_col: str,
    cfg_hash: str,
):
    splitter = TimeSplitService(timestamp_col=timestamp_col, mode=mode, cfg={**split_cfg, "group_col": group_col})
    result = splitter.split(df)
    return result.train, result.val, result.test, result.coverage, result.meta


@st.cache_data(show_spinner=False)
def cache_preprocess(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    prep_cfg: Dict,
    cfg_hash: str,
):
    pre = Preprocessor(prep_cfg)
    train_subset = train_df[feature_cols + [target_col]]
    val_subset = val_df[feature_cols + [target_col]]
    test_subset = test_df[feature_cols + [target_col]]
    artifacts = pre.fit(train_subset, target_col)
    X_train = pre.transform(train_subset.drop(columns=[target_col]))
    X_val = pre.transform(val_subset.drop(columns=[target_col]))
    X_test = pre.transform(test_subset.drop(columns=[target_col]))
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    artifact_dict = artifacts.__dict__ if artifacts else {}
    return X_train, X_val, X_test, y_train, y_val, y_test, artifact_dict


@st.cache_data(show_spinner=False)
def cache_drift(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    features: List[str],
    drift_cfg: Dict,
    label: str,
    cfg_hash: str,
):
    analyzer = DriftAnalyzer(drift_cfg)
    report = analyzer.compare(reference=reference, comparison=comparison, features=features, label=label)
    return report.table, report.coverage


@st.cache_data(show_spinner=False)
def cache_icc(
    df: pd.DataFrame,
    icc_cfg: Dict,
    y_col: str,
    yhat_col: str,
    group_col: str,
    time_col: str,
    use_residuals: bool,
    cfg_hash: str,
):
    analyzer = ICCAnalyzer(icc_cfg)
    result = analyzer.evaluate(
        df,
        y_col=y_col,
        yhat_col=yhat_col,
        group_col=group_col,
        time_col=time_col,
        use_residuals=use_residuals,
    )
    return result.icc_table, result.group_summary, result.residual_frame, result.default_metric


@st.cache_data(show_spinner=False)
def cache_tree_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tree_cfg: Dict,
    drift_table: pd.DataFrame,
    cfg_hash: str,
):
    analyzer = TreeScoreAnalyzer(tree_cfg)
    result = analyzer.score(X_train, y_train, X_val, y_val, drift_table)
    return result.table, result.split_details, result.radar_payload


def main():
    settings = load_config()
    cfg = OmegaConf.to_container(settings.dump(), resolve=True)
    cfg_hash = stable_hash(cfg)
    if "probe_history" not in st.session_state:
        st.session_state["probe_history"] = []
    sidebar = st.sidebar
    sidebar.header("Settings")
    data_file = sidebar.file_uploader("Upload df_data (CSV/Feather)", type=["csv", "feather"])
    factor_file = sidebar.file_uploader("Upload df_factors", type=["csv", "feather"])
    default_rows = cfg["data"]["synthetic"].get("rows", 5000)
    default_groups = cfg["data"]["synthetic"].get("groups", 10)
    df_data, df_factors, meta = load_sources(
        data_file.getvalue() if data_file else None,
        data_file.name if data_file else None,
        factor_file.getvalue() if factor_file else None,
        factor_file.name if factor_file else None,
        cfg_hash,
        default_rows,
        default_groups,
    )
    targets = df_factors["FactorName"].tolist()
    target_col = sidebar.selectbox(
        "Target column",
        options=targets,
        index=_default_index(targets, cfg["data"]["defaults"]["target"]),
    )
    columns_list = df_data.columns.tolist()
    group_col = sidebar.selectbox(
        "Group column",
        options=columns_list,
        index=_default_index(columns_list, cfg["data"]["defaults"]["group"]),
    )
    timestamp_col = sidebar.selectbox(
        "Timestamp column",
        options=columns_list,
        index=_default_index(columns_list, cfg["data"]["defaults"]["timestamp"]),
    )
    df_data[timestamp_col] = pd.to_datetime(df_data[timestamp_col], errors="coerce")
    feature_pool = [col for col in columns_list if col not in {target_col, group_col, timestamp_col}]
    if not feature_pool:
        st.error("No usable features. Please upload data with additional columns.")
        st.stop()
    selected_features = sidebar.multiselect("Feature subset", feature_pool, default=feature_pool)
    if not selected_features:
        selected_features = feature_pool
    default_mode = cfg["data"]["split"].get("mode", "by_ratio")
    split_mode = sidebar.radio(
        "Split mode",
        options=["by_ratio", "by_date"],
        index=0 if default_mode == "by_ratio" else 1,
    )
    psi_slider = sidebar.slider(
        "PSI risk threshold",
        min_value=0.01,
        max_value=0.5,
        value=float(cfg["drift"]["thresholds"]["psi"]),
        step=0.01,
    )
    ks_slider = sidebar.slider(
        "KS risk threshold",
        min_value=0.01,
        max_value=0.5,
        value=float(cfg["drift"]["thresholds"]["ks"]),
        step=0.01,
    )
    if sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        sidebar.success("Cache cleared.")
    sidebar.success("Cache enabled (st.cache_data)")
    st.sidebar.markdown(f"**Dataset hash:** `{meta['hash'][:12]}` ({meta['source']})")
    st.sidebar.markdown(f"**Cached plots:** {len(viz.SAVED_FIGURES)}")

    split_cols = list(dict.fromkeys([timestamp_col, group_col, target_col] + selected_features))
    dataset_view = df_data[split_cols].copy()
    train_df, val_df, test_df, coverage, split_meta = cache_split(
        dataset_view,
        timestamp_col,
        cfg["data"]["split"],
        split_mode,
        group_col,
        cfg_hash,
    )
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        prep_artifacts,
    ) = cache_preprocess(train_df, val_df, test_df, target_col, selected_features, cfg["preprocess"], cfg_hash)

    tabs = st.tabs(
        [
            "Data Explorer",
            "Time Split & Prep",
            "Drift Lab",
            "Modeling",
            "ICC Studio",
            "Tree Score",
            "Investigation",
            "Reports & Artifacts",
        ]
    )

    with tabs[0]:
        st.subheader("Data Explorer")
        explorer_limit = cfg["data"]["explorer"]["max_rows"]
        sample = maybe_sample_df(df_data, explorer_limit)
        if sample.attrs.get("sampled"):
            st.info(f"Showing sampled subset of {len(sample)} rows (full={len(df_data)}).")
        st.dataframe(sample.head(500))
        st.metric("Rows", len(df_data))
        st.metric("Columns", len(df_data.columns))
        desc = sample[selected_features].describe().T
        st.dataframe(desc)

    with tabs[1]:
        st.subheader("Splits & Preprocess")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train rows", len(train_df))
        c2.metric("Val rows", len(val_df))
        c3.metric("Test rows", len(test_df))
        st.caption(f"Split hash: {split_meta['hash'][:12]}")
        st.dataframe(coverage)
        st.json(prep_artifacts)

    modeling_cfg = cfg["modeling"]
    model_name = st.sidebar.selectbox("Model", ["lightgbm", "catboost", "ebm"])
    params = modeling_cfg["models"].get(model_name, {})
    with tabs[3]:
        st.subheader("Modeling")
        result = run_time_series_cv(X_train, y_train, cfg=modeling_cfg["cv"], model_name=model_name, model_params=params)
        st.json(result.aggregate_metrics)
        fi_df = result.feature_importance.reset_index()
        fi_df.columns = ["feature", "importance"]
        fi_chart = px.bar(fi_df.head(30), x="feature", y="importance", title="Feature importance (avg across folds)")
        figure_block(fi_chart, "feature_importance")
        final_model = result.final_model
        val_holdout = {name: func(y_val, final_model.predict(X_val)) for name, func in METRIC_FNS.items()}
        test_holdout = {name: func(y_test, final_model.predict(X_test)) for name, func in METRIC_FNS.items()}
        st.metric("Holdout RMSE (val)", round(val_holdout["rmse"], 4))
        st.metric("Holdout RMSE (test)", round(test_holdout["rmse"], 4))
        st.json({"val": val_holdout, "test": test_holdout})

    drift_cfg = copy.deepcopy(cfg["drift"])
    drift_cfg["thresholds"]["psi"] = psi_slider
    drift_cfg["thresholds"]["ks"] = ks_slider
    drift_val, cov_val = cache_drift(
        train_df[selected_features],
        val_df[selected_features],
        selected_features,
        drift_cfg,
        "Train vs Val",
        cfg_hash,
    )
    drift_test, cov_test = cache_drift(
        train_df[selected_features],
        test_df[selected_features],
        selected_features,
        drift_cfg,
        "Train vs Test",
        cfg_hash,
    )

    with tabs[2]:
        st.subheader("Drift Lab")
        st.markdown("**Train vs Val**")
        st.dataframe(drift_val)
        figure_block(viz.plot_drift_table(drift_val), "drift_val")
        st.markdown("**Train vs Test**")
        st.dataframe(drift_test)
        figure_block(viz.plot_drift_table(drift_test), "drift_test")
        feature_choice = st.selectbox("Visualize feature drift", selected_features, key="drift_feature")
        combined = pd.concat(
            [
                train_df[[feature_choice]].assign(split="train"),
                val_df[[feature_choice]].assign(split="val"),
                test_df[[feature_choice]].assign(split="test"),
            ],
            axis=0,
        )
        dist_fig = viz.plot_distribution(combined, feature_choice, "split", f"{feature_choice} distribution by split")
        figure_block(dist_fig, f"drift_distribution_{feature_choice}")
        box_fig = viz.plot_box(combined, "split", feature_choice, title=f"{feature_choice} boxplot by split")
        figure_block(box_fig, f"drift_box_{feature_choice}")

    tree_cfg = cfg["tree_score"]
    tree_table, tree_details, radar_payload = cache_tree_score(
        X_train,
        y_train,
        X_val,
        y_val,
        tree_cfg,
        drift_val,
        cfg_hash,
    )

    with tabs[5]:
        st.subheader("Tree Score")
        st.dataframe(tree_table.head(50))
        figure_block(viz.plot_tree_score(tree_table, "delta_mse"), "tree_delta")
        figure_block(viz.plot_radar(radar_payload), "tree_radar_app")

    final_trainer = result.final_model
    val_preds = final_trainer.predict(X_val)
    test_preds = final_trainer.predict(X_test)
    val_eval = val_df.assign(pred=val_preds)
    val_eval["residual"] = val_eval[target_col] - val_eval["pred"]
    test_eval = test_df.assign(pred=test_preds)
    test_eval["residual"] = test_eval[target_col] - test_eval["pred"]
    icc_cfg = cfg["icc"]
    icc_val = cache_icc(val_eval, icc_cfg, target_col, "pred", group_col, timestamp_col, False, cfg_hash)

    with tabs[4]:
        st.subheader("ICC Studio")
        icc_table, group_summary, residual_frame, default_metric = icc_val
        st.dataframe(icc_table)
        figure_block(viz.plot_icc_bars(icc_table), "icc_studio")
        st.caption(f"Default ICC metric: {default_metric}. Assumes random effects per group and consistent timestamp cadence.")
        scatter_fig = px.scatter(
            val_eval,
            x=target_col,
            y="pred",
            color=group_col,
            title="y vs ŷ (colored by group)",
        )
        figure_block(scatter_fig, "icc_pred_vs_actual")
        resid_fig = px.line(
            val_eval.sort_values(timestamp_col),
            x=timestamp_col,
            y="residual",
            color=group_col,
            title="Residual vs time",
        )
        figure_block(resid_fig, "icc_residual_time")
        group_fig = px.bar(group_summary, x=group_col, y="mean_value", error_y="ci95", title="Group means ± CI")
        figure_block(group_fig, "icc_group_ci")

    with tabs[6]:
        st.subheader("Investigation")
        probe_choice = st.selectbox("Probe", ProbeRegistry.available())
        if probe_choice == "residual_by_time":
            payload = {
                "df": val_eval,
                "time_col": timestamp_col,
                "residual_col": "residual",
                "bins": cfg["probes"]["bins"],
            }
        elif probe_choice == "residual_by_group":
            payload = {
                "df": val_eval,
                "group_col": group_col,
                "residual_col": "residual",
                "top_n": cfg["probes"].get("top_groups", 10),
            }
        elif probe_choice == "importance_stability":
            payload = {
                "feature_importance": result.feature_importance,
                "permutation": result.permutation_importance,
                "jaccard": result.topk_stability,
            }
        elif probe_choice == "drift_importance_matrix":
            payload = {"drift_table": drift_val, "importance": result.feature_importance}
        elif probe_choice == "icc_group_stability":
            payload = {"group_summary": group_summary, "icc_table": icc_table}
        else:
            payload = {"df": df_data, "feature": selected_features[0], "target": target_col}
        probe_result = ProbeRegistry.run(probe_choice, **payload)
        st.markdown(probe_result.summary_md)
        for fig in probe_result.figs:
            figure_block(fig, f"probe_{probe_choice}")
        for name, table in probe_result.tables.items():
            st.dataframe(table)
        history = st.session_state.setdefault("probe_history", [])
        history.append({"title": probe_result.title, "summary": probe_result.summary_md, "tags": probe_result.tags})
        st.session_state["probe_history"] = history[-10:]

    with tabs[7]:
        st.subheader("Reports & Artifacts")
        probe_notes = st.session_state.get("probe_history", [])
        report_payload = {
            "config": cfg,
            "metrics": result.aggregate_metrics,
            "drift_risks": drift_val.to_dict("records"),
            "icc": icc_table.to_dict("records"),
            "probes": probe_notes,
        }
        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report_payload, indent=2).encode("utf-8"),
            file_name="report.json",
            mime="application/json",
        )
        md_lines = [
            "# Temporal Regression Lab Report",
            "## Metrics",
        ]
        for metric, value in result.aggregate_metrics.items():
            md_lines.append(f"- **{metric.upper()}**: {value:.4f}")
        md_lines.append("## Drift Highlights (Train→Val)")
        for row in drift_val.head(10).to_dict("records"):
            md_lines.append(f"- {row['feature']}: PSI={row.get('psi', 0):.3f} ({row['risk_level']})")
        md_lines.append("## ICC Summary")
        for _, row in icc_table.iterrows():
            md_lines.append(f"- {row.get('Type', row.get('targets', 'ICC'))}: {row.get('ICC', 0):.3f}")
        if probe_notes:
            md_lines.append("## Probe Highlights")
            for note in probe_notes[-5:]:
                md_lines.append(f"- **{note['title']}** ({', '.join(note['tags'])}): {note['summary']}")
        md_report = "\n".join(md_lines)
        html_report = "<html><body>" + "</br>".join(md_lines) + "</body></html>"
        st.download_button(
            "Download report (Markdown)",
            data=md_report.encode("utf-8"),
            file_name="report.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download report (HTML)",
            data=html_report.encode("utf-8"),
            file_name="report.html",
            mime="text/html",
        )
        st.write("Artifacts stored in `artifacts/plots`.")


if __name__ == "__main__":
    main()
