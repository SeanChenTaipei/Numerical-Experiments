import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
from plotly import graph_objects as go
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn import compose

# Optional dependencies
try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SHAP = False

try:
    from rulefit import RuleFit

    HAS_RULEFIT = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_RULEFIT = False

try:
    from evidently import ColumnMapping
    from evidently.metrics import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.report import Report

    HAS_EVIDENTLY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_EVIDENTLY = False

st.set_page_config(page_title="Binary Anomaly Analysis", layout="wide")


# ------------------------------------------------------------
# Data loading helpers (csv / feather / pickle)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_any_from_path(path: str) -> pd.DataFrame:
    """Load a DataFrame from disk based on extension (csv / feather / pickle)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file extension: {ext}")


@st.cache_data(show_spinner=False)
def load_any_from_upload(uploaded_file) -> pd.DataFrame:
    """Load a DataFrame from an uploaded Streamlit file object."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    if ext == ".feather":
        return pd.read_feather(uploaded_file)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(uploaded_file)
    # Fallback: try CSV if extension is unexpected
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def detect_feature_columns(
    df: pd.DataFrame, target_col: str, extra_exclude: Optional[List[str]] = None
) -> List[str]:
    """Infer training feature columns: prefer ALL_CAPS_WITH_UNDERSCORE, exclude target."""
    exclude = {target_col}
    if extra_exclude:
        exclude.update(extra_exclude)
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and c.isupper() and all(ch.isalnum() or ch == "_" for ch in c)
    ]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


@st.cache_data(show_spinner=False)
def compute_class_weights(y: np.ndarray, w0: float, w1: float) -> np.ndarray:
    """Return per-sample weights given class-specific weights."""
    return np.where(y == 1, w1, w0)


# ------------------------------------------------------------
# Model building / CV
# ------------------------------------------------------------

def build_model(model_name: str, params: Dict) -> object:
    """Instantiate model by name with provided parameters."""
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            n_jobs=-1,
            random_state=42,
        )
    if model_name == "CatBoost":
        return CatBoostClassifier(
            iterations=params.get("iterations", 300),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )
    if model_name == "EBM":
        return ExplainableBoostingClassifier(
            interactions=params.get("interactions", 10),
            max_bins=params.get("max_bins", 256),
            max_leaves=params.get("max_leaves", 3),
            learning_rate=params.get("learning_rate", 0.01),
            outer_bags=params.get("outer_bags", 4),
            inner_bags=params.get("inner_bags", 0),
            random_state=42,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def run_cv_training(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    model_name: str,
    params: Dict,
    n_splits: int = 5,
) -> Dict:
    """Run stratified CV, return OOF predictions, fitted models, and metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_pred_proba = np.zeros(len(y), dtype=float)
    models = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        w_train = sample_weight[train_idx] if sample_weight is not None else None

        model = build_model(model_name, params)

        if model_name == "CatBoost":
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=(X_valid, y_valid),
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, sample_weight=w_train)

        proba_valid = model.predict_proba(X_valid)[:, 1]
        oof_pred_proba[valid_idx] = proba_valid
        models.append(model)

    y_pred_05 = (oof_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y, oof_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred_05, average="binary", zero_division=0
    )
    acc = accuracy_score(y, y_pred_05)

    return {
        "oof_pred_proba": oof_pred_proba,
        "models": models,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def get_ebm_contributions(model: ExplainableBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return total logit contributions per sample for an EBM model."""
    local_exp = model.explain_local(X)
    scores = np.array(local_exp.data()["scores"])  # (n_samples, n_terms)
    return scores.sum(axis=1)


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

def plot_univariate_feature(df: pd.DataFrame, feature: str, target_col: str) -> None:
    """Plot histogram/KDE and violin split by target for one feature."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(
        data=df,
        x=feature,
        hue=target_col,
        stat="density",
        kde=True,
        common_norm=False,
        ax=ax[0],
    )
    ax[0].set_title(f"Histogram + KDE by {target_col}")

    sns.violinplot(data=df, x=target_col, y=feature, ax=ax[1])
    ax[1].set_title(f"Distribution by {target_col} (violin)")

    plt.tight_layout()
    st.pyplot(fig)


def plot_pairwise_scatter_target(df: pd.DataFrame, feature1: str, feature2: str, target_col: str) -> None:
    """Scatterplot for two features colored by target."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=df, x=feature1, y=feature2, hue=target_col, alpha=0.6, ax=ax)
    ax.set_title(f"{feature1} vs {feature2} (colored by {target_col})")
    plt.tight_layout()
    st.pyplot(fig)


def plot_pairwise_scatter_contrib_plotly(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    contrib: np.ndarray,
    title_suffix: str = "EBM contribution",
) -> None:
    """Plotly scatter for two features colored by contribution/score."""
    fig = px.scatter(
        df,
        x=feature1,
        y=feature2,
        color=contrib,
        color_continuous_scale="Viridis",
        opacity=0.7,
        labels={feature1: feature1, feature2: feature2, "color": title_suffix},
        title=f"{feature1} vs {feature2} (colored by {title_suffix})",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)


def plot_ebm_main_effect(model: ExplainableBoostingClassifier, feature: str) -> None:
    """Plot 1D EBM main effect for a single feature."""
    try:
        global_exp = model.explain_global()
        data = global_exp.data()
        feature_names = list(model.feature_names_)
        if feature not in feature_names:
            st.info(f"EBM main effects do not contain feature: {feature}")
            return
        idx = feature_names.index(feature)
        scores = np.array(data["scores"][idx])
        bin_labels = np.array(data["bin_labels"][idx])

        fig = go.Figure(
            data=go.Scatter(
                x=bin_labels,
                y=scores,
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title=f"EBM main effect: {feature}",
            xaxis_title=feature,
            yaxis_title="logit contribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # pragma: no cover - plotting fallback
        st.info(f"Could not plot EBM main effect for {feature} because: {exc}")


def plot_ebm_interaction_surface(
    model: ExplainableBoostingClassifier, feature1: str, feature2: str
) -> None:
    """Plot 2D EBM interaction heatmap for a feature pair."""
    try:
        global_exp = model.explain_global()
        data = global_exp.data()

        target_scores = None
        target_bins = None
        target_name = None

        for name, scores, bins in zip(data["names"], data["scores"], data["bin_labels"]):
            if feature1 in name and feature2 in name and len(bins) == 2:
                target_name = name
                target_scores = np.array(scores)
                target_bins = bins
                break

        if target_scores is None:
            st.info(
                "No interaction term found for this pair. Increase EBM interactions or pick a different pair."
            )
            return

        x_bins = np.array(target_bins[0])
        y_bins = np.array(target_bins[1])
        z = target_scores if target_scores.shape[0] == len(x_bins) else target_scores.T

        fig = go.Figure(
            data=go.Heatmap(
                x=x_bins,
                y=y_bins,
                z=z,
                colorbar=dict(title="logit contribution"),
            )
        )
        fig.update_layout(
            title=f"EBM interaction surface: {target_name}",
            xaxis_title=feature1,
            yaxis_title=feature2,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # pragma: no cover - plotting fallback
        st.info(f"Could not plot EBM interaction surface because: {exc}")


# ------------------------------------------------------------
# Feature importance (SHAP / EBM)
# ------------------------------------------------------------

def compute_feature_importance_df(
    model,
    X: pd.DataFrame,
    model_name: str,
    max_samples: int = 2000,
) -> Optional[pd.DataFrame]:
    """Compute feature importance via SHAP (tree models) or global EBM scores."""
    X_sample = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X

    if model_name in ["RandomForest", "CatBoost"]:
        if not HAS_SHAP:
            return None
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            return (
                pd.DataFrame({"feature": X_sample.columns, "importance": mean_abs})
                .sort_values("importance", ascending=False)
            )
        except Exception:
            return None

    if model_name == "EBM":
        try:
            global_exp = model.explain_global(name="EBM")
            data = global_exp.data()
            feature_names = model.feature_names_
            main_term_scores = data["scores"][: len(feature_names)]
            mean_abs = np.array([np.mean(np.abs(s)) for s in main_term_scores])
            return (
                pd.DataFrame({"feature": feature_names, "importance": mean_abs})
                .sort_values("importance", ascending=False)
            )
        except Exception:
            return None

    return None


# ------------------------------------------------------------
# RuleFit helper
# ------------------------------------------------------------

def run_rulefit(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    max_rules: int = 30,
) -> Optional[pd.DataFrame]:
    """Fit RuleFit to extract interaction rules; returns rules DataFrame."""
    if not HAS_RULEFIT:
        return None

    rf_model = RuleFit(
        tree_generator=RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        ),
        max_rules=max_rules,
        random_state=42,
    )

    rf_model.fit(X.values, y, feature_names=X.columns, sample_weight=sample_weight)
    rules = rf_model.get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    return rules


# ------------------------------------------------------------
# Drift detection (Evidently)
# ------------------------------------------------------------

def _split_by_type(
    df: pd.DataFrame, type_col: str
) -> Dict[str, pd.DataFrame]:
    """Split frame into TRAIN/VALID/TEST partitions using a TYPE column (case-insensitive)."""
    split = {}
    for part in ["TRAIN", "VALID", "TEST"]:
        mask = df[type_col].astype(str).str.upper() == part
        if mask.any():
            split[part] = df.loc[mask].copy()
    return split


def run_drift_report(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> str:
    """Run Evidently data/target drift + quality report and return HTML."""
    numerical = [c for c in feature_cols if pd.api.types.is_numeric_dtype(reference_df[c])]
    categorical = [c for c in feature_cols if c not in numerical]
    column_mapping = ColumnMapping(
        target=target_col,
        prediction=None,
        numerical_features=numerical,
        categorical_features=categorical,
    )
    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )
    report.run(reference_data=reference_df, current_data=comparison_df, column_mapping=column_mapping)
    return report.get_html()


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------

def main() -> None:
    st.title("Tabular Binary Anomaly Analysis & Explanation")

    # ---------------- Sidebar: Data Loading -------------------
    st.sidebar.header("1. Data Loading")

    use_dir = st.sidebar.text_input("Working directory (optional)", value="")
    selected_file_path = None

    if use_dir and os.path.isdir(use_dir):
        all_files = os.listdir(use_dir)
        valid_files = [
            f
            for f in all_files
            if os.path.splitext(f)[1].lower()
            in (".csv", ".feather", ".pkl", ".pickle")
        ]
        if valid_files:
            selected_name = st.sidebar.selectbox(
                "Choose file from directory", ["<None>"] + valid_files
            )
            if selected_name != "<None>":
                selected_file_path = os.path.join(use_dir, selected_name)
    elif use_dir:
        st.sidebar.warning("Working directory not found or invalid.")

    uploaded_file = st.sidebar.file_uploader(
        "Or upload a file", type=["csv", "feather", "pkl", "pickle"]
    )

    df = None
    data_source = None

    if selected_file_path:
        df = load_any_from_path(selected_file_path)
        data_source = f"File: {selected_file_path}"
    elif uploaded_file is not None:
        df = load_any_from_upload(uploaded_file)
        data_source = f"Uploaded: {uploaded_file.name}"

    if df is None:
        st.info("Select a working directory + file in the sidebar, or upload a CSV/Feather/Pickle file.")
        return

    # ---------------- Sidebar: Target & Features --------------
    st.sidebar.header("2. Target & Features")

    binary_cols = [
        c
        for c in df.columns
        if df[c].nunique(dropna=True) <= 2 and df[c].dtype != "object"
    ]
    if "target" in df.columns:
        default_target = "target"
    elif binary_cols:
        default_target = binary_cols[0]
    else:
        default_target = df.columns[0]

    target_col = st.sidebar.selectbox(
        "Target (binary)", df.columns, index=df.columns.get_loc(default_target)
    )

    feature_filter = st.sidebar.text_input("Feature name filter (substring)", value="")
    auto_features = detect_feature_columns(df, target_col)

    if feature_filter:
        feature_options = [
            c for c in df.columns if feature_filter.lower() in c.lower() and c != target_col
        ]
    else:
        feature_options = [c for c in df.columns if c != target_col]

    default_selection = [c for c in auto_features if c in feature_options]

    feature_cols = st.sidebar.multiselect(
        "Training feature columns",
        options=feature_options,
        default=default_selection,
    )

    if not feature_cols:
        st.sidebar.warning("Select at least one feature for training.")
        return

    # ---------------- Sidebar: Class Weights & CV -------------
    st.sidebar.header("3. Class Weight & CV")

    w0 = st.sidebar.number_input("Weight for class 0", min_value=0.0, value=1.0, step=0.1)
    w1 = st.sidebar.number_input("Weight for class 1", min_value=0.0, value=5.0, step=0.5)

    n_splits = st.sidebar.slider("CV folds", min_value=3, max_value=10, value=5, step=1)

    y = df[target_col].values.astype(int)
    sample_weight = compute_class_weights(y, w0, w1)
    X = df[feature_cols]

    # ---------------- Sidebar: Model Settings -----------------
    st.sidebar.header("4. Model Settings")

    model_name = st.sidebar.selectbox("Base Model", ["RandomForest", "CatBoost", "EBM"])

    model_params: Dict[str, float | int | None] = {}
    if model_name == "RandomForest":
        model_params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
        max_depth_slider = st.sidebar.slider("max_depth (0 = None)", 0, 20, 10, 1)
        model_params["max_depth"] = None if max_depth_slider == 0 else max_depth_slider
        model_params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1, 1)
    elif model_name == "CatBoost":
        model_params["iterations"] = st.sidebar.slider("iterations", 50, 1000, 300, 50)
        model_params["depth"] = st.sidebar.slider("depth", 2, 10, 6, 1)
        model_params["learning_rate"] = st.sidebar.number_input(
            "learning_rate", 0.001, 1.0, 0.1, 0.01
        )
    elif model_name == "EBM":
        model_params["interactions"] = st.sidebar.slider("interactions", 0, 50, 10, 1)
        model_params["max_bins"] = st.sidebar.slider("max_bins", 16, 512, 256, 16)
        model_params["max_leaves"] = st.sidebar.slider("max_leaves", 2, 10, 3, 1)
        model_params["learning_rate"] = st.sidebar.number_input(
            "learning_rate", 0.001, 0.1, 0.01, 0.001
        )
        model_params["outer_bags"] = st.sidebar.slider("outer_bags", 1, 10, 4, 1)
        model_params["inner_bags"] = st.sidebar.slider("inner_bags", 0, 10, 0, 1)

    # ---------------- Sidebar: Run CV -------------------------
    st.sidebar.header("5. Run")
    run_cv = st.sidebar.button("Run CV Training")

    cv_results = None
    final_models = None

    if run_cv:
        with st.spinner("Running cross-validation..."):
            cv_results = run_cv_training(
                X=X,
                y=y,
                sample_weight=sample_weight,
                model_name=model_name,
                params=model_params,
                n_splits=n_splits,
            )
            final_models = cv_results["models"]
        st.sidebar.success("CV finished!")

    # ---------------- Main Tabs ------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Data Overview",
            "Univariate Dist.",
            "Pairwise Relations",
            "Model & CV",
            "Rules / Interactions",
            "Drift & Summary",
        ]
    )

    with tab1:
        st.subheader("Data Overview")
        st.write(f"**Data source:** {data_source}")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Preview:")
        st.dataframe(df.head())

        st.markdown("### Target distribution")
        target_counts = df[target_col].value_counts().sort_index()
        st.bar_chart(target_counts)

        st.markdown("### Training feature columns")
        st.write(feature_cols)

    with tab2:
        st.subheader("Univariate Feature Distributions (by target)")
        chosen_feature = st.selectbox("Select feature", feature_cols)
        plot_univariate_feature(df, chosen_feature, target_col)

    with tab3:
        st.subheader("Pairwise Relations")

        col_left, col_right = st.columns(2)
        with col_left:
            feature1 = st.selectbox("Feature 1 (X-axis)", feature_cols, key="pair_f1")
        with col_right:
            feature2 = st.selectbox("Feature 2 (Y-axis)", feature_cols, key="pair_f2")

        if feature1 == feature2:
            st.warning("Please choose two different features.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Colored by target (0/1)")
                plot_pairwise_scatter_target(df, feature1, feature2, target_col)

            with col_b:
                st.markdown("#### EBM-based interaction views (Plotly)")

                if cv_results is not None and model_name == "EBM":
                    ebm_model = final_models[0]

                    ebm_contrib = get_ebm_contributions(ebm_model, X)
                    st.markdown("**Sample scatter (color = EBM total score)**")
                    plot_pairwise_scatter_contrib_plotly(
                        df, feature1, feature2, ebm_contrib, title_suffix="EBM total score"
                    )

                    st.markdown("**EBM interaction surface (2D heatmap)**")
                    plot_ebm_interaction_surface(ebm_model, feature1, feature2)

                    with st.expander("EBM main effects (1D) for selected features"):
                        st.markdown(f"**Main effect: {feature1}**")
                        plot_ebm_main_effect(ebm_model, feature1)
                        st.markdown(f"**Main effect: {feature2}**")
                        plot_ebm_main_effect(ebm_model, feature2)

                else:
                    st.info("Select EBM in the sidebar and run CV to see interaction plots.")

    with tab4:
        st.subheader("Model & Cross-Validation Performance")

        if cv_results is None:
            st.info("Click 'Run CV Training' in the sidebar to start.")
        else:
            st.markdown(f"**Model:** {model_name}")

            metrics_df = pd.DataFrame(
                {
                    "metric": ["AUC", "Accuracy", "Precision", "Recall", "F1"],
                    "value": [
                        cv_results["auc"],
                        cv_results["accuracy"],
                        cv_results["precision"],
                        cv_results["recall"],
                        cv_results["f1"],
                    ],
                }
            )
            st.markdown("### Overall metrics (threshold = 0.5 baseline)")
            st.table(metrics_df.style.format({"value": "{:.4f}"}))

            st.markdown("### ROC Curve")
            plot_roc_curve(y, cv_results["oof_pred_proba"])

            threshold = st.slider("Decision threshold for class 1", 0.0, 1.0, 0.5, 0.01)
            y_pred = (cv_results["oof_pred_proba"] >= threshold).astype(int)
            cm = confusion_matrix(y, y_pred)

            st.markdown("### Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df)

            st.markdown("### Out-of-fold prediction distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(cv_results["oof_pred_proba"], bins=50, kde=True, ax=ax)
            ax.axvline(threshold, color="red", linestyle="--")
            ax.set_title("OOF predicted probability (class=1)")
            st.pyplot(fig)

            st.markdown("### Text classification report")
            report = classification_report(y, y_pred, output_dict=False)
            st.text(report)

            st.markdown("### Feature importance ranking (SHAP / EBM)")
            compute_imp_btn = st.button("Compute feature importance")

            if compute_imp_btn:
                model_for_imp = cv_results["models"][0]
                with st.spinner("Computing feature importance..."):
                    imp_df = compute_feature_importance_df(model_for_imp, X, model_name=model_name)
                if imp_df is None or imp_df.empty:
                    if model_name in ["RandomForest", "CatBoost"] and not HAS_SHAP:
                        st.warning("Install shap to enable tree model feature importance.")
                    else:
                        st.info("Could not compute feature importance or result is empty.")
                else:
                    st.dataframe(imp_df)
                    fig, ax = plt.subplots(figsize=(6, max(4, len(imp_df) * 0.3)))
                    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
                    ax.set_title("Feature importance ranking")
                    plt.tight_layout()
                    st.pyplot(fig)

    with tab5:
        st.subheader("Rule-based Interaction Mining (RuleFit)")

        if not HAS_RULEFIT:
            st.warning("Install rulefit (pip install rulefit) to enable this tab.")
        else:
            max_rules = st.number_input("Max rules", min_value=10, max_value=200, value=30, step=10)
            run_rules_btn = st.button("Run RuleFit")

            if run_rules_btn:
                with st.spinner("Running RuleFit..."):
                    rules_df = run_rulefit(X, y, sample_weight, max_rules=int(max_rules))
                if rules_df is None or rules_df.empty:
                    st.info("No rules extracted or all coefficients are zero.")
                else:
                    st.markdown("### Top rules")
                    st.dataframe(rules_df.head(50))

    with tab6:
        st.subheader("Drift & Summary (Evidently)")
        if not HAS_EVIDENTLY:
            st.warning("Install evidently to enable drift reporting (pip install 'evidently>=0.4,<0.5').")
            return

        type_candidates = [c for c in df.columns if c.lower() == "type"]
        if not type_candidates:
            st.info("Provide a TYPE column with TRAIN/VALID/TEST to run drift checks.")
            return

        type_col = type_candidates[0]
        splits = _split_by_type(df, type_col)
        if "TRAIN" not in splits:
            st.warning("TYPE column found, but no TRAIN rows present.")
            return

        reference_choice = st.selectbox("Reference split", options=list(splits.keys()), index=0)
        comparison_options = [k for k in splits.keys() if k != reference_choice]
        if not comparison_options:
            st.info("Need at least one comparison split besides the reference.")
            return

        comparison_choice = st.selectbox("Comparison split", options=comparison_options, index=0)
        sample_limit = st.slider("Max rows per split (sampling for speed)", 100, 20000, 5000, 100)

        ref_df = splits[reference_choice]
        cur_df = splits[comparison_choice]
        if len(ref_df) > sample_limit:
            ref_df = ref_df.sample(sample_limit, random_state=42)
        if len(cur_df) > sample_limit:
            cur_df = cur_df.sample(sample_limit, random_state=42)

        run_report = st.button("Run drift report")
        if run_report:
            with st.spinner("Generating Evidently report..."):
                try:
                    html = run_drift_report(
                        reference_df=ref_df,
                        comparison_df=cur_df,
                        target_col=target_col,
                        feature_cols=feature_cols,
                    )
                    st.components.v1.html(html, height=900, scrolling=True)
                except Exception as exc:  # pragma: no cover - runtime safety
                    st.error(f"Failed to build drift report: {exc}")


if __name__ == "__main__":
    main()
