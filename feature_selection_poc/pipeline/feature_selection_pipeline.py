"""End-to-end regression feature selection pipeline."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from feature_selection_poc.config import FeatureSelectionConfig
from feature_selection_poc.graph import FeatureGraphBuilder
from feature_selection_poc.integrations import create_graph_repository
from feature_selection_poc.kernel.entities import FeatureSelectionResult, SelectionRound, ensure_feature_metadata
from feature_selection_poc.kernel.strategies import FeatureQualityFilter, IterativeImportancePruningStrategy, ScorebookBuilder, make_tail_sample_weight, tail_metrics
from feature_selection_poc.kernel.utils import validate_regression_inputs, write_dataframe, write_json
from feature_selection_poc.models import create_model_adapter
from feature_selection_poc.reports import build_summary_report


class FeatureSelectionPipeline:
    """Template-method style pipeline with fixed stages and pluggable strategies."""

    def __init__(self, config: FeatureSelectionConfig | None = None) -> None:
        self.config = config or FeatureSelectionConfig()
        self.selected_features_: list[str] | None = None
        self.result_: FeatureSelectionResult | None = None

    def fit_select(self, X_train: pd.DataFrame, y_train: pd.Series, feature_metadata: pd.DataFrame | None = None) -> FeatureSelectionResult:
        validate_regression_inputs(X_train, y_train)
        y_train = pd.Series(y_train).reset_index(drop=True)
        X_train = X_train.reset_index(drop=True).select_dtypes(include=[np.number])
        metadata = ensure_feature_metadata(list(X_train.columns), feature_metadata)

        start = time.perf_counter()
        X_quality, quality_metrics = FeatureQualityFilter(self.config.quality).filter(X_train, y_train)
        scorebook = ScorebookBuilder(self.config.scoring).build(X_train, y_train, quality_metrics)
        candidates = scorebook.loc[scorebook["status"].eq("candidate"), "feature_name"].tolist()
        history = [SelectionRound(
            round_id="quality",
            candidate_feature_count=X_train.shape[1],
            removed_feature_count=int(scorebook["status"].eq("removed").sum()),
            selected_feature_count=len(candidates),
            removed_features=scorebook.loc[scorebook["status"].eq("removed"), "feature_name"].tolist(),
            selected_features=candidates,
            metrics={},
            removal_reason="quality_filters",
            runtime_seconds=time.perf_counter() - start,
            config_snapshot=self.config.to_dict(),
        )]

        pruning = IterativeImportancePruningStrategy(self.config.pruning)
        model_eval: dict[str, float] = {}
        for round_idx in range(self.config.pruning.max_rounds):
            if len(candidates) <= self.config.pruning.min_features:
                break
            round_start = time.perf_counter()
            metrics, importances = self._cross_validate_importance(X_quality[candidates], y_train)
            scorebook = self._merge_importance(scorebook, importances)
            selected = pruning.choose_features(scorebook, candidates, round_idx)
            removed = [f for f in candidates if f not in selected]
            scorebook.loc[scorebook["feature_name"].isin(removed), ["status", "selection_round", "removal_reason"]] = ["removed", f"model_round_{round_idx+1}", "low_model_importance"]
            scorebook.loc[scorebook["feature_name"].isin(selected), ["status", "selection_round"]] = ["selected", f"model_round_{round_idx+1}"]
            tail = tail_metrics(y_train, self._fit_predict_in_sample(X_quality[selected], y_train), self.config.scoring.tail_quantile)
            history.append(SelectionRound(
                round_id=f"model_round_{round_idx+1}",
                candidate_feature_count=len(candidates),
                removed_feature_count=len(removed),
                selected_feature_count=len(selected),
                removed_features=removed,
                selected_features=selected,
                metrics=metrics,
                tail_metrics=tail,
                removal_reason="iterative_importance_pruning",
                runtime_seconds=time.perf_counter() - round_start,
                config_snapshot=self.config.to_dict(),
            ))
            candidates = selected
            model_eval = {**metrics, **tail}
            if len(candidates) <= self.config.pruning.keep_top_k:
                break

        nodes, edges, family = FeatureGraphBuilder().build(X_train, scorebook, metadata)
        scorebook = scorebook.drop(columns=["family_id"], errors="ignore").merge(family[["feature_name", "family_id"]], on="feature_name", how="left")
        selected_features = candidates
        removed_features = [f for f in scorebook["feature_name"] if f not in selected_features]
        result = FeatureSelectionResult(selected_features, removed_features, scorebook, metadata, family, nodes, edges, history, model_eval)
        self.selected_features_ = selected_features
        self.result_ = result
        return result

    def _cross_validate_importance(self, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], pd.Series]:
        weights = make_tail_sample_weight(y, self.config.scoring.tail_quantile)
        adapter = create_model_adapter(self.config.model)
        cv = KFold(n_splits=self.config.model.cv_folds, shuffle=True, random_state=self.config.pruning.random_state)
        try:
            preds = cross_val_predict(adapter.estimator, X.fillna(X.median(numeric_only=True)), y, cv=cv, fit_params={"sample_weight": weights})
        except TypeError:
            preds = cross_val_predict(adapter.estimator, X.fillna(X.median(numeric_only=True)), y, cv=cv)
        adapter.fit(X.fillna(X.median(numeric_only=True)), y, sample_weight=weights)
        metrics = {
            "r2": float(r2_score(y, preds)),
            "rmse": float(mean_squared_error(y, preds, squared=False)),
            "mae": float(mean_absolute_error(y, preds)),
        }
        return metrics, adapter.feature_importance(list(X.columns))

    def _fit_predict_in_sample(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        adapter = create_model_adapter(self.config.model)
        Xf = X.fillna(X.median(numeric_only=True))
        adapter.fit(Xf, y, sample_weight=make_tail_sample_weight(y, self.config.scoring.tail_quantile))
        return adapter.predict(Xf)

    def _merge_importance(self, scorebook: pd.DataFrame, importances: pd.Series) -> pd.DataFrame:
        out = scorebook.copy()
        out["model_importance_score"] = out["feature_name"].map(importances).fillna(out["model_importance_score"])
        out["shap_attribution_score"] = out["model_importance_score"]
        out["final_feature_score"] = out["final_feature_score"] + out["model_importance_score"] * 0.25
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit_select before transform")
        return X[self.selected_features_].copy()

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series, feature_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
        result = self.fit_select(X_train, y_train, feature_metadata)
        return X_train[result.selected_features].copy()

    def save_artifacts(self, result: FeatureSelectionResult | None = None, output_dir: str | Path | None = None) -> None:
        result = result or self.result_
        if result is None:
            raise RuntimeError("No result to save")
        out = Path(output_dir or self.config.output.output_dir); out.mkdir(parents=True, exist_ok=True)
        write_dataframe(pd.DataFrame({"feature_name": result.selected_features}), out / "selected_features.csv")
        write_dataframe(pd.DataFrame({"feature_name": result.removed_features}), out / "removed_features.csv")
        write_dataframe(result.scorebook, out / "feature_scorebook.parquet")
        write_dataframe(result.feature_metadata, out / "feature_metadata.parquet")
        write_dataframe(result.feature_family, out / "feature_family.parquet")
        write_dataframe(result.graph_nodes, out / "feature_graph_nodes.csv")
        write_dataframe(result.graph_edges, out / "feature_graph_edges.csv")
        write_json([h.to_dict() for h in result.selection_history], out / "selection_history.json")
        write_json(result.model_evaluation, out / "model_evaluation.json")
        write_json({"backend": self.config.graph.backend, "status": "not_synced"}, out / "graph_backend_sync_report.json")
        build_summary_report(result, out / "summary_report.md")

    def load_artifacts(self, output_dir: str | Path) -> dict[str, pd.DataFrame]:
        out = Path(output_dir)
        return {
            "selected_features": pd.read_csv(out / "selected_features.csv"),
            "removed_features": pd.read_csv(out / "removed_features.csv"),
            "scorebook": pd.read_parquet(out / "feature_scorebook.parquet"),
        }

    def export_scorebook(self, result: FeatureSelectionResult, path: str | Path) -> None:
        write_dataframe(result.scorebook, path)

    def export_feature_graph(self, result: FeatureSelectionResult, output_dir: str | Path) -> None:
        repo = create_graph_repository(self.config.graph.__class__(**{**self.config.graph.__dict__, "backend": "local", "output_dir": str(output_dir)}))
        repo.save_nodes(result.graph_nodes); repo.save_edges(result.graph_edges); repo.sync()

    def sync_feature_graph(self, feature_graph: tuple[pd.DataFrame, pd.DataFrame] | None = None, backend: str | None = None):
        graph_config = self.config.graph
        if backend:
            graph_config = graph_config.__class__(**{**graph_config.__dict__, "backend": backend})
        repo = create_graph_repository(graph_config)
        nodes, edges = feature_graph or (self.result_.graph_nodes, self.result_.graph_edges)  # type: ignore[union-attr]
        repo.save_nodes(nodes); repo.save_edges(edges); repo.sync()
        return repo

    def query_feature_graph(self, feature_name: str, max_depth: int = 2) -> pd.DataFrame:
        repo = self.sync_feature_graph(backend="local")
        return repo.query_feature_neighbors(feature_name, max_depth)

    def export_to_neo4j(self, result: FeatureSelectionResult | None = None):
        return self.sync_feature_graph((result or self.result_).feature_graph, backend="neo4j")  # type: ignore[union-attr]

    def rescue_features(self, result: FeatureSelectionResult, feature_names: list[str]) -> FeatureSelectionResult:
        result.scorebook.loc[result.scorebook["feature_name"].isin(feature_names), "status"] = "rescued"
        result.selected_features = sorted(set(result.selected_features).union(feature_names))
        result.removed_features = [f for f in result.removed_features if f not in feature_names]
        return result

    def rerun_with_rescued_features(self, X_train: pd.DataFrame, y_train: pd.Series, rescued_features: list[str]) -> dict[str, float]:
        if self.selected_features_ is None:
            raise RuntimeError("Run fit_select before rescue rerun")
        features = sorted(set(self.selected_features_).union(rescued_features))
        return self._cross_validate_importance(X_train[features], pd.Series(y_train))[0]
