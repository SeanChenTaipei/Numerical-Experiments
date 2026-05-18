# Feature Selection POC Framework Design

## 1. Architecture

The framework is a reusable regression feature-selection package with these layers:

1. **Config layer** (`feature_selection_poc.config`) defines dataclass configuration for quality filters, scoring, pruning, model adapters, graph backends, and artifact output.
2. **Kernel layer** (`feature_selection_poc.kernel`) contains entities, protocols, strategies, validation, and serialization. Strategies are pluggable and testable.
3. **Model adapter layer** (`feature_selection_poc.models`) hides LightGBM, CatBoost, EBM, and fallback sklearn regressors behind one adapter interface.
4. **Graph layer** (`feature_selection_poc.graph`) builds feature family node/edge tables and provides local, NetworkX, and Neo4j repository implementations.
5. **Pipeline layer** (`feature_selection_poc.pipeline`) coordinates data validation, quality filters, scorebook generation, graph construction, model pruning, rescue, and artifact export.
6. **Notebook/report layer** demonstrates an end-to-end POC without embedding core business logic in notebook cells.

## 2. Pipeline Flow

```text
X, y, metadata
  -> validate numeric regression inputs
  -> quality filters: constant, missing, unique ratio, variance, target correlation
  -> scorebook: correlation, mutual information, quality, tail, rescue, final score
  -> family graph: correlation families + metadata-enriched nodes/edges
  -> graph repository sync: local files / NetworkX / Neo4j
  -> model pruning: CV metrics + feature importance + coarse-to-fine reduction
  -> tail report: tail RMSE/MAE and weighted training
  -> optional domain rescue and rerun
  -> artifacts: csv/parquet/json/markdown graph and history outputs
```

## 3. Project Structure

The implementation follows the requested structure under `feature_selection_poc/`, plus `notebooks/01_feature_selection_pipeline_walkthrough.ipynb`, tests, and this design document.

## 4. Core Entities and Interfaces

- `FeatureSelectionConfig`: top-level config loaded from YAML or constructed in Python.
- `FeatureSelectionResult`: selected/removed features, scorebook, metadata, family table, graph tables, histories.
- `SelectionRound`: auditable round-level selection history.
- `RegressionModelAdapter`: fit/predict/feature-importance protocol.
- `FeatureGraphRepository`: save/load/query/sync protocol for graph backends.

## 5. Feature Quality Filters

| Filter | Purpose | Inputs | Outputs | Parameters | Cost | High-dimensional acceleration |
| --- | --- | --- | --- | --- | --- | --- |
| Constant / near-constant | remove non-informative columns | `X` | removal reason | `near_constant_threshold` | O(n p) | streaming value counts, sample rows |
| Missing rate | remove sparse unusable features | `X` | missing rate | `missing_rate_threshold` | O(n p) | chunked null counts |
| Unique ratio | catch IDs/leaky identifiers or degenerate columns | `X` | unique ratio | min/max unique ratio | O(n p) | approximate distinct counting |
| Low variance | remove numerically flat features | `X` | variance | variance threshold | O(n p) | vectorized/chunked variance |
| Target correlation | cheap univariate relevance | `X, y` | Pearson corr | min abs corr | O(n p) | compute after quality screen, sample rows |
| Mutual information | non-linear univariate relevance | `X, y` | MI score | min MI / top-k cap | O(n p log n) | run only after quality filtering; sample features/rows |
| Contribution filter | extensible importance/contribution clustering | scorebook/model outputs | removal reason | strategy-specific | model-dependent | family representative pruning |

## 6. Scorebook Design

The scorebook keeps one row per feature and includes: quality score, missing penalty, unique-ratio score, variance score, univariate contribution, target correlation, mutual information, model importance, SHAP/attribution placeholder, tail importance, outlier sensitivity, CV stability placeholder, representative score, redundancy, domain rescue score, final score, status, reason, round, and family ID. Scores are normalized with min-max scaling where needed and combined through configurable weights.

Risks: univariate scores miss interactions, mutual information can be expensive/noisy, correlation can over-rank redundant families, and tail scores can overfit when tail sample count is tiny.

## 7. Feature Family / Graph Metadata

Feature nodes contain feature name, data source, station, process step, tool, recipe, missing rate, variance, unique ratio, target correlation, MI, model importance, attribution score, tail score, family ID, selected/removed/rescued status, reason, and selection round. Edges represent `CORRELATED_WITH` relationships with correlation and weight.

Large-scale approximations: pre-cluster by metadata, compute correlation only inside candidate blocks, use sampled rows, use approximate nearest neighbors on feature embeddings, and cap exported edges.

## 8. Knowledge Graph / Neo4j Design

Labels: `Feature`, `FeatureFamily`, `DataSource`, `Station`, `ProcessStep`, `ModelRun`, `SelectionRound`.
Relationships: `BELONGS_TO_FAMILY`, `FROM_DATA_SOURCE`, `MEASURED_AT_STATION`, `IN_PROCESS_STEP`, `CORRELATED_WITH`, `SELECTED_IN`, `REMOVED_IN`, `RESCUED_IN`, `IMPORTANT_IN_MODEL`.

Example Cypher queries are provided in `feature_selection_poc/graph/cypher_templates.py`: feature neighbors, selected station features, family representative, rescue candidates, model-run memberships, and run comparison.

## 9. Model-based Selection

The default strategy is coarse-to-fine:

1. Quality and univariate scoring reduce very large feature sets.
2. Cross-validated model importance ranks surviving candidates.
3. Iterative pruning keeps a configurable fraction or top-k.
4. Domain rescue can add removed high-tail-score features.
5. Artifacts preserve selected features, removed features, scorebook, graph, history, and model metrics.

Adapters can target LightGBM, CatBoost, EBM, or fallback sklearn estimators.

## 10. Tail / Outlier-aware Handling

Tail-aware regression uses y-quantile sample weights, tail RMSE, tail MAE, and tail correlation scores. For extreme sparse spike targets (for example 1000 rows with 4-5 non-zero spikes), standard regression is usually not stable. Reuse the quality, scorebook, metadata, graph, and artifact layers, but branch the modeling stage into anomaly detection, rare-event classification + magnitude regression, ranking, one-class/semi-supervised learning, or extreme-sample retrieval.

## 11. POC Datasets

- General regression: sklearn California Housing or synthetic bell-shaped regression.
- Skewed target regression: Ames Housing / Bike Sharing / synthetic exponentiated target.
- Outlier or sparse spike: NYC taxi duration with extreme trips, industrial fault magnitude datasets, or the included synthetic sparse spike generator.

## 12. Notebook Design

The notebook has 14 sections: overview, setup, load data, metadata, config, quality filters, scorebook, family graph, graph sync/query, model selection, tail analysis, domain rescue, export artifacts, final summary.

## 13. Artifact Formats

- `selected_features.csv`, `removed_features.csv`
- `feature_scorebook.parquet`, `feature_metadata.parquet`, `feature_family.parquet`
- `feature_graph_nodes.csv`, `feature_graph_edges.csv`
- `selection_history.json`, `model_evaluation.json`, `graph_backend_sync_report.json`
- `summary_report.md`

## 14. Extension Points

- New model: implement `fit`, `predict`, `feature_importance` and add it to `create_model_adapter`.
- New scorer/selector: implement a strategy class and inject it into the pipeline or config factory.
- New graph backend: implement `FeatureGraphRepository` and register it in `create_graph_repository`.
- Agent workflow/Skill: call the unified pipeline API, use scorebook/graph repository queries as tools, and persist every run artifact for reproducibility.
