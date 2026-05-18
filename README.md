# SmartCategoricalEncoder Manual

A lightweight yet extensible categorical feature encoder that surveys every column, picks a strategy that matches its statistics, and exposes a registry for plugging in custom encoders. This document explains what is bundled, how the auto-strategy rules work, and how to integrate the encoder inside scikit-learn pipelines.

## Highlights
- **One class, many encoders** - Target/CatBoost/James-Stein/WoE/LOO/Hashing/Embedding/Helmert/Rare grouping/etc. are all unified under `SmartCategoricalEncoder`.
- **Auto strategy** - cardinality, unique ratio, rare share, and missing rate drive the recommendation when `strategy="auto"`.
- **Manual overrides** - pass `strategy={"col_name": "hashing"}` or a single strategy string to force behaviour.
- **Plug-in ready** - custom column encoders can be registered via `SmartCategoricalEncoder.register_encoder("name", cls)`.
- **Sklearn compatible** - implements `fit`, `transform`, `fit_transform`, and works inside `Pipeline`/`ColumnTransformer`.
- **Persistence** - save/load through `joblib` for reproducible deployments.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start
```python
import pandas as pd
from src.pipeline.smart_categorical_encoder import SmartCategoricalEncoder

X = pd.DataFrame(
    {
        "city": ["NY", "NY", "SF", "LA", None],
        "device": ["ios", "android", "web", "ios", "web"],
        "gender": ["F", "M", "F", "F", "M"],
    }
)
y = pd.Series([1, 0, 1, 0, 1], name="label")

encoder = SmartCategoricalEncoder(
    strategy="auto",
    params={
        "general": {
            "rare_threshold": 0.05,
            "max_onehot_cardinality": 12,
            "max_target_cardinality": 60,
            "max_hash_cardinality": 200,
        },
        "target": {"smoothing": 15.0},
        "hashing": {"n_features": 32},
    },
    random_state=42,
)

X_enc = encoder.fit_transform(X, y)
print(X_enc.head())
print(encoder.get_feature_names())
encoder.save("artifacts/smart_encoder.joblib")
```

## Auto Strategy Cheat Sheet
| Condition | Suggested Strategy |
| --- | --- |
| Cardinality <= 12 | One-Hot |
| 12 < card <= 60 & label available | CatBoost / Target |
| 12 < card <= 60 & no label | Ordinal |
| 60 < card <= 200 & label | Target (smoothing) |
| 60 < card <= 200 & no label | Hashing |
| Cardinality > 200 & label | Multi-Hash Embedding |
| Cardinality > 200 & no label | Entity Embedding |
| Missing rate > 20% & label | CatBoost |
| Unique ratio > 0.9 & label | Multi-Hash |
| Rare share over threshold | Rare Category Grouping (then One-Hot/Target) |
| Leakage risk / CV pipelines | Leave-One-Out with proper folds |

All thresholds are configurable under `params["general"]`.

## Built-in Encoders
| Strategy | Works Best For | Notes |
| --- | --- | --- |
| `onehot` | Low-cardinality nominal features | Produces dense matrix (no sparse output). |
| `ordinal` | Medium cardinality without a target | Encodes categories as integers with -1 for unseen. |
| `target` | Supervised tasks with enough samples | Mean target with smoothing + optional noise. |
| `catboost` | High-cardinality supervised data | Online-style mean encoding to reduce leakage. |
| `james_stein` | Regression tasks needing shrinkage | Weighted mix of group mean and global mean. |
| `woe` | Binary classification / risk models | Requires binary target, outputs log odds. |
| `leave_one_out` | Leakage-sensitive setups | Per-row target mean excluding the row itself. |
| `count`/`frequency` | Unsupervised feature weighting | Maps categories to counts or normalized counts. |
| `hashing` | Huge vocabularies without y | Feature hashing with configurable dimension. |
| `binary` | Mid-cardinality without target | Converts category index to binary digits. |
| `helmert` | ANOVA / contrast coding | Produces (k-1) orthogonal contrasts. |
| `rare_grouping` | Columns with long tails | Buckets infrequent values into a shared "rare" column. |
| `embedding` | Dense representation for high card | One-Hot + TruncatedSVD approximation. |
| `multi_hash` | Recommender-style IDs | Multiple hashes with learned embeddings per bucket. |

Each encoder accepts parameters via the `params` dictionary (`{"hashing": {"n_features": 128}}`, etc.).

## Column Survey Metrics
`fit` computes per-column stats stored under `column_stats_`:
- `cardinality`: distinct non-null categories
- `unique_ratio`: `cardinality / non_null_rows`
- `rare_share`: portion of rows occupied by rare categories
- `missing_rate`: NaN share
- `most_common_freq`: dominant category frequency

Use these values (and `decision_log_`) to inspect what the auto-strategy picked.

## Extending With Custom Encoders
```python
from src.pipeline.smart_categorical_encoder import SmartCategoricalEncoder, _BaseColumnEncoder

class ConstantEncoder(_BaseColumnEncoder):
    """Maps every category to the same scalar value (demo)."""

    def __init__(self, column: str, value: float = 0.0) -> None:
        super().__init__(column)
        self.value = value
        self.feature_names_ = [f"{column}__constant"]

    def fit(self, series, y=None):
        return self

    def transform(self, series):
        return pd.DataFrame(self.value, index=series.index, columns=self.feature_names_)

SmartCategoricalEncoder.register_encoder("constant", ConstantEncoder)
enc = SmartCategoricalEncoder(strategy={"city": "constant"}, params={"constant": {"value": 0.7}})
```

## Pipeline / ColumnTransformer Integration
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

cat_cols = ["city", "device", "gender"]
num_cols = ["age", "income"]

preprocess = ColumnTransformer(
    transformers=[
        ("categorical", SmartCategoricalEncoder(strategy="auto", random_state=21), cat_cols),
        ("numeric", "passthrough", num_cols),
    ]
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier()),
])

pipeline.fit(train_X, train_y)
y_pred = pipeline.predict(test_X)
```

## Persistence & Reuse
```python
encoder.save("artifacts/encoder.joblib")
loaded = SmartCategoricalEncoder.load("artifacts/encoder.joblib")
assert loaded.get_feature_names() == encoder.get_feature_names()
```

## Troubleshooting
- **Missing columns at transform** - the transformer expects the same categorical columns observed during `fit`.
- **Binary-only strategies (WoE)** - ensure `y` has exactly two unique values.
- **High variance in target encoders** - increase smoothing or add Gaussian noise via params.
- **Rare categories not grouped** - raise `params["general"]["rare_threshold"]` or pre-group with domain rules.

Happy encoding!
---

# Feature Selection POC Framework

A reusable Python framework for regression feature-selection POCs with auditable scorebooks, feature families, graph artifacts, model-based pruning, tail-aware metrics, and domain-expert rescue workflows.

## Suitable Use Cases

- Bell-shaped, skewed, tail-heavy, or moderately outlier-sensitive regression targets.
- High-dimensional tabular feature spaces that need staged reduction.
- Feature review workflows where domain experts need removal reasons, redundancy families, and rescue candidates.
- POCs that may later sync feature lineage to Neo4j or another knowledge graph engine.

For extremely sparse spike targets, reuse the quality/scorebook/graph stages but replace the modeling stage with anomaly detection, rare-event two-stage modeling, ranking, one-class learning, or retrieval.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from feature_selection_poc.config import FeatureSelectionConfig
from feature_selection_poc.data import make_regression_poc_dataset
from feature_selection_poc.pipeline import FeatureSelectionPipeline

X, y, metadata = make_regression_poc_dataset(target_type="skewed")
config = FeatureSelectionConfig()
pipeline = FeatureSelectionPipeline(config)
result = pipeline.fit_select(X, y, feature_metadata=metadata)

print(result.selected_features)
pipeline.save_artifacts(result, output_dir="outputs/demo_run")
repo = pipeline.sync_feature_graph(result.feature_graph, backend="local")
print(repo.query_feature_neighbors(result.selected_features[0], max_depth=2))
```

## Notebook

Run the walkthrough:

```bash
jupyter notebook notebooks/01_feature_selection_pipeline_walkthrough.ipynb
```

The notebook covers setup, synthetic/public-data fallback, metadata, configuration, quality filters, scorebook, feature family graph, local/Neo4j graph repository usage, model-based pruning, tail analysis, domain rescue, and artifact export.

## Pipeline API

- `fit_select(X_train, y_train, feature_metadata=None)`
- `transform(X)` / `fit_transform(X, y, metadata)`
- `save_artifacts(result, output_dir)` / `load_artifacts(output_dir)`
- `export_scorebook(result, path)`
- `export_feature_graph(result, output_dir)`
- `sync_feature_graph(result.feature_graph, backend="local|networkx|neo4j")`
- `query_feature_graph(feature_name, max_depth)`
- `export_to_neo4j(result)`
- `rescue_features(result, feature_names)`
- `rerun_with_rescued_features(X, y, rescued_features)`

## Output Artifacts

- `selected_features.csv` and `removed_features.csv`
- `feature_scorebook.parquet`
- `feature_metadata.parquet`
- `feature_family.parquet`
- `feature_graph_nodes.csv` and `feature_graph_edges.csv`
- `selection_history.json`
- `model_evaluation.json`
- `graph_backend_sync_report.json`
- `summary_report.md`

## Extending Models, Scorers, Selectors, and Graph Backends

- Add a model adapter by implementing the `RegressionModelAdapter` protocol and registering it in `feature_selection_poc.models.create_model_adapter`.
- Add a scorer by following `ScorebookBuilder` and writing normalized columns into the scorebook.
- Add a selector/pruner by following `IterativeImportancePruningStrategy`.
- Add a graph backend by implementing `FeatureGraphRepository` and registering it in `feature_selection_poc.integrations.create_graph_repository`.

## Neo4j Integration

Set `GraphBackendConfig(backend="neo4j", neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="...")` and call `pipeline.export_to_neo4j(result)`. Schema labels, relationship types, constraints, and example Cypher templates live in `feature_selection_poc/graph/graph_schema.py` and `feature_selection_poc/graph/cypher_templates.py`.

## Feature Graph Queries

Local, NetworkX, and Neo4j repositories share a common interface:

```python
repo.query_feature_neighbors("f_001", max_depth=2)
repo.query_family_members("corr_family_0001")
```

## Domain Rescue

Use the scorebook, graph neighbors, and tail scores to choose candidate features, then rerun:

```python
rescued = ["f_007", "f_dup_002"]
result = pipeline.rescue_features(result, rescued)
metrics = pipeline.rerun_with_rescued_features(X, y, rescued)
```

## FAQ

**Can this handle hundreds of thousands of features?** Use staged filtering, top-k caps, sampled MI/correlation, metadata blocking, and edge caps before model pruning.

**Do I need Neo4j?** No. The local repository writes node and edge tables that are easy to visualize or load later.

**Where is the architecture design?** See `docs/feature_selection_poc_design.md`.
