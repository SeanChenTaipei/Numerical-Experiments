# Semiconductor Causal Discovery Knowledge Graph Proposal

> Scope: wafer-level long-table or wide-DataFrame data in semiconductor manufacturing, where each feature column is associated with process station, data source, measurement semantics, aggregation window, and physical unit. The downstream objective is feature selection for an end-of-line metrology value, WAT electrical KPI, resistance (`RS`), yield proxy, or other quality target.

## 1. Problem definition

### 1.1 Business and modeling objective

The short-term objective is not to prove final physical causality. It is to build a **causal-discovery-assisted knowledge graph** that turns thousands of wafer-level columns into a searchable, auditable graph of candidate causes, mediators, proxies, and leakage risks.

The graph should support three practical outcomes:

1. **Feature selection**: select a compact feature set around the target's local causal neighborhood instead of relying only on correlation or black-box feature importance.
2. **Interpretability**: explain selected features by station order, data source, measurement behavior, equipment/chamber context, and causal path to the target.
3. **KPI lift validation**: compare model performance and stability between baseline feature selection and graph-constrained causal feature selection.

### 1.2 Data abstraction

Assume the modeling table has one row per wafer, lot-wafer, wafer-step, or wafer-target snapshot. Each column is a `Feature` with attributes:

- `station_id`: process or metrology station.
- `operation_id` / `recipe_id`: process step and recipe if available.
- `source_type`: sensor, metrology, MES, dispatch/queue, maintenance, chamber, equipment event, inline test, WAT, final inspection.
- `behavior_type`: temperature, pressure, flow, RF power, waiting time, dwell time, measurement value, count, event flag, derived aggregation, etc.
- `aggregation`: mean, max, min, std, slope, last, delta, p95, count, missingness flag, and the window used.
- `unit` and `normalization`.
- `time_index` or `station_sequence`: physical order along the route.
- `availability_time`: when the feature is known relative to the prediction target, to prevent leakage.

### 1.3 Causal discovery formulation

Let `Y` be the prediction target. Let `X_i` be wafer-level features. Let `M` be feature metadata. The problem is:

```text
Learn a sparse, metadata-constrained causal candidate graph G = (V, E)
where V contains Features, Stations, DataSources, Behaviors, Targets, and optional Equipment/Recipe nodes,
and E contains domain edges plus statistically discovered candidate causal edges.
Use the Markov blanket / local neighborhood of Y, filtered by domain constraints and validation metrics,
as the recommended feature-selection set.
```

The graph must distinguish three edge classes:

- **Domain structural edges**: station order, feature generated at station, feature derived from raw signal, target measured at final station.
- **Data availability edges**: feature is available before/after target; after-target edges are leakage warnings, not selectable predictors.
- **Discovered causal candidate edges**: learned by causal discovery, with algorithm, effect sign, score, p-value, bootstrap stability, lag/order constraint, and validation status.

## 2. Three-paper survey and implications

### Paper A — Causal discovery in manufacturing

**Vuković and Thalmann (2022), "Causal Discovery in Manufacturing: A Structured Literature Review".** The review frames causal discovery as a way to go beyond traditional machine-learning statistical dependency in Industry 4.0 manufacturing, where ML opacity limits acceptance and organizational learning. It identifies manufacturing motivations such as decision support, production insight, quality, and sustainability, and highlights implementation challenges in scattered industrial applications.

**Implication for this project**

- Do not treat causal discovery as a standalone algorithm run. It should be embedded in a manufacturing workflow with process knowledge, station order, and human validation.
- The first deliverable should be a knowledge graph because it makes discovered links inspectable by process engineers.
- Graph edges must preserve provenance: algorithm, dataset slice, route, chamber, recipe, and validation status.

**Source**: Vuković, M.; Thalmann, S. 2022. DOI: `10.3390/jmmp6010010`. The search result page states that the paper investigates causal discovery in manufacturing by motivations, application scenarios, approaches, impacts, and implementation challenges, and identifies four core areas with a research agenda.

### Paper B — Local causal discovery and Markov blanket feature selection

**Aliferis et al. (2010), "Local Causal and Markov Blanket Induction for Causal Discovery and Feature Selection for Classification Part I".** The paper proposes a Generalized Local Learning framework for learning local causal structure around target variables, especially direct causes/effects and Markov blankets, for very large data sets with relatively small samples. It reports that local causal feature selection can provide parsimony, predictivity, and local causal interpretability under assumptions, while non-causal feature selection should not be interpreted causally even if predictive.

**Implication for this project**

- For short-term feature selection, focus on the target-local graph instead of attempting a full fab-wide DAG over all columns.
- Use `Target <- parent features`, `Target -> child/proxy features`, and `spouse/co-parent features` as categories in the graph.
- The recommended candidate feature set is the target Markov blanket after leakage and station-order filters.

**Source**: JMLR page for Aliferis et al. lists the goal as learning direct causes/effects and Markov blankets around target variables, applicable to very large data sets with relatively small samples, and notes parsimony, predictivity, and local causal interpretability.

### Paper C — NOTEARS continuous optimization for DAG learning

**Zheng et al. (2018), "DAGs with NO TEARS: Continuous Optimization for Structure Learning".** NOTEARS formulates DAG structure learning as a continuous optimization problem over real matrices with a smooth, exact acyclicity characterization, avoiding combinatorial DAG search. The method is attractive for numerical wafer-level feature tables, especially after reducing features by station/source grouping and applying process-order constraints.

**Implication for this project**

- Use NOTEARS or a NOTEARS-family method on curated feature blocks, not on all raw columns at once.
- Encode prior knowledge by masking impossible edges: downstream station cannot cause upstream station; after-target measurements cannot cause target; derived features should not cause their raw ancestors.
- Store learned edge weights as candidate evidence, not final truth.

**Source**: NeurIPS proceedings describe NOTEARS as replacing super-exponential combinatorial DAG search with continuous optimization using a smooth exact acyclicity constraint.

### Optional extension paper for time-dependent sensor traces

If the wafer table includes high-frequency or station-sequence sensor traces, use **Runge et al. (2019), PCMCI**, which targets high-dimensional nonlinear time series with time delays and autocorrelation. This is not the first short-term checkpoint, but it is useful for later converting sensor history into lagged causal edges.

## 3. Proposed Neo4j graph schema

### 3.1 Node labels

| Label | Purpose | Required properties |
| --- | --- | --- |
| `:Feature` | One DataFrame column or engineered column | `feature_id`, `column_name`, `dtype`, `unit`, `aggregation`, `window`, `availability_time`, `missing_rate`, `selectable` |
| `:Target` | Prediction target, also a feature-like variable | `target_id`, `name`, `target_type`, `station_id`, `measurement_time`, `unit` |
| `:Station` | Process/metrology station | `station_id`, `station_name`, `sequence_index`, `module`, `is_metrology` |
| `:DataSource` | Source system or raw table | `source_id`, `source_type`, `system_name`, `refresh_rate`, `owner` |
| `:Behavior` | Physical or semantic behavior | `behavior_id`, `name`, `category` |
| `:Equipment` | Tool/chamber if available | `equipment_id`, `tool_id`, `chamber_id`, `vendor`, `tool_type` |
| `:Recipe` | Recipe/context if available | `recipe_id`, `recipe_name`, `version` |
| `:CausalRun` | One causal discovery experiment | `run_id`, `algorithm`, `dataset_version`, `target_id`, `date`, `constraints_hash`, `notes` |
| `:FeatureSet` | Selected feature set from a checkpoint | `set_id`, `method`, `target_id`, `created_at`, `n_features`, `kpi_delta` |

### 3.2 Relationship types

| Relationship | Direction | Key properties | Meaning |
| --- | --- | --- | --- |
| `(:Feature)-[:OBSERVED_AT]->(:Station)` | feature to station | `confidence` | Feature originates from station. |
| `(:Feature)-[:FROM_SOURCE]->(:DataSource)` | feature to source | `table`, `field`, `lineage` | Data lineage. |
| `(:Feature)-[:DESCRIBES]->(:Behavior)` | feature to behavior | `parser_rule`, `confidence` | Temperature/pressure/wait-time/etc. semantics. |
| `(:Feature)-[:MEASURED_ON]->(:Equipment)` | feature to equipment | `scope` | Tool/chamber association. |
| `(:Feature)-[:USES_RECIPE]->(:Recipe)` | feature to recipe | `scope` | Recipe context. |
| `(:Station)-[:PRECEDES]->(:Station)` | station to station | `route_id`, `min_seq_gap` | Process order. |
| `(:Feature)-[:DERIVED_FROM]->(:Feature)` | engineered to raw | `transform`, `window` | Feature lineage. |
| `(:Feature)-[:AVAILABLE_BEFORE]->(:Target)` | feature to target | `time_gap`, `safe_for_training` | Anti-leakage check. |
| `(:Feature)-[:CANDIDATE_CAUSE_OF]->(:Feature or :Target)` | cause to effect | `run_id`, `algorithm`, `weight`, `effect_sign`, `p_value`, `bootstrap_stability`, `direction_rule`, `status` | Discovered candidate causal edge. |
| `(:Feature)-[:IN_MARKOV_BLANKET_OF]->(:Target)` | feature to target | `run_id`, `role`, `rank`, `reason` | Target-local causal feature-selection evidence. |
| `(:FeatureSet)-[:CONTAINS]->(:Feature)` | feature set to feature | `rank`, `role` | Selected model input. |
| `(:FeatureSet)-[:EVALUATED_ON]->(:Target)` | set to target | `metric`, `baseline`, `candidate`, `delta`, `cv_scheme` | KPI validation. |

### 3.3 Minimal constraints and indexes

```cypher
CREATE CONSTRAINT feature_id_unique IF NOT EXISTS
FOR (f:Feature) REQUIRE f.feature_id IS UNIQUE;

CREATE CONSTRAINT station_id_unique IF NOT EXISTS
FOR (s:Station) REQUIRE s.station_id IS UNIQUE;

CREATE CONSTRAINT target_id_unique IF NOT EXISTS
FOR (t:Target) REQUIRE t.target_id IS UNIQUE;

CREATE CONSTRAINT run_id_unique IF NOT EXISTS
FOR (r:CausalRun) REQUIRE r.run_id IS UNIQUE;

CREATE INDEX feature_column_name IF NOT EXISTS
FOR (f:Feature) ON (f.column_name);

CREATE INDEX station_sequence IF NOT EXISTS
FOR (s:Station) ON (s.sequence_index);
```

## 4. Graph construction workflow

### Step 1 — Feature catalog ingestion

Create a feature catalog table with one row per DataFrame column.

Required columns:

```text
feature_id, column_name, dtype, source_id, source_type, station_id,
station_sequence, behavior_name, unit, aggregation, window,
availability_time, is_target, selectable, raw_lineage, parser_confidence
```

Output graph:

```text
Feature -> Station
Feature -> DataSource
Feature -> Behavior
Feature -> Equipment/Recipe when available
Station -> Station process order
Feature -> Target availability/leakage relation
```

### Step 2 — Domain-constrained causal discovery

Build an adjacency mask before running causal discovery:

```text
allow_edge(i -> j) =
  availability_time(i) <= availability_time(j)
  AND station_sequence(i) <= station_sequence(j) + allowed_same_station_lag
  AND i is not a known target-leakage feature
  AND i is not a downstream metrology proxy measured after Y
  AND source/behavior pair is not blocked by expert rules
```

Recommended first algorithms:

1. **Local Markov blanket / conditional-independence search** for `Y` to get a small target neighborhood.
2. **NOTEARS on the reduced candidate set** to orient and weight edges.
3. **Bootstrap stability** over lots/time windows/products to avoid one-split artifacts.

### Step 3 — Store evidence, not truth

Every candidate causal edge should include:

```text
run_id, algorithm, dataset_version, target_id, train_window,
product_family, route_id, edge_weight, effect_sign, p_value,
bootstrap_stability, allowed_by_domain_rule, violates_order_rule,
status = proposed | accepted | rejected | needs_review
```

This lets engineers query why a feature was selected, and whether evidence is statistical, domain-based, or both.

### Step 4 — Feature selection from graph

A pragmatic scoring rule:

```text
feature_score =
  0.35 * normalized_causal_strength
+ 0.25 * bootstrap_stability
+ 0.15 * domain_prior_score
+ 0.10 * source_quality_score
+ 0.10 * model_ablation_gain
- 0.30 * leakage_risk_score
- 0.10 * redundancy_penalty
```

Candidate feature set:

1. Include accepted direct candidate causes of `Y`.
2. Include Markov blanket members with roles `parent`, `spouse`, or stable `child/proxy` only if they are available before prediction time.
3. Enforce diversity by station and behavior to avoid selecting 30 near-duplicate temperature aggregates.
4. Compare against baseline feature selection using the same cross-validation split.

## 5. Example Neo4j queries

### 5.1 Find all safe temperature features before the target

```cypher
MATCH (f:Feature)-[:DESCRIBES]->(b:Behavior),
      (f)-[:OBSERVED_AT]->(s:Station),
      (f)-[a:AVAILABLE_BEFORE]->(t:Target {name: $target_name})
WHERE b.name = 'temperature'
  AND a.safe_for_training = true
RETURN f.feature_id, f.column_name, s.station_id, s.sequence_index, f.aggregation, f.window
ORDER BY s.sequence_index;
```

### 5.2 Retrieve target Markov blanket for feature selection

```cypher
MATCH (f:Feature)-[mb:IN_MARKOV_BLANKET_OF]->(t:Target {name: $target_name})
WHERE mb.run_id = $run_id
  AND f.selectable = true
RETURN f.feature_id, f.column_name, mb.role, mb.rank, mb.reason
ORDER BY mb.rank ASC;
```

### 5.3 Explain causal paths from upstream station to WAT target

```cypher
MATCH path = (f:Feature)-[:CANDIDATE_CAUSE_OF*1..3]->(t:Target {name: $target_name})
MATCH (f)-[:OBSERVED_AT]->(s:Station)
WHERE s.sequence_index <= $max_station_sequence
RETURN path
LIMIT 50;
```

### 5.4 Find leakage-suspect features

```cypher
MATCH (f:Feature)-[a:AVAILABLE_BEFORE]->(t:Target {name: $target_name})
WHERE a.safe_for_training = false OR f.availability_time > t.measurement_time
RETURN f.feature_id, f.column_name, f.availability_time, t.measurement_time, a.time_gap;
```

## 6. Short-term checkpoints

### Checkpoint 0 — Scope lock, 0.5 day

Deliverables:

- Pick one target: e.g., final metrology value, WAT `RS`, or WAT electrical KPI.
- Freeze one product family / route / time window.
- Define prediction time: at which station or timestamp the model must make prediction.

Exit criteria:

- Target has enough non-missing rows.
- Feature catalog has station/source/behavior coverage for at least 70% of columns.

### Checkpoint 1 — Metadata graph MVP, 1–2 days

Deliverables:

- Neo4j nodes for `Feature`, `Station`, `DataSource`, `Behavior`, and `Target`.
- Edges for `OBSERVED_AT`, `FROM_SOURCE`, `DESCRIBES`, `PRECEDES`, and `AVAILABLE_BEFORE`.
- Cypher queries for source/station/behavior search and leakage filtering.

Exit criteria:

- Engineers can answer: "Which pressure features from stations before target are safe?"
- All target-leakage columns are flagged.

### Checkpoint 2 — Causal evidence graph, 2–4 days

Deliverables:

- Run target-local causal feature search or conditional-dependence screening.
- Run NOTEARS on reduced candidate set with station-order mask.
- Store `CANDIDATE_CAUSE_OF` and `IN_MARKOV_BLANKET_OF` edges with scores and run provenance.

Exit criteria:

- Graph can return top-`k` candidate causal features for the target.
- Each edge has run ID, score, algorithm, and status.

### Checkpoint 3 — Feature-selection experiment, 2–3 days

Deliverables:

- Compare baseline model using current feature-selection method against graph-selected feature set.
- Use time-based or lot-based validation to reduce distribution leakage.
- Report KPI: RMSE/MAE/R² for regression or AUC/F1/yield-capture for classification, plus feature count and stability.

Exit criteria:

- Graph feature set is either smaller with similar KPI, or improves KPI with interpretable added features.
- A ranked list of accepted/rejected causal candidates is available for engineer review.

## 7. Key risks and mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Correlation is mistaken for causality | Manufacturing has common drivers, route/product confounding, tool matching, and rework loops. | Store edges as `CANDIDATE_CAUSE_OF`; require station-order masks, bootstrap stability, and expert review. |
| Target leakage | Downstream metrology or engineered labels can dominate KPI but fail in production. | Make `availability_time` and `AVAILABLE_BEFORE.safe_for_training` mandatory. |
| Too many columns | Full DAG discovery over thousands of features is unstable and expensive. | Start with metadata grouping, target-local Markov blanket search, then NOTEARS on reduced candidates. |
| Mixed products/routes | Different causal mechanisms can be pooled incorrectly. | Build `CausalRun` per product family, route, and time window; compare stable edges across runs. |
| Rework and non-DAG loops | Wafer process may revisit stations. | Model route occurrence index, split rework loops into event-indexed stations, or restrict MVP to non-rework wafers. |

## 8. Recommended MVP deliverable

Build the graph first, before optimizing algorithms. The immediate value is a searchable feature knowledge graph with leakage checks and source/station/behavior lineage. Then add causal discovery evidence as an additional relationship layer. This keeps the short-term task feasible while creating a foundation for long-term causal feature selection and engineer-facing explainability.

## References

1. Vuković, M.; Thalmann, S. "Causal Discovery in Manufacturing: A Structured Literature Review." *Journal of Manufacturing and Materials Processing*, 2022. https://doi.org/10.3390/jmmp6010010
2. Aliferis, C. F.; Statnikov, A.; Tsamardinos, I.; Mani, S.; Koutsoukos, X. D. "Local Causal and Markov Blanket Induction for Causal Discovery and Feature Selection for Classification Part I." *JMLR*, 2010. https://jmlr.org/papers/v11/aliferis10a.html
3. Zheng, X.; Aragam, B.; Ravikumar, P.; Xing, E. P. "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS*, 2018. https://proceedings.neurips.cc/paper/2018/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html
4. Runge, J.; Nowack, P.; Kretschmer, M.; Flaxman, S.; Sejdinovic, D. "Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets." *Science Advances*, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6881151/
