"""Configuration objects for the feature selection POC framework."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency fallback
    yaml = None


@dataclass
class QualityFilterConfig:
    missing_rate_threshold: float = 0.4
    near_constant_threshold: float = 0.98
    unique_ratio_min: float = 0.0
    unique_ratio_max: float = 0.98
    variance_threshold: float = 1e-8
    min_abs_target_corr: float = 0.0
    min_mutual_info: float = 0.0
    max_features_after_quality: int | None = 5000


@dataclass
class ScoringConfig:
    tail_quantile: float = 0.9
    final_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "quality_score": 0.20,
            "target_correlation_score": 0.20,
            "mutual_info_score": 0.20,
            "variance_score": 0.10,
            "tail_importance_score": 0.15,
            "feature_family_representative_score": 0.10,
            "domain_rescue_score": 0.05,
        }
    )


@dataclass
class PruningConfig:
    max_rounds: int = 3
    keep_top_k: int = 25
    keep_fraction: float = 0.5
    min_features: int = 5
    random_state: int = 42


@dataclass
class ModelConfig:
    name: str = "random_forest"
    params: dict[str, Any] = field(default_factory=lambda: {"n_estimators": 80, "random_state": 42})
    cv_folds: int = 3
    early_stopping_rounds: int | None = None


@dataclass
class GraphBackendConfig:
    backend: str = "local"
    output_dir: str = "outputs/feature_graph"
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    node_format: str = "csv"
    edge_format: str = "csv"


@dataclass
class OutputConfig:
    output_dir: str = "outputs/feature_selection_run"
    save_parquet: bool = True
    save_csv: bool = True
    run_id: str = "demo_run"


@dataclass
class FeatureSelectionConfig:
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphBackendConfig = field(default_factory=GraphBackendConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureSelectionConfig":
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files")
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            quality=QualityFilterConfig(**data.get("quality", {})),
            scoring=ScoringConfig(**data.get("scoring", {})),
            pruning=PruningConfig(**data.get("pruning", {})),
            model=ModelConfig(**data.get("model", {})),
            graph=GraphBackendConfig(**data.get("graph", {})),
            output=OutputConfig(**data.get("output", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if yaml is None:
            raise ImportError("PyYAML is required to write YAML config files")
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
