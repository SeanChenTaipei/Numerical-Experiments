"""Configuration loading utilities for the ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    name: str
    test_size: float = 0.2
    random_state: Optional[int] = None


@dataclass
class PreprocessingConfig:
    impute_strategy: str = "mean"
    scale: bool = True


@dataclass
class ModelConfig:
    type: str
    params: Mapping[str, Any]


@dataclass
class CrossValidationConfig:
    enabled: bool = False
    folds: int = 5


@dataclass
class TrainingConfig:
    use_class_weight: bool = False
    scoring: str = "accuracy"
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig)


@dataclass
class EvaluationConfig:
    metrics: List[str]


@dataclass
class OutputsConfig:
    model_dir: Path
    reports_dir: Path


@dataclass
class PipelineConfig:
    experiment_name: str
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    outputs: OutputsConfig
    config_path: Path

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any],
                  config_path: Path) -> "PipelineConfig":
        dataset = DatasetConfig(**raw["dataset"])
        preprocessing = PreprocessingConfig(**raw.get("preprocessing", {}))
        model = ModelConfig(**raw["model"])
        raw_training = raw.get("training", {})
        cross_val = raw_training.get("cross_validation", {})
        training = TrainingConfig(
            use_class_weight=raw_training.get("use_class_weight", False),
            scoring=raw_training.get("scoring", "accuracy"),
            cross_validation=CrossValidationConfig(
                enabled=cross_val.get("enabled", False),
                folds=cross_val.get("folds", 5),
            ),
        )
        evaluation = EvaluationConfig(
            metrics=list(raw.get("evaluation", {}).get("metrics", [])))

        base_dir = config_path.parent
        outputs = raw.get("outputs", {})
        outputs_cfg = OutputsConfig(
            model_dir=(base_dir /
                       outputs.get("model_dir", "artifacts/models")).resolve(),
            reports_dir=(
                base_dir /
                outputs.get("reports_dir", "artifacts/reports")).resolve(),
        )

        return cls(
            experiment_name=raw.get("experiment_name", "experiment"),
            dataset=dataset,
            preprocessing=preprocessing,
            model=model,
            training=training,
            evaluation=evaluation,
            outputs=outputs_cfg,
            config_path=config_path.resolve(),
        )


def load_config(
        config_path: str | Path,
        overrides: Optional[Mapping[str, Any]] = None) -> PipelineConfig:
    """Load and resolve a YAML configuration into strongly typed objects."""
    path = Path(config_path)
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg,
                              OmegaConf.create(_convert_nested(overrides)))
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_container, Mapping):
        raise ValueError("Configuration root must be a mapping.")
    return PipelineConfig.from_dict(cfg_container, path)


def _convert_nested(value: Any) -> Any:
    """Recursively convert dataclass/Path objects to plain python types so OmegaConf can merge."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _convert_nested(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_nested(v) for v in value]
    return value
