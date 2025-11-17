from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@dataclass(slots=True)
class Settings:
    """Structured view over the global DictConfig."""

    cfg: DictConfig

    def get(self, path: str, default: Any | None = None) -> Any:
        """Return a config value via OmegaConf path with optional default."""
        result = OmegaConf.select(self.cfg, path, default=default)
        return result

    def dump(self) -> DictConfig:
        """Return the underlying config."""
        return self.cfg


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        return OmegaConf.create()
    return OmegaConf.load(path)


@lru_cache(maxsize=4)
def load_config(
    experiment_path: Optional[str] = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load default + experiment YAML into an immutable DictConfig."""

    default_cfg = _load_yaml(CONFIG_DIR / "default.yaml")
    exp_path = experiment_path or Path(
        OmegaConf.select(default_cfg, "experiment.override_path", default="configs/experiment.yaml")
    )
    experiment_cfg = _load_yaml(PROJECT_ROOT / exp_path) if isinstance(exp_path, str) else _load_yaml(exp_path)

    merged = OmegaConf.merge(default_cfg, experiment_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(overrides))

    OmegaConf.set_struct(merged, False)
    return Settings(cfg=merged)


def resolve_artifact_path(*parts: str) -> Path:
    """Create (if needed) and return a path under artifacts."""

    target = ARTIFACT_DIR.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
