"""Selection history entities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SelectionRound:
    round_id: str
    candidate_feature_count: int
    removed_feature_count: int
    selected_feature_count: int
    removed_features: list[str]
    selected_features: list[str]
    metrics: dict[str, float]
    tail_metrics: dict[str, float] = field(default_factory=dict)
    removal_reason: str = ""
    runtime_seconds: float = 0.0
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
