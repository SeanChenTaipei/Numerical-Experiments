from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection, MeanSquaredError
from torchmetrics.regression import MeanAbsoluteError, R2Score

MetricFactory = Callable[[], Metric]


class MeanSignedError(Metric):
    """Average signed difference (prediction bias) between preds and targets."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_diff",
                       default=torch.tensor(0.0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.reshape(-1).float()
        target = target.reshape(-1).float()
        if preds.shape != target.shape:
            raise ValueError("preds and target must share the same shape.")
        self.sum_diff += torch.sum(preds - target)
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("MeanSignedError received no samples.")
        return self.sum_diff / self.count


METRIC_REGISTRY: Dict[str, MetricFactory] = {
    "rmse": lambda: MeanSquaredError(squared=False),
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "r2": R2Score,
    "bias": MeanSignedError,
}


def register_metric(name: str, factory: MetricFactory) -> None:
    """Register ``name`` -> ``factory`` so it becomes selectable at runtime.

    Example::

        from torchmetrics import SpearmanCorrCoef
        register_metric(\"spearman\", lambda: SpearmanCorrCoef())
        cfg.metric_names = (\"rmse\", \"spearman\")

    Call this once before building the :class:`MetricManager` (for example at
    the top of your training script or inside a notebook cell).
    """
    if not callable(factory):
        raise TypeError(
            "factory must be callable and return a torchmetrics.Metric instance."
        )
    METRIC_REGISTRY[name] = factory


class MetricManager(nn.Module):
    """Orchestrates torchmetrics collections across stages.

    ``ConditionalSNN`` injects this class so you automatically track metrics
    for ``train``/``val`` (or any custom stage names). Pass
    ``metric_names`` (or tweak :attr:`MixedGAMConfig.metric_names`) to select
    from the registry. To track your own metric: register it via
    :func:`register_metric`, include its key in ``metric_names``, and the
    training loop will log it alongside built-ins.
    """

    def __init__(
        self,
        *,
        stages: Sequence[str] = ("train", "val"),
        metric_names: Optional[Sequence[str]] = None,
        registry: Optional[Mapping[str, MetricFactory]] = None,
    ) -> None:
        super().__init__()
        self._registry: Mapping[str,
                                MetricFactory] = registry or METRIC_REGISTRY
        available = set(self._registry.keys())
        requested = list(metric_names) if metric_names else list(available)
        missing = set(requested) - available
        if missing:
            raise ValueError(f"Unknown metrics requested: {sorted(missing)}. "
                             f"Available metrics: {sorted(available)}")
        self.metric_names = tuple(requested)
        self._stage_to_key = {stage: f"stage_{stage}" for stage in stages}
        self.collections = nn.ModuleDict({
            key: self._build_collection(stage)
            for stage, key in self._stage_to_key.items()
        })

    def _build_collection(self, stage: str) -> MetricCollection:
        metrics = {name: self._registry[name]() for name in self.metric_names}
        return MetricCollection(metrics, prefix=f"{stage}/")

    def update(self, stage: str, preds: torch.Tensor,
               target: torch.Tensor) -> None:
        key = self._stage_to_key.get(stage)
        if key is None:
            raise KeyError(f"Stage '{stage}' is not tracked.")
        self.collections[key].update(preds, target)

    def compute(self, stage: str) -> Dict[str, torch.Tensor]:
        key = self._stage_to_key.get(stage)
        if key is None:
            raise KeyError(f"Stage '{stage}' is not tracked.")
        return self.collections[key].compute()

    def reset(self, stage: str) -> None:
        key = self._stage_to_key.get(stage)
        if key is None:
            raise KeyError(f"Stage '{stage}' is not tracked.")
        self.collections[key].reset()


__all__ = [
    "MetricManager", "register_metric", "METRIC_REGISTRY", "MeanSignedError"
]
