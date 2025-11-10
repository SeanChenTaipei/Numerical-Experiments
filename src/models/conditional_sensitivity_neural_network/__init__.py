"""Conditional Sensitivity Neural Network building blocks."""

from .helpers import (
    BaseGAM,
    ConditionalSNN,
    FeatureMLP,
    GroupAffineHead,
    GroupResidualHead,
    MixedGAMConfig,
    ParallelFeatureMLP,
)
from .model import ConditionalSNNRegressor, MixedGAMRegressor
from .metrics import METRIC_REGISTRY, MetricManager, register_metric
from .loss import (
    LossAccumulator,
    affine_regularization,
    base_alignment_loss,
    residual_regularization,
)

__all__ = [
    "BaseGAM",
    "ConditionalSNN",
    "ConditionalSNNRegressor",
    "FeatureMLP",
    "GroupAffineHead",
    "GroupResidualHead",
    "LossAccumulator",
    "METRIC_REGISTRY",
    "MetricManager",
    "MixedGAMConfig",
    "MixedGAMRegressor",
    "ParallelFeatureMLP",
    "affine_regularization",
    "base_alignment_loss",
    "register_metric",
    "residual_regularization",
]
