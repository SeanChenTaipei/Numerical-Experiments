"""Backward-compatible shim for ConditionalSNN regressors."""

from .conditional_sensitivity_neural_network.helpers import MixedGAMConfig
from .conditional_sensitivity_neural_network.model import (
    ConditionalSNNRegressor,
    MixedGAMRegressor,
)

__all__ = ["MixedGAMRegressor", "ConditionalSNNRegressor", "MixedGAMConfig"]
