from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


class LossAccumulator:
    """Collects and names individual loss terms before aggregation.

    Typical usage inside a Lightning ``training_step``::

        losses = LossAccumulator()
        losses.add(\"mse\", F.mse_loss(preds, target))
        losses.add(\"contrastive\", my_custom_loss(preds, target))
        total = losses.total()

    Returning ``losses.as_dict()`` allows the training loop to surface each
    component through progress bars or loggers. To add a new regularizer,
    simply call ``losses.add(\"my_term\", tensor)`` wherever the value is
    computed—no further plumbing is required.
    """

    def __init__(self) -> None:
        self._terms: Dict[str, torch.Tensor] = OrderedDict()

    def add(self, name: str, value: Optional[torch.Tensor]) -> None:
        """Register a scalar tensor under ``name`` if it exists."""
        if value is None:
            return
        self._terms[name] = value

    def total(self) -> torch.Tensor:
        """Return the sum of all registered loss terms."""
        if not self._terms:
            raise ValueError("LossAccumulator requires at least one term.")
        total: Optional[torch.Tensor] = None
        for value in self._terms.values():
            total = value if total is None else total + value
        assert total is not None
        return total

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return a copy so callers can log each term individually."""
        return dict(self._terms)


def residual_regularization(
    cfg: Any,
    head: nn.Module,
    group_idx: torch.Tensor,
    residual_terms: torch.Tensor,
    residual_sum: torch.Tensor,
    base_sum: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Regularization used by the residual head variant."""
    regs = torch.zeros(1, device=residual_terms.device)

    if getattr(cfg, "lambda_center", 0.0) > 0:
        sums = torch.zeros(n_groups,
                           device=residual_terms.device).scatter_add_(
                               0, group_idx, residual_sum)
        counts = torch.zeros(n_groups,
                             device=residual_terms.device).scatter_add_(
                                 0, group_idx, torch.ones_like(residual_sum))
        mask = counts > 0
        if mask.any():
            means = torch.zeros_like(sums)
            means[mask] = sums[mask] / counts[mask]
            regs = regs + cfg.lambda_center * torch.mean(means[mask]**2)

    if getattr(cfg, "lambda_group_l1", 0.0) > 0 and hasattr(
            head, "group_params"):
        l1_terms = [param.abs().mean()
                    for param in head.group_params()]  # type: ignore[call-arg]
        if l1_terms:
            regs = regs + cfg.lambda_group_l1 * torch.stack(l1_terms).sum()

    if getattr(cfg, "lambda_group_l2", 0.0) > 0 and hasattr(
            head, "group_params"):
        l2_terms = [(param**2).mean()
                    for param in head.group_params()]  # type: ignore[call-arg]
        if l2_terms:
            regs = regs + cfg.lambda_group_l2 * torch.stack(l2_terms).sum()

    if getattr(cfg, "residual_contribution_l1", 0.0) > 0:
        regs = regs + cfg.residual_contribution_l1 * residual_terms.abs().mean(
        )

    if getattr(cfg, "residual_contribution_l2", 0.0) > 0:
        regs = regs + cfg.residual_contribution_l2 * (residual_terms**2).mean()

    if getattr(cfg, "lambda_orth", 0.0) > 0:
        centered_base = base_sum - base_sum.mean()
        centered_residual = residual_sum - residual_sum.mean()
        cov = (centered_base * centered_residual).mean()
        regs = regs + cfg.lambda_orth * cov.pow(2)

    return regs


def affine_regularization(
    cfg: Any,
    head: nn.Module,
    delta_sum: torch.Tensor,
    base_sum: torch.Tensor,
) -> torch.Tensor:
    """Regularization used by the affine head variant."""
    regs = torch.zeros(1, device=delta_sum.device)

    if hasattr(head, "parameter_matrices"):
        delta_scale, delta_shift = head.parameter_matrices(
        )  # type: ignore[call-arg]
        if getattr(cfg, "lambda_affine_scale", 0.0) > 0:
            regs = regs + cfg.lambda_affine_scale * delta_scale.abs().mean()
        if getattr(cfg, "lambda_affine_shift", 0.0) > 0:
            regs = regs + cfg.lambda_affine_shift * delta_shift.abs().mean()
        if getattr(cfg, "lambda_affine_center", 0.0) > 0:
            regs = regs + cfg.lambda_affine_center * delta_shift.mean().pow(2)

    if getattr(cfg, "lambda_orth", 0.0) > 0:
        centered_base = base_sum - base_sum.mean()
        centered_delta = delta_sum - delta_sum.mean()
        cov = (centered_base * centered_delta).mean()
        regs = regs + cfg.lambda_orth * cov.pow(2)

    return regs


def base_alignment_loss(cfg: Any, base_sum: torch.Tensor,
                        target: torch.Tensor) -> Optional[torch.Tensor]:
    """Optional penalty that aligns the shared base with the target signal."""
    if getattr(cfg, "lambda_base_target", 0.0) <= 0:
        return None
    return cfg.lambda_base_target * F.mse_loss(base_sum, target)


__all__ = [
    "LossAccumulator",
    "residual_regularization",
    "affine_regularization",
    "base_alignment_loss",
]
