from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from .loss import (LossAccumulator, affine_regularization, base_alignment_loss,
                   residual_regularization)
from .metrics import MetricManager


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"selu", "silu"}:
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    return nn.Tanh()


def _make_layer_norm(norm_type: str,
                     num_features: int,
                     *,
                     parallel: bool = False) -> Optional[nn.Module]:
    norm_type = (norm_type or "none").lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(num_features)
    return None


class FeatureMLP(nn.Module):
    """Tiny per-feature MLP used by the additive base learner.

    Args:
        hidden_units: Layer widths for the 1D tower.
        activation: Non-linearity name.
        dropout: Dropout rate applied between hidden layers.
        norm: Normalization strategy (currently ``layernorm`` or ``none``).

    Shapes:
        x: ``(batch, 1)`` single-feature column.
        returns: ``(batch, 1)`` contribution for that feature.
    """

    def __init__(self,
                 hidden_units: Tuple[int, ...],
                 activation: str,
                 dropout: float,
                 norm: str = "none") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 1
        norms = [_make_layer_norm(norm, hidden) for hidden in hidden_units]
        for idx, hidden in enumerate(hidden_units):
            layers.append(nn.Linear(in_dim, hidden))
            norm_layer = norms[idx]
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ParallelFeatureMLP(nn.Module):
    """Vectorized per-feature MLP implemented with einsum for scalability.

    Shapes:
        x: ``(batch, n_features)`` dense feature matrix.
        returns: ``(batch, n_features)`` stacked per-feature outputs.
    """

    def __init__(
        self,
        n_features: int,
        hidden_units: Tuple[int, ...],
        activation: str,
        dropout: float,
        norm: str = "none",
    ) -> None:
        super().__init__()
        dims: List[int] = [1, *hidden_units, 1]
        self.weights = nn.ParameterList()
        self.biases: List[Optional[nn.Parameter]] = []
        self.norms = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1],
                                                          dims[1:])):
            weight = nn.Parameter(torch.empty(n_features, in_dim, out_dim))
            nn.init.kaiming_uniform_(weight.view(-1, out_dim),
                                     a=math.sqrt(5.0))
            self.weights.append(weight)
            if layer_idx == len(dims) - 2:
                self.biases.append(None)
            else:
                bias = nn.Parameter(torch.empty(n_features, out_dim))
                bound = 1 / math.sqrt(in_dim) if in_dim > 0 else 0.0
                nn.init.uniform_(bias, -bound, bound)
                self.biases.append(bias)
                norm_layer = _make_layer_norm(norm, out_dim)
                self.norms.append(norm_layer if norm_layer else nn.Identity())
        while len(self.norms) < len(hidden_units):
            self.norms.append(nn.Identity())
        self.activations = nn.ModuleList(
            _make_activation(activation) for _ in hidden_units)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-1)
        for layer_idx, weight in enumerate(self.weights):
            h = torch.einsum("bfi,fij->bfj", h, weight)
            bias = self.biases[layer_idx]
            if bias is not None:
                h = h + bias.unsqueeze(0)
            if layer_idx < len(self.activations):
                h = self.norms[layer_idx](h)
                h = self.activations[layer_idx](h)
                if self.dropout is not None:
                    h = self.dropout(h)
        return h.squeeze(-1)


class BaseGAM(nn.Module):
    """Additive GAM trunk with a global bias term.

    Shapes:
        x: ``(batch, n_features)`` scaled input.
        returns: ``(base_per_feature, base_sum)`` where
            - ``base_per_feature`` is ``(batch, n_features)``
            - ``base_sum`` is ``(batch, 1)`` summed with bias.
    """

    def __init__(
        self,
        n_features: int,
        hidden_units: Tuple[int, ...],
        activation: str = "tanh",
        dropout: float = 0.05,
        parallel_features: bool = True,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        if parallel_features:
            self.feature_net: Optional[nn.Module] = ParallelFeatureMLP(
                n_features, hidden_units, activation, dropout, norm)
            self.feature_nets: Optional[nn.ModuleList] = None
        else:
            self.feature_net = None
            self.feature_nets = nn.ModuleList(
                FeatureMLP(hidden_units, activation, dropout, norm)
                for _ in range(n_features))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.feature_net is not None:
            stacked = self.feature_net(x)
        else:
            assert self.feature_nets is not None
            contributions = [
                net(x[:, idx:idx + 1])
                for idx, net in enumerate(self.feature_nets)
            ]
            stacked = torch.cat(contributions, dim=1)
        total = stacked.sum(dim=1, keepdim=True) + self.bias
        return stacked, total


class GroupResidualHead(nn.Module):
    """Group-specific residual correction via FiLM/LoRA-style adapters.

    Shapes:
        x: ``(batch, input_dim)`` scaled features.
        group_idx: ``(batch,)`` integer group ids (currently one group column for all features).
        returns: ``(batch, n_features)`` residual deltas added to the base.

    Note:
        Per-feature group columns can be supported in the future by expanding ``group_idx``
        to ``(batch, n_features)`` and broadcasting gamma/beta accordingly; current implementation
        assumes a single group id per sample for all features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Tuple[int, ...],
        n_groups: int,
        n_features: int,
        activation: str = "relu",
        dropout: float = 0.1,
        unknown_index: Optional[int] = None,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.unknown_index = unknown_index

        prev_dim = input_dim
        for hidden in hidden_units:
            self.layers.append(nn.Linear(prev_dim, hidden))
            self.activations.append(_make_activation(activation))
            norm_layer = _make_layer_norm(norm, hidden)
            self.norms.append(norm_layer if norm_layer else nn.Identity())
            prev_dim = hidden
        self.out = nn.Linear(prev_dim, n_features)

        self.gamma_params = nn.ParameterList([
            nn.Parameter(torch.zeros(n_groups, hidden))
            for hidden in hidden_units
        ])
        self.beta_params = nn.ParameterList([
            nn.Parameter(torch.zeros(n_groups, hidden))
            for hidden in hidden_units
        ])

    def forward(self, x: torch.Tensor,
                group_idx: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_idx, linear in enumerate(self.layers):
            h = linear(h)
            h = self.norms[layer_idx](h)
            gamma = self.gamma_params[layer_idx][group_idx]
            beta = self.beta_params[layer_idx][group_idx]
            if self.unknown_index is not None:
                mask = group_idx == self.unknown_index
                if mask.any():
                    gamma = gamma.clone()
                    beta = beta.clone()
                    gamma[mask] = 0.0
                    beta[mask] = 0.0
            h = h * (1 + gamma) + beta
            h = self.activations[layer_idx](h)
            if self.dropout is not None:
                h = self.dropout(h)
        return self.out(h)

    def group_params(self) -> List[torch.Tensor]:
        return list(self.gamma_params) + list(self.beta_params)


class GroupAffineHead(nn.Module):
    """Group-wise affine calibration head following hierarchical regression.

    Shapes:
        base_contribs: ``(batch, n_features)`` base contributions.
        group_idx: ``(batch,)`` integer group ids (currently one group column for all features).
        returns: tuple of
            - adjusted: ``(batch, n_features)`` after affine warp
            - total: ``(batch,)`` summed contribution
            - delta_sum: ``(batch,)`` sum of adjustments vs. base

    Note:
        Extension to per-feature group columns would accept ``group_idx`` as ``(batch, n_features)``
        and gather scale/shift per feature; this implementation assumes a single group id per sample.
    """

    def __init__(
        self,
        n_features: int,
        n_groups: int,
        unknown_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.delta_scale = nn.Parameter(torch.zeros(n_groups, n_features))
        self.shift = nn.Parameter(torch.zeros(n_groups, n_features))
        self.unknown_index = unknown_index

    def forward(
        self,
        base_contribs: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scales = 1.0 + self.delta_scale[group_idx]
        offsets = self.shift[group_idx]
        adjusted = base_contribs * scales + offsets
        if self.unknown_index is not None:
            mask = group_idx == self.unknown_index
            if mask.any():
                adjusted = adjusted.clone()
                adjusted[mask] = base_contribs[mask]
        delta = adjusted - base_contribs
        total = adjusted.sum(dim=1)
        return adjusted, total, delta.sum(dim=1)

    def parameter_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.delta_scale, self.shift


@dataclass
class MixedGAMConfig:
    mode: str = "group_affine"
    scaler: str = "standard"
    base_hidden_units: Tuple[int, ...] = (96, 48)
    base_activation: str = "tanh"
    base_dropout: float = 0.1
    base_norm: str = "layernorm"
    residual_hidden_units: Tuple[int, ...] = (48, )
    residual_activation: str = "relu"
    residual_dropout: float = 0.2
    residual_norm: str = "layernorm"
    learning_rate: float = 7e-4
    weight_decay: float = 5e-5
    batch_size: int = 256
    n_epochs: int = 200
    patience: int = 200
    lambda_center: float = 5e-4
    lambda_group_l1: float = 5e-4
    lambda_group_l2: float = 1e-4
    lambda_orth: float = 1e-4
    lambda_affine_scale: float = 5e-4
    lambda_affine_shift: float = 5e-4
    lambda_affine_center: float = 1e-4
    lambda_base_target: float = 5e-4
    residual_contribution_l1: float = 0.0
    residual_contribution_l2: float = 0.0
    parallel_feature_mlp: bool = False
    verbose: bool = False
    random_state: int = 42
    device: Optional[str] = None
    num_shape_points: int = 200
    checkpoint_dir: str = "checkpoints"
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    metric_names: Tuple[str, ...] = ("rmse", "mae", "r2")


class ConditionalSNN(pl.LightningModule):
    """LightningModule that exposes the Conditional SNN hierarchy.

    The module combines the additive GAM base with either a residual or affine
    group head, tracks every configured loss term, and logs the metrics listed
    in :attr:`MixedGAMConfig.metric_names`. You normally do not instantiate it
    manually—``MixedGAMRegressor`` wires it up for you—but advanced users can
    treat it like any other Lightning model.
    """

    def __init__(
        self,
        base_gam: BaseGAM,
        head: nn.Module,
        config: MixedGAMConfig,
        mode: str,
        n_groups: int,
        metric_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.base_gam = base_gam
        self.head = head
        self.cfg = config
        self.mode = mode
        self.n_groups = n_groups
        self.n_features = base_gam.n_features
        self.training_losses: List[float] = []
        names = tuple(metric_names) if metric_names else tuple(
            config.metric_names)
        self.metric_manager = MetricManager(metric_names=names)

    def forward(
            self, xb: torch.Tensor,
            gb: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        contribs, base_sum = self.base_gam(xb)
        base_sum = base_sum.squeeze(-1)
        if self.mode == "group_residual":
            assert isinstance(self.head, GroupResidualHead)
            residual_terms = self.head(xb, gb)
            residual_sum = residual_terms.sum(dim=1)
            preds = base_sum + residual_sum
            return preds, {
                "base_sum": base_sum,
                "residual_terms": residual_terms,
                "residual_sum": residual_sum,
                "per_feature_base": contribs,
            }
        assert isinstance(self.head, GroupAffineHead)
        adjusted, total, delta_sum = self.head(contribs, gb)
        preds = total + self.base_gam.bias.view(1)
        return preds, {
            "base_sum": base_sum,
            "delta_sum": delta_sum,
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor], batch_idx: int):
        xb, yb, gb = batch
        preds, aux = self(xb, gb)
        losses = LossAccumulator()
        losses.add("mse", F.mse_loss(preds, yb))
        if self.mode == "group_residual":
            assert isinstance(self.head, GroupResidualHead)
            losses.add(
                "residual_regularization",
                residual_regularization(self.cfg, self.head, gb,
                                        aux["residual_terms"],
                                        aux["residual_sum"], aux["base_sum"],
                                        self.n_groups),
            )
        else:
            losses.add(
                "affine_regularization",
                affine_regularization(self.cfg, self.head, aux["delta_sum"],
                                      aux["base_sum"]),
            )
        losses.add("base_alignment",
                   base_alignment_loss(self.cfg, aux["base_sum"], yb))
        loss_terms = losses.as_dict()
        loss = losses.total()
        self.metric_manager.update("train", preds, yb)
        self.training_losses.append(float(loss.detach().cpu()))
        for name, value in loss_terms.items():
            self.log(f"train/{name}",
                     value,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=False)
        self.log("train/total_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=False)
        self.log("train_loss",
                 loss,
                 prog_bar=False,
                 on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor], batch_idx: int):
        xb, yb, gb = batch
        preds, _ = self(xb, gb)
        loss = F.mse_loss(preds, yb)
        self.metric_manager.update("val", preds, yb)
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=False)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_metrics(stage="train", prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(stage="val", prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.base_gam.parameters()) + list(self.head.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer

    def _log_metrics(self, stage: str, prog_bar: bool) -> None:
        metrics = self.metric_manager.compute(stage)
        for name, value in metrics.items():
            self.log(name,
                     value,
                     prog_bar=prog_bar,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=False)
        self.metric_manager.reset(stage)


__all__ = [
    "FeatureMLP",
    "ParallelFeatureMLP",
    "BaseGAM",
    "GroupResidualHead",
    "GroupAffineHead",
    "MixedGAMConfig",
    "ConditionalSNN",
]
