"""Hierarchical explainable regression with additive trunks plus group adapters.

References:
    - Lou et al., "Interpretable Machine Learning via GA2M" (KDD 2013)
    - Caruana et al. (KDD 2015)
    - Nori et al., "Explainable Boosting Machine" (2019)
    - Perez et al., "FiLM" (CVPR 2018); Hu et al., "LoRA" (ICLR 2022)
    - Yang et al., "GAMI-Net" (2021); Agarwal et al., NAM (NeurIPS 2021)
    - Long et al., "Explainable AI – Latest Advancements and Trends" (2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ArrayLike = Union[pd.DataFrame, np.ndarray]
MetricFn = Callable[[np.ndarray, np.ndarray], float]
DEFAULT_HYPERPARAMETER_SPACE: List[Dict[str, Any]] = [
    {
        "learning_rate": 1e-3
    },
    {
        "learning_rate": 2e-3,
        "base_dropout": 0.1
    },
    {
        "learning_rate": 5e-4,
        "base_hidden_units": (128, 64)
    },
    {
        "residual_dropout": 0.2
    },
    {
        "lambda_base_target": 5e-3
    },
]


def _ensure_tuple(units: Optional[Sequence[int]],
                  default: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(units) if units else default


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


class _IdentityScaler:
    """Simple passthrough scaler with a scikit-learn compatible API."""

    def fit(self, X: np.ndarray) -> "_IdentityScaler":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class GroupCrossEncoder:
    """Map arbitrary group labels to contiguous indices for adapter lookup."""

    def __init__(self, unknown_token: str = "__unknown__") -> None:
        self.unknown_token = unknown_token
        self.mapping: Dict[str, int] = {}
        self.inverse: Dict[int, str] = {}
        self.unknown_index: Optional[int] = None
        self.n_groups_: int = 0
        self._fitted = False

    def fit(self, groups: Sequence[str]) -> "GroupCrossEncoder":
        normalized = [self._normalize(g) for g in groups]
        seen: List[str] = []
        for label in normalized:
            if label == self.unknown_token:
                continue
            if label not in seen:
                seen.append(label)
        self.mapping = {label: idx for idx, label in enumerate(seen)}
        self.unknown_index = len(self.mapping)
        self.mapping[self.unknown_token] = self.unknown_index
        self.inverse = {idx: label for label, idx in self.mapping.items()}
        self.n_groups_ = len(self.mapping)
        self._fitted = True
        return self

    def transform(self, groups: Sequence[str]) -> np.ndarray:
        if not self._fitted or self.unknown_index is None:
            raise RuntimeError(
                "GroupCrossEncoder must be fitted before calling transform.")
        normalized = [self._normalize(g) for g in groups]
        encoded = [
            self.mapping.get(label, self.unknown_index) for label in normalized
        ]
        return np.asarray(encoded, dtype=np.int64)

    def inverse_transform(self, indices: Sequence[int]) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Fit encoder before calling inverse_transform.")
        return [
            self.inverse.get(int(idx), self.unknown_token) for idx in indices
        ]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mapping": self.mapping,
            "inverse": self.inverse,
            "unknown_token": self.unknown_token,
            "unknown_index": self.unknown_index,
            "n_groups": self.n_groups_,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "GroupCrossEncoder":
        self.mapping = state["mapping"]
        self.inverse = state["inverse"]
        self.unknown_token = state["unknown_token"]
        self.unknown_index = state["unknown_index"]
        self.n_groups_ = state["n_groups"]
        self._fitted = True
        return self

    def _normalize(self, value: Any) -> str:
        if value is None:
            return self.unknown_token
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return self.unknown_token
        except Exception:
            pass
        return str(value)


class FeatureMLP(nn.Module):
    """Tiny per-feature MLP used by the additive base learner."""

    def __init__(
        self,
        hidden_units: Tuple[int, ...],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 1
        for hidden in hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BaseGAM(nn.Module):
    """Additive GAM trunk (Lou et al., 2013; Caruana et al., 2015; Nori et al., 2019)."""

    def __init__(
        self,
        n_features: int,
        hidden_units: Tuple[int, ...],
        activation: str = "tanh",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.feature_nets = nn.ModuleList(
            FeatureMLP(hidden_units, activation, dropout)
            for _ in range(n_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        contributions = [
            net(x[:, idx:idx + 1]) for idx, net in enumerate(self.feature_nets)
        ]
        stacked = torch.cat(contributions, dim=1)
        # stacked: (batch, n_features) additive contributions per feature
        total = stacked.sum(dim=1, keepdim=True)
        # total: (batch, 1) shared base prediction before adapters
        return stacked, total


class GroupResidualHead(nn.Module):
    """Group-specific residual correction via FiLM/LoRA-style low-rank adapters."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Tuple[int, ...],
        n_groups: int,
        activation: str = "relu",
        dropout: float = 0.1,
        unknown_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.unknown_index = unknown_index

        prev_dim = input_dim
        for hidden in hidden_units:
            self.layers.append(nn.Linear(prev_dim, hidden))
            self.activations.append(_make_activation(activation))
            prev_dim = hidden
        self.out = nn.Linear(prev_dim, 1)

        self.gamma_params = nn.ParameterList([
            nn.Parameter(torch.zeros(n_groups, hidden))
            for hidden in hidden_units
        ])  # FiLM scales, shape: (n_groups, hidden_dim_l)
        self.beta_params = nn.ParameterList([
            nn.Parameter(torch.zeros(n_groups, hidden))
            for hidden in hidden_units
        ])  # FiLM shifts, shape matches gamma_params

    def forward(self, x: torch.Tensor,
                group_idx: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_idx, linear in enumerate(self.layers):
            h = linear(h)
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
        residual = self.out(h).squeeze(-1)
        return residual

    def group_params(self) -> List[torch.Tensor]:
        return list(self.gamma_params) + list(self.beta_params)


class GroupAffineHead(nn.Module):
    """Group-wise affine calibration head following hierarchical regression practice."""

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
        if self.unknown_index is not None:
            mask = group_idx == self.unknown_index
            if mask.any():
                scales = scales.clone()
                offsets = offsets.clone()
                scales[mask] = 1.0
                offsets[mask] = 0.0
        adjusted = scales * base_contribs + offsets
        # adjusted: (batch, n_features) calibrated per-feature terms
        delta = adjusted - base_contribs
        total = adjusted.sum(dim=1)
        return adjusted, total, delta.sum(dim=1)

    def parameter_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.delta_scale, self.shift


@dataclass
class MixedGAMConfig:
    mode: str = "group_residual"
    scaler: str = "standard"
    base_hidden_units: Tuple[int, ...] = (64, 32)
    base_activation: str = "tanh"
    base_dropout: float = 0.05
    residual_hidden_units: Tuple[int, ...] = (64, )
    residual_activation: str = "relu"
    residual_dropout: float = 0.1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    n_epochs: int = 300
    lambda_center: float = 1e-3
    lambda_group_l1: float = 1e-4
    lambda_group_l2: float = 1e-4
    lambda_orth: float = 1e-4
    lambda_affine_scale: float = 1e-4
    lambda_affine_shift: float = 1e-4
    lambda_affine_center: float = 1e-4
    lambda_base_target: float = 1e-3
    verbose: bool = False
    random_state: int = 42
    device: Optional[str] = None
    num_shape_points: int = 200


class MixedGAMLightningModule(pl.LightningModule):
    """LightningModule wrapper to optimize the shared base + group head."""

    def __init__(
        self,
        base_gam: BaseGAM,
        head: nn.Module,
        config: MixedGAMConfig,
        mode: str,
        n_groups: int,
    ) -> None:
        super().__init__()
        self.base_gam = base_gam
        self.head = head
        self.cfg = config
        self.mode = mode
        self.n_groups = n_groups
        self.training_losses: List[float] = []

    def forward(
            self, xb: torch.Tensor,
            gb: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        contribs, base_sum = self.base_gam(xb)
        base_sum = base_sum.squeeze(-1)
        if self.mode == "group_residual":
            assert isinstance(self.head, GroupResidualHead)
            residual = self.head(xb, gb)
            preds = base_sum + residual
            return preds, {"base_sum": base_sum, "residual": residual}
        assert isinstance(self.head, GroupAffineHead)
        adjusted, total, delta_sum = self.head(contribs, gb)
        return total, {"base_sum": base_sum, "delta_sum": delta_sum}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor], batch_idx: int):
        xb, yb, gb = batch
        preds, aux = self(xb, gb)
        mse = F.mse_loss(preds, yb)
        if self.mode == "group_residual":
            regs = self._residual_regularization(aux["residual"],
                                                 aux["base_sum"], gb)
        else:
            regs = self._affine_regularization(aux["delta_sum"],
                                               aux["base_sum"])
        base_align = torch.zeros(1, device=xb.device)
        if self.cfg.lambda_base_target > 0:
            base_align = self.cfg.lambda_base_target * F.mse_loss(
                aux["base_sum"], yb)
        loss = mse + regs + base_align
        self.training_losses.append(loss.detach().cpu().item())
        self.log("train_loss",
                 loss,
                 prog_bar=False,
                 on_step=False,
                 on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.base_gam.parameters()) + list(self.head.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer

    def _residual_regularization(
        self,
        residual: torch.Tensor,
        base_sum: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        regs = torch.zeros(1, device=residual.device)

        if cfg.lambda_center > 0:
            sums = torch.zeros(self.n_groups,
                               device=residual.device).scatter_add_(
                                   0, group_idx, residual)
            counts = torch.zeros(self.n_groups,
                                 device=residual.device).scatter_add_(
                                     0, group_idx, torch.ones_like(residual))
            mask = counts > 0
            if mask.any():
                means = torch.zeros_like(sums)
                means[mask] = sums[mask] / counts[mask]
                regs = regs + cfg.lambda_center * torch.mean(means[mask]**2)

        if cfg.lambda_group_l1 > 0 and isinstance(self.head,
                                                  GroupResidualHead):
            l1_terms = [
                param.abs().mean() for param in self.head.group_params()
            ]
            regs = regs + cfg.lambda_group_l1 * torch.stack(l1_terms).sum()

        if cfg.lambda_group_l2 > 0 and isinstance(self.head,
                                                  GroupResidualHead):
            l2_terms = [(param**2).mean()
                        for param in self.head.group_params()]
            regs = regs + cfg.lambda_group_l2 * torch.stack(l2_terms).sum()

        if cfg.lambda_orth > 0:
            centered_base = base_sum - base_sum.mean()
            centered_residual = residual - residual.mean()
            cov = (centered_base * centered_residual).mean()
            regs = regs + cfg.lambda_orth * cov.pow(2)

        return regs

    def _affine_regularization(
        self,
        delta_sum: torch.Tensor,
        base_sum: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        regs = torch.zeros(1, device=delta_sum.device)

        if isinstance(self.head, GroupAffineHead):
            delta_scale, delta_shift = self.head.parameter_matrices()
            if cfg.lambda_affine_scale > 0:
                regs = regs + cfg.lambda_affine_scale * delta_scale.abs().mean(
                )
            if cfg.lambda_affine_shift > 0:
                regs = regs + cfg.lambda_affine_shift * delta_shift.abs().mean(
                )
            if cfg.lambda_affine_center > 0:
                regs = regs + cfg.lambda_affine_center * delta_shift.mean(
                ).pow(2)

        if cfg.lambda_orth > 0:
            centered_base = base_sum - base_sum.mean()
            centered_delta = delta_sum - delta_sum.mean()
            cov = (centered_base * centered_delta).mean()
            regs = regs + cfg.lambda_orth * cov.pow(2)

        return regs


class MixedGAMRegressor:
    """Hierarchical additive regressor with Lightning training and plotting utilities."""

    def __init__(
        self,
        metric_fn: Optional[MetricFn] = None,
        *,
        auto_search: bool = False,
        search_space: Optional[Sequence[Dict[str, Any]]] = None,
        search_metric: Optional[MetricFn] = None,
        search_val_ratio: float = 0.2,
        **kwargs: Any,
    ) -> None:
        cfg_fields = set(f.name for f in dataclass_fields(MixedGAMConfig))
        filtered = {
            key: value
            for key, value in kwargs.items() if key in cfg_fields
        }
        self.config = MixedGAMConfig(**filtered)
        self._init_kwargs = filtered
        self.mode = self.config.mode
        self.scaler_type = self.config.scaler
        self.random_state = self.config.random_state
        self.metric_fn = metric_fn
        self.auto_search = auto_search
        self.search_space = list(
            search_space) if search_space is not None else list(
                DEFAULT_HYPERPARAMETER_SPACE)
        self.search_metric = search_metric
        self.search_val_ratio = search_val_ratio
        self._search_performed = False
        self.search_summary: Optional[Dict[str, Any]] = None

        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        pl.seed_everything(self.random_state, workers=True)

        self.base_gam: Optional[BaseGAM] = None
        self.head: Optional[nn.Module] = None
        self.scaler: Optional[Any] = None
        self.group_encoder: Optional[GroupCrossEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self._feature_reference: Optional[np.ndarray] = None
        self._feature_ranges: Optional[np.ndarray] = None
        self._feature_name_to_idx: Dict[str, int] = {}
        self._n_features: Optional[int] = None
        self._n_groups: Optional[int] = None
        self._group_counts: Dict[str, int] = {}
        self.default_groups_: List[str] = []
        self._is_fitted = False
        self.training_history: List[float] = []

    def fit(
        self,
        X: ArrayLike,
        y: Sequence[float],
        groups: Optional[Sequence[Any]] = None,
    ) -> "MixedGAMRegressor":
        verbose_flag = self.config.verbose
        if verbose_flag:
            print(
                f"[MixedGAM] Training mode={self.mode} via PyTorch Lightning..."
            )

        X_array, feature_names = self._prepare_features(X)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        if groups is None:
            group_labels = np.array(["__global__"] * len(y_array))
        else:
            group_labels = np.asarray(groups).astype(str)
            if group_labels.shape[0] != len(y_array):
                raise ValueError("groups must match the number of rows in X.")

        self._maybe_run_auto_search(X, y, group_labels)

        self.feature_names = feature_names
        self._feature_name_to_idx = {
            name: idx
            for idx, name in enumerate(feature_names)
        }
        self._feature_reference = X_array.mean(axis=0)
        mins = X_array.min(axis=0)
        maxs = X_array.max(axis=0)
        self._feature_ranges = np.stack([mins, maxs], axis=1)
        self.scaler = self._build_scaler()
        X_scaled = self.scaler.fit_transform(X_array)

        self.group_encoder = GroupCrossEncoder()
        self.group_encoder.fit(group_labels)
        group_idx = self.group_encoder.transform(group_labels)
        self._group_counts = dict(
            zip(*np.unique(group_labels, return_counts=True)))
        self.default_groups_ = [
            g for g, _ in sorted(self._group_counts.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True)[:4]
        ]

        self._n_features = X_scaled.shape[1]
        self._n_groups = self.group_encoder.n_groups_
        self._build_networks()

        dataset = TensorDataset(
            torch.from_numpy(X_scaled).float(),
            torch.from_numpy(y_array).float(),
            torch.from_numpy(group_idx).long(),
        )
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        assert self.base_gam is not None and self.head is not None and self._n_groups is not None
        module = MixedGAMLightningModule(self.base_gam, self.head, self.config,
                                         self.mode, self._n_groups)

        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        trainer = pl.Trainer(
            max_epochs=self.config.n_epochs,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=verbose_flag,
        )
        trainer.fit(module, train_loader)

        self.training_history = module.training_losses
        self.base_gam = module.base_gam.eval()
        self.head = module.head.eval()
        self._is_fitted = True

        if verbose_flag:
            print(
                f"[MixedGAM] Finished training. Last loss={self.training_history[-1]:.5f}"
            )

        return self

    def predict(
        self,
        X: ArrayLike,
        groups: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        preds, _ = self.predict_and_contrib(X, groups=groups)
        return preds

    def predict_and_contrib(
        self,
        X: ArrayLike,
        groups: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self._check_fitted()
        assert self.base_gam is not None and self.head is not None and self.scaler is not None

        X_array, _ = self._prepare_features(
            X, expect_feature_names=self.feature_names)
        X_scaled = self.scaler.transform(X_array)
        group_idx = self._encode_groups(groups, len(X_scaled))

        with torch.no_grad():
            xb = torch.from_numpy(X_scaled).float().to(self.device)
            gb = torch.from_numpy(group_idx).long().to(self.device)
            contribs, base_sum = self.base_gam(xb)
            base_sum = base_sum.squeeze(-1)

            if self.mode == "group_residual":
                assert isinstance(self.head, GroupResidualHead)
                residual = self.head(xb, gb)
                preds = (base_sum + residual).cpu().numpy()
                contribution_dict = {
                    "per_feature": contribs.cpu().numpy(),
                    "group_residual": residual.cpu().numpy(),
                }
            else:
                assert isinstance(self.head, GroupAffineHead)
                adjusted, total, _ = self.head(contribs, gb)
                preds = total.cpu().numpy()
                contribution_dict = {
                    "per_feature": adjusted.cpu().numpy(),
                    "base_per_feature": contribs.cpu().numpy(),
                }

        return preds, contribution_dict

    def feature_importance(
        self,
        X: ArrayLike,
        groups: Optional[Sequence[Any]] = None,
    ) -> Dict[str, float]:
        _, contribs = self.predict_and_contrib(X, groups=groups)
        per_feature = contribs["per_feature"]
        importance = np.abs(per_feature).mean(axis=0)
        assert self.feature_names is not None
        return {
            name: float(importance[idx])
            for idx, name in enumerate(self.feature_names)
        }

    def get_feature_shape(
        self,
        feature: Union[int, str],
        grid: Optional[np.ndarray] = None,
        num: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the shared base shape function f_j(x_j)."""
        self._check_fitted()
        idx = self._resolve_feature_index(feature)
        grid_vals, xb = self._prepare_feature_grid(idx, grid, num)
        assert self.base_gam is not None
        with torch.no_grad():
            contribs, _ = self.base_gam(xb)
        shape = contribs[:, idx].cpu().numpy()
        return grid_vals, shape

    def get_group_affine_shape(
        self,
        feature: Union[int, str],
        group_id: Union[int, str],
        grid: Optional[np.ndarray] = None,
        num: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return group-calibrated per-feature contribution curves (affine mode only)."""
        if self.mode != "group_affine":
            raise RuntimeError(
                "Group affine shapes are only available in 'group_affine' mode."
            )
        self._check_fitted()
        assert isinstance(self.head, GroupAffineHead)

        idx = self._resolve_feature_index(feature)
        group_idx = self._resolve_group_index(group_id)
        grid_vals, xb = self._prepare_feature_grid(idx, grid, num)
        with torch.no_grad():
            contribs, _ = self.base_gam(xb)
            gb = torch.full((xb.shape[0], ),
                            group_idx,
                            dtype=torch.long,
                            device=xb.device)
            adjusted, _, _ = self.head(contribs, gb)
        shape = adjusted[:, idx].cpu().numpy()
        return grid_vals, shape

    def plot_feature_shapes(
        self,
        feature: Union[int, str],
        groups: Optional[Sequence[Union[int, str]]] = None,
        mode: str = "auto",
        grid: Optional[np.ndarray] = None,
        num: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        marker: Optional[str] = "o",
        markevery: Optional[int] = None,
        linewidth: float = 2.0,
    ) -> plt.Axes:
        """Plot global shape with optional group overlays."""
        self._check_fitted()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        idx = self._resolve_feature_index(feature)
        feature_name = self.feature_names[idx] if self.feature_names else str(
            feature)
        x_global, y_global = self.get_feature_shape(idx, grid=grid, num=num)
        spacing = markevery or max(1, len(x_global) // 20)
        global_kwargs = {
            "label": "Global base",
            "color": "black",
            "linewidth": linewidth
        }
        if marker:
            global_kwargs["marker"] = marker
            global_kwargs["markevery"] = spacing
        ax.plot(x_global, y_global, **global_kwargs)

        if mode == "global":
            ax.set_title(f"{feature_name} – Global Shape")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Contribution")
            ax.legend()
            return ax

        draw_groups = mode == "group" or (mode == "auto" and
                                          (groups or self.default_groups_))
        if draw_groups:
            target_groups = groups or self.default_groups_
            colors = plt.cm.tab10(np.linspace(0, 1, len(target_groups)))
            for color, group in zip(colors, target_groups):
                if self.mode == "group_affine":
                    gx, gy = self.get_group_affine_shape(idx,
                                                         group,
                                                         grid=grid,
                                                         num=num)
                    label = f"{group} (affine)"
                else:
                    gx, gy = self._simulate_group_response(
                        idx, group, grid, num)
                    label = f"{group} (residual)"
                kwargs = {
                    "label": label,
                    "color": color,
                    "linewidth": linewidth
                }
                if marker:
                    kwargs["marker"] = marker
                    kwargs["markevery"] = spacing
                ax.plot(gx, gy, **kwargs)
            ax.set_title(f"{feature_name} – Group Overlay")
        else:
            ax.set_title(f"{feature_name} – Global Shape")

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Contribution" if self.mode ==
                      "group_affine" else "Prediction")
        ax.legend()
        return ax

    def save(self, model_path: str | Path, extras_path: str | Path) -> None:
        self._check_fitted()
        assert self.base_gam is not None and self.head is not None
        torch.save(
            {
                "base_state_dict": self.base_gam.state_dict(),
                "head_state_dict": self.head.state_dict(),
                "mode": self.mode,
            },
            str(model_path),
        )
        joblib.dump(
            {
                "config":
                self._init_kwargs,
                "feature_names":
                self.feature_names,
                "scaler":
                self.scaler,
                "encoder_state":
                self.group_encoder.state_dict()
                if self.group_encoder else None,
                "n_features":
                self._n_features,
                "n_groups":
                self._n_groups,
                "mode":
                self.mode,
                "feature_reference":
                self._feature_reference,
                "feature_ranges":
                self._feature_ranges,
                "feature_name_to_idx":
                self._feature_name_to_idx,
                "default_groups":
                self.default_groups_,
            },
            str(extras_path),
        )

    @classmethod
    def load(cls, model_path: str | Path,
             extras_path: str | Path) -> "MixedGAMRegressor":
        extras = joblib.load(str(extras_path))
        model = cls(**extras.get("config", {}))
        model.mode = extras.get("mode", model.mode)
        model.feature_names = extras.get("feature_names")
        model.scaler = extras.get("scaler")
        encoder_state = extras.get("encoder_state")
        if encoder_state is not None:
            encoder = GroupCrossEncoder()
            encoder.load_state_dict(encoder_state)
            model.group_encoder = encoder
        model._n_features = extras.get("n_features")
        model._n_groups = extras.get("n_groups")
        model._feature_reference = extras.get("feature_reference")
        model._feature_ranges = extras.get("feature_ranges")
        model._feature_name_to_idx = extras.get("feature_name_to_idx", {})
        model.default_groups_ = extras.get("default_groups", [])
        model._build_networks()

        checkpoint = torch.load(str(model_path), map_location=model.device)
        assert model.base_gam is not None and model.head is not None
        model.base_gam.load_state_dict(checkpoint["base_state_dict"])
        model.head.load_state_dict(checkpoint["head_state_dict"])
        model.base_gam.eval()
        model.head.eval()
        model._is_fitted = True
        return model

    @classmethod
    def hyperparameter_search(
        cls,
        X: ArrayLike,
        y: Sequence[float],
        groups: Optional[Sequence[Any]] = None,
        param_grid: Optional[Sequence[Dict[str, Any]]] = None,
        metric_fn: Optional[MetricFn] = None,
        val_ratio: float = 0.2,
        base_kwargs: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Simple grid search helper with a user-defined metric."""
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be in (0, 1).")
        if not param_grid:
            param_grid = DEFAULT_HYPERPARAMETER_SPACE

        metric = metric_fn or default_metric
        X_array, y_array = cls._ensure_array_inputs(X, y)
        if groups is not None:
            groups_arr = np.asarray(groups).astype(str)
        else:
            groups_arr = np.array(["__global__"] * len(y_array))

        stratify = groups_arr if len(np.unique(groups_arr)) > 1 else None
        (
            X_train,
            X_val,
            y_train,
            y_val,
            g_train,
            g_val,
        ) = train_test_split(
            X_array,
            y_array,
            groups_arr,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify,
        )

        best_score: Optional[float] = None
        best_model: Optional[MixedGAMRegressor] = None
        best_params: Optional[Dict[str, Any]] = None
        all_results: List[Dict[str, Any]] = []

        for params in param_grid:
            cfg = dict(base_kwargs or {})
            cfg.update(params)
            model = cls(metric_fn=metric_fn, **cfg)
            model.fit(X_train, y_train, groups=g_train)
            preds = model.predict(X_val, groups=g_val)
            score = metric(y_val, preds)
            result = {"params": params, "score": score}
            all_results.append(result)
            if best_score is None or score < best_score:
                best_score = score
                best_model = model
                best_params = params

        return {
            "best_score": best_score,
            "best_params": best_params,
            "results": all_results,
            "best_model": best_model,
        }

    def _maybe_run_auto_search(
        self,
        X: ArrayLike,
        y: Sequence[float],
        groups: Sequence[str],
    ) -> None:
        if not self.auto_search or self._search_performed:
            return
        if self.config.verbose:
            print("[MixedGAM] Running automatic hyperparameter search...")
        base_kwargs = {
            field.name: getattr(self.config, field.name)
            for field in dataclass_fields(MixedGAMConfig)
        }
        param_grid = self.search_space or DEFAULT_HYPERPARAMETER_SPACE
        metric = self.search_metric or self.metric_fn or default_metric
        search_result = self.hyperparameter_search(
            param_grid=param_grid,
            X=X,
            y=y,
            groups=[str(g) for g in groups],
            metric_fn=metric,
            val_ratio=self.search_val_ratio,
            base_kwargs=base_kwargs,
            random_state=self.random_state,
        )
        self.search_summary = search_result
        best_params = search_result.get("best_params") or {}
        for key, value in best_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self._init_kwargs[key] = value
        self.mode = self.config.mode
        self.scaler_type = self.config.scaler
        self._search_performed = True
        if self.config.verbose and best_params:
            print(f"[MixedGAM] Auto-search best params: {best_params}")

    def _prepare_features(
        self,
        X: ArrayLike,
        expect_feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if isinstance(X, pd.DataFrame):
            values = X.values.astype(np.float32)
            names = list(X.columns)
        else:
            array = np.asarray(X, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            values = array
            if expect_feature_names is None and self.feature_names is None:
                names = [f"x{i}" for i in range(values.shape[1])]
            else:
                names = expect_feature_names or self.feature_names  # type: ignore[arg-type]
        return values, names

    def _build_scaler(self) -> Any:
        if self.scaler_type is None or self.scaler_type == "none":
            return _IdentityScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        return StandardScaler()

    def _build_networks(self) -> None:
        if self._n_features is None or self._n_groups is None:
            raise RuntimeError("Fit the model before building networks.")
        self.base_gam = BaseGAM(
            n_features=self._n_features,
            hidden_units=_ensure_tuple(self.config.base_hidden_units,
                                       (64, 32)),
            activation=self.config.base_activation,
            dropout=self.config.base_dropout,
        ).to(self.device)
        if self.mode == "group_residual":
            self.head = GroupResidualHead(
                input_dim=self._n_features,
                hidden_units=_ensure_tuple(self.config.residual_hidden_units,
                                           (64, )),
                n_groups=self._n_groups,
                activation=self.config.residual_activation,
                dropout=self.config.residual_dropout,
                unknown_index=self.group_encoder.unknown_index
                if self.group_encoder else None,
            ).to(self.device)
        elif self.mode == "group_affine":
            self.head = GroupAffineHead(
                n_features=self._n_features,
                n_groups=self._n_groups,
                unknown_index=self.group_encoder.unknown_index
                if self.group_encoder else None,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'.")

    def _encode_groups(self, groups: Optional[Sequence[Any]],
                       n_samples: int) -> np.ndarray:
        if self.group_encoder is None or self.group_encoder.unknown_index is None:
            raise RuntimeError(
                "Group encoder is unavailable. Fit the model first.")
        if groups is None:
            return np.full(n_samples,
                           self.group_encoder.unknown_index,
                           dtype=np.int64)
        labels = [str(label) for label in groups]
        if len(labels) != n_samples:
            raise ValueError("groups must match number of rows in X.")
        return self.group_encoder.transform(labels)

    def _prepare_feature_grid(
        self,
        feature_idx: int,
        grid: Optional[np.ndarray],
        num: Optional[int],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        if self._feature_reference is None or self._feature_ranges is None:
            raise RuntimeError("Call `fit` before requesting shape functions.")

        if grid is None:
            num_points = num or self.config.num_shape_points
            low, high = self._feature_ranges[feature_idx]
            if np.isclose(low, high):
                low -= 1.0
                high += 1.0
            grid = np.linspace(low, high, num=num_points, dtype=np.float32)
        else:
            grid = np.asarray(grid, dtype=np.float32)

        base = np.tile(self._feature_reference, (len(grid), 1))
        base[:, feature_idx] = grid
        assert self.scaler is not None
        scaled = self.scaler.transform(base)
        xb = torch.from_numpy(scaled).float().to(self.device)
        return grid, xb

    def _simulate_group_response(
        self,
        feature: Union[int, str],
        group_id: Union[int, str],
        grid: Optional[np.ndarray],
        num: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._resolve_feature_index(feature)
        group_idx = self._resolve_group_index(group_id)
        grid_vals, xb = self._prepare_feature_grid(idx, grid, num)
        gb = torch.full((xb.shape[0], ),
                        group_idx,
                        dtype=torch.long,
                        device=xb.device)
        assert self.base_gam is not None and self.head is not None
        with torch.no_grad():
            contribs, base_sum = self.base_gam(xb)
            base_sum = base_sum.squeeze(-1)
            if self.mode == "group_residual":
                residual = self.head(xb, gb)  # type: ignore[arg-type]
                preds = base_sum + residual
                values = preds.cpu().numpy()
            else:
                assert isinstance(self.head, GroupAffineHead)
                adjusted, _, _ = self.head(contribs, gb)
                values = adjusted[:, idx].cpu().numpy()
        return grid_vals, values

    def _resolve_feature_index(self, feature: Union[int, str]) -> int:
        if isinstance(feature, int):
            return feature
        if not self._feature_name_to_idx:
            raise RuntimeError(
                "Feature metadata unavailable. Fit the model first.")
        if feature not in self._feature_name_to_idx:
            raise KeyError(f"Unknown feature '{feature}'.")
        return self._feature_name_to_idx[feature]

    def _resolve_group_index(self, group: Union[int, str]) -> int:
        if self.group_encoder is None or self.group_encoder.unknown_index is None:
            raise RuntimeError("Group encoder unavailable.")
        if isinstance(group, int):
            return group
        normalized = self.group_encoder._normalize(group)
        return self.group_encoder.mapping.get(normalized,
                                              self.group_encoder.unknown_index)

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.base_gam is None or self.head is None:
            raise RuntimeError("Call `fit` before using MixedGAMRegressor.")

    @staticmethod
    def _ensure_array_inputs(
        X: ArrayLike,
        y: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        return X_array, y_array


def default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred)**2))


def dataclass_fields(cls: Any) -> List[Any]:
    return list(getattr(cls, "__dataclass_fields__", {}).values())
