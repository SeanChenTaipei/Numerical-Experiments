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

from dataclasses import replace
from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import sys

project_root = Path.cwd().resolve()
if not (project_root / "src").exists():
    project_root = project_root.parent
if project_root.exists() and str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print(f"Using project root: {project_root}")

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .helpers import (
    BaseGAM,
    ConditionalSNN,
    GroupAffineHead,
    GroupResidualHead,
    MixedGAMConfig,
)

ArrayLike = Union[pd.DataFrame, np.ndarray]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _ensure_tuple(units: Optional[Sequence[int]],
                  default: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(units) if units else default


def default_ray_search_space(tune: Any, max_epochs: int) -> Dict[str, Any]:
    """Factory for the fallback Ray Tune search space."""
    return {
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "base_hidden_units": tune.choice([(64, 32), (96, 48), (128, 64, 32)]),
        "residual_hidden_units": tune.choice([(32, ), (48, ), (64, 32)]),
        "base_dropout": tune.uniform(0.05, 0.25),
        "residual_dropout": tune.uniform(0.05, 0.4),
        "base_norm": tune.choice(["layernorm", "none"]),
        "residual_norm": tune.choice(["layernorm", "none"]),
        "lambda_group_l1": tune.loguniform(1e-5, 5e-4),
        "lambda_group_l2": tune.loguniform(1e-5, 5e-4),
        "residual_contribution_l1": tune.choice([0.0, 5e-5, 1e-4]),
        "batch_size": tune.choice([128, 256, 512]),
        "n_epochs": tune.choice([max_epochs // 2, max_epochs]),
        "mode": tune.choice(["group_residual", "group_affine"]),
        "scaler": tune.choice(["standard", "robust"]),
    }


class _IdentityScaler:
    """Simple passthrough scaler with a scikit-learn compatible API."""

    def fit(self, X: np.ndarray) -> "_IdentityScaler":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


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


class MixedGAMRegressor:
    """Hierarchical additive regressor with Lightning training and plotting utilities."""

    def __init__(
        self,
        metric_fn: Optional[MetricFn] = None,
        *,
        auto_search: bool = False,
        search_space: Optional[Dict[str, Any]] = None,
        search_metric: str = "val_loss",
        search_mode: str = "min",
        search_num_samples: int = 20,
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
        self.search_space = search_space
        self.search_metric = search_metric or "val_loss"
        if search_mode not in {"min", "max"}:
            raise ValueError("search_mode must be either 'min' or 'max'.")
        self.search_mode = search_mode
        if search_num_samples <= 0:
            raise ValueError("search_num_samples must be a positive integer.")
        self.search_num_samples = int(search_num_samples)
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
        self.target_scaler: Optional[Any] = None
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
        self.cached_contribs_: Optional[Dict[str, np.ndarray]] = None

    def fit(
        self,
        X: ArrayLike,
        y: Sequence[float],
        groups: Optional[Sequence[Any]] = None,
        *,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[Sequence[float]] = None,
        groups_val: Optional[Sequence[Any]] = None,
        val_ratio: Optional[float] = 0.2,
        callbacks: Optional[Sequence[Callback]] = None,
        progress_refresh_rate: int = 10,
    ) -> "MixedGAMRegressor":
        """
        分步訓練 MixedGAMRegressor。

        步驟：
            1. 轉成 numpy 陣列並檢查 X/y/groups 尺寸，必要時切分驗證集。
            2. 擬合特徵縮放器與目標縮放器，同時計算 shape function 所需的參考統計。
            3. 編碼群組 ID、建立 Lightning DataLoader 與 GAM/Head 網路。
            4. 設定 tqdm 進度條、EarlyStopping、LR monitor 與 ModelCheckpoint 後開始訓練。
            5. 訓練完成後使用全量資料呼叫 `predict_and_contrib`，快取 feature 貢獻供 feature_importance 使用。

        參數：
            X: 特徵矩陣 (DataFrame 或 ndarray)。
            y: 目標值向量。
            groups: 每筆資料的群組標籤，若未提供則以 `__global__` 視為單一群組。
            X_val / y_val / groups_val: 額外驗證集；缺省時依 `val_ratio` 切分。
            val_ratio: 驗證集比例。
            callbacks: 自訂 Lightning callback。
            progress_refresh_rate: tqdm 進度條更新頻率。

        返回：
            已訓練完成的 `MixedGAMRegressor`。
        """
        verbose_flag = self.config.verbose
        logger_flag = bool(verbose_flag)
        if verbose_flag:
            print(
                f"[MixedGAM] Training mode={self.mode} via PyTorch Lightning..."
            )
        self.cached_contribs_ = None

        # 步驟 1：將輸入轉為 numpy 陣列並確認維度
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

        # 步驟 2：若啟用 Ray Tune，自動搜尋超參數
        self._maybe_run_auto_search(X, y, group_labels)

        X_train_array = X_array
        y_train_array = y_array
        groups_train = group_labels
        X_val_array: Optional[np.ndarray] = None
        y_val_array: Optional[np.ndarray] = None
        groups_val_array: Optional[np.ndarray] = None

        # 步驟 3：判斷是否提供外部驗證集，否則依比例切分
        explicit_val = any(value is not None
                           for value in (X_val, y_val, groups_val))

        if explicit_val:
            if X_val is None or y_val is None:
                raise ValueError(
                    "Both X_val and y_val must be provided for validation.")
            X_val_array, _ = self._prepare_features(
                X_val, expect_feature_names=feature_names)
            y_val_array = np.asarray(y_val, dtype=np.float32).reshape(-1)
            if X_val_array.shape[0] != y_val_array.shape[0]:
                raise ValueError(
                    "X_val and y_val must contain the same number of rows.")
            if groups_val is None:
                groups_val_array = np.array(["__global__"] * len(y_val_array))
            else:
                groups_val_array = np.asarray(groups_val).astype(str)
                if groups_val_array.shape[0] != len(y_val_array):
                    raise ValueError(
                        "groups_val must match the number of rows in X_val.")
        elif val_ratio is not None and val_ratio > 0:
            if not 0 < val_ratio < 1:
                raise ValueError("val_ratio must be between 0 and 1.")
            stratify = group_labels if len(
                np.unique(group_labels)) > 1 else None
            (
                X_train_array,
                X_val_array,
                y_train_array,
                y_val_array,
                groups_train,
                groups_val_array,
            ) = train_test_split(
                X_array,
                y_array,
                group_labels,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=stratify,
            )

        X_train_array = np.asarray(X_train_array, dtype=np.float32)
        y_train_array = np.asarray(y_train_array, dtype=np.float32).reshape(-1)
        groups_train = np.asarray(groups_train).astype(str)
        if groups_val_array is not None:
            X_val_array = np.asarray(
                X_val_array,
                dtype=np.float32) if X_val_array is not None else None
            y_val_array = np.asarray(y_val_array, dtype=np.float32).reshape(-1)
            groups_val_array = np.asarray(groups_val_array).astype(str)

        self.feature_names = feature_names
        self._feature_name_to_idx = {
            name: idx
            for idx, name in enumerate(feature_names)
        }
        self._feature_reference = X_train_array.mean(axis=0)
        mins = X_train_array.min(axis=0)
        maxs = X_train_array.max(axis=0)
        self._feature_ranges = np.stack([mins, maxs], axis=1)
        # 步驟 4：擬合特徵與目標縮放器
        self.scaler = self._build_scaler()
        self.target_scaler = self._build_scaler()
        y_train_scaled = self.target_scaler.fit_transform(
            y_train_array.reshape(-1, 1)).reshape(-1)
        X_train_scaled = self.scaler.fit_transform(X_train_array)
        X_val_scaled: Optional[np.ndarray] = None
        y_val_scaled: Optional[np.ndarray] = None
        if X_val_array is not None:
            X_val_scaled = self.scaler.transform(X_val_array)
        if y_val_array is not None:
            y_val_scaled = self.target_scaler.transform(
                y_val_array.reshape(-1, 1)).reshape(-1)

        # 步驟 5：編碼群組並建立資料載入器
        self.group_encoder = GroupCrossEncoder()
        self.group_encoder.fit(groups_train.tolist())
        group_idx = self.group_encoder.transform(groups_train.tolist())
        group_idx_val: Optional[np.ndarray] = None
        if groups_val_array is not None:
            group_idx_val = self.group_encoder.transform(
                groups_val_array.tolist())

        self._group_counts = dict(
            zip(*np.unique(groups_train, return_counts=True)))
        self.default_groups_ = [
            g for g, _ in sorted(self._group_counts.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True)[:4]
        ]

        self._n_features = X_train_scaled.shape[1]
        self._n_groups = self.group_encoder.n_groups_
        self._build_networks()

        dataset = TensorDataset(
            torch.from_numpy(X_train_scaled).float(),
            torch.from_numpy(y_train_scaled).float(),
            torch.from_numpy(group_idx).long(),
        )
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader: Optional[DataLoader] = None
        if X_val_scaled is not None and y_val_scaled is not None and group_idx_val is not None:
            val_dataset = TensorDataset(
                torch.from_numpy(X_val_scaled).float(),
                torch.from_numpy(y_val_scaled).float(),
                torch.from_numpy(group_idx_val).long(),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
            )

        assert self.base_gam is not None and self.head is not None and self._n_groups is not None
        module = ConditionalSNN(self.base_gam, self.head, self.config,
                                self.mode, self._n_groups)

        # 步驟 6：建立 tqdm 進度條與監控回呼
        callbacks_list: List[Callback] = list(callbacks) if callbacks else []
        if not any(isinstance(cb, TQDMProgressBar) for cb in callbacks_list):
            callbacks_list.append(
                TQDMProgressBar(refresh_rate=max(1, progress_refresh_rate)))
        # ---- default callbacks injection ----
        have_es = any(isinstance(cb, EarlyStopping) for cb in callbacks_list)
        have_lr = any(
            isinstance(cb, LearningRateMonitor) for cb in callbacks_list)
        have_ckpt = any(
            isinstance(cb, ModelCheckpoint) for cb in callbacks_list)

        # EarlyStopping 只在有驗證集時才有意義；沒有 val_loader 時 Lightning 會自動忽略
        if not have_es:
            callbacks_list.append(
                EarlyStopping(monitor="val_loss",
                              patience=self.config.patience,
                              mode="min"))
        if not logger_flag and have_lr:
            raise ValueError(
                "LearningRateMonitor requires verbose=True so that the Trainer logger is enabled."
            )
        if logger_flag and not have_lr:
            callbacks_list.append(
                LearningRateMonitor(logging_interval="epoch"))
        if not have_ckpt:
            ckpt_dir = Path(self.config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            callbacks_list.append(
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename="mixed-gam-{epoch:03d}-{val_loss:.4f}",
                    monitor=self.config.checkpoint_monitor,
                    mode=self.config.checkpoint_mode,
                    save_top_k=1,
                    save_last=True,
                ))

        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        trainer = pl.Trainer(
            max_epochs=self.config.n_epochs,
            accelerator=accelerator,
            devices=1,
            logger=logger_flag,
            enable_model_summary=False,
            enable_progress_bar=True,
            callbacks=callbacks_list,
        )
        trainer.fit(module, train_loader, val_loader)

        self.training_history = module.training_losses
        self.base_gam = module.base_gam.eval()
        self.head = module.head.eval()
        self._is_fitted = True
        # 步驟 7：快取全量資料的貢獻，便於後續重要性分析
        self._cache_training_contribs(X_array, group_labels)

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
        """Return predictions together with decomposed terms.

        Examples:
            >>> preds, contribs = model.predict_and_contrib(X_test, groups=g_test)
            >>> base_terms = contribs["base_per_feature"]        # shape: (n_samples, n_features)
            >>> bias = contribs["bias"]                          # scalar bias broadcast to each sample
            >>> if model.mode == "group_residual":
            ...     residual = contribs["residual_per_feature"]  # adapter deltas per feature
            ...     per_feature = contribs["per_feature"]        # base + residual for each feature
            >>> else:
            ...     before_affine = contribs["base_per_feature"] # contributions before affine adapter
            ...     after_affine = contribs["per_feature"]       # contributions after affine adapter
        """
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
            bias_value = float(self.base_gam.bias.detach().cpu().item())
            bias_vec = np.full((len(X_scaled), ), bias_value, dtype=np.float32)

            if self.mode == "group_residual":
                assert isinstance(self.head, GroupResidualHead)
                residual = self.head(xb, gb)
                residual_sum = residual.sum(dim=1)
                preds_scaled = (base_sum + residual_sum).cpu().numpy()
                base_np = contribs.cpu().numpy()
                residual_np = residual.cpu().numpy()
                contribution_dict = {
                    "per_feature": base_np + residual_np,
                    "base_per_feature": base_np,
                    "residual_per_feature": residual_np,
                    "bias": bias_vec,
                    "residual_sum": residual_np.sum(axis=1),
                }
            else:
                assert isinstance(self.head, GroupAffineHead)
                adjusted, total, _ = self.head(contribs, gb)
                total_np = total.cpu().numpy()
                preds_scaled = total_np + bias_vec
                contribution_dict = {
                    "per_feature": adjusted.cpu().numpy(),
                    "base_per_feature": contribs.cpu().numpy(),
                    "bias": bias_vec,
                }

        preds = self._inverse_target_predictions(preds_scaled)
        contribution_dict = self._rescale_contributions(contribution_dict)
        return preds, contribution_dict

    def _inverse_target_predictions(self, preds: np.ndarray) -> np.ndarray:
        if self.target_scaler is None or not hasattr(self.target_scaler,
                                                     "inverse_transform"):
            return preds
        restored = self.target_scaler.inverse_transform(preds.reshape(
            -1, 1)).reshape(-1)
        return restored

    def _target_scale_offset(self) -> Tuple[float, float]:
        if self.target_scaler is None:
            return 1.0, 0.0
        scale_attr = getattr(self.target_scaler, "scale_", None)
        if scale_attr is None:
            scale = 1.0
        else:
            scale = float(np.asarray(scale_attr).reshape(-1)[0])
        offset_attr = None
        if hasattr(self.target_scaler, "mean_"):
            offset_attr = getattr(self.target_scaler, "mean_")
        elif hasattr(self.target_scaler, "center_"):
            offset_attr = getattr(self.target_scaler, "center_")
        offset = 0.0 if offset_attr is None else float(
            np.asarray(offset_attr).reshape(-1)[0])
        return scale, offset

    def _rescale_contributions(
            self, contribs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        scale, offset = self._target_scale_offset()
        if np.isclose(scale, 1.0) and np.isclose(offset, 0.0):
            return contribs
        adjusted: Dict[str, np.ndarray] = {}
        for key, value in contribs.items():
            arr = np.asarray(value, dtype=np.float32)
            if key == "bias":
                adjusted[key] = arr * scale + offset
            else:
                adjusted[key] = arr * scale
        return adjusted

    def feature_importance(
        self,
        X: Optional[ArrayLike] = None,
        groups: Optional[Sequence[Any]] = None,
    ) -> Dict[str, float]:
        self._check_fitted()
        if X is None:
            if self.cached_contribs_ is None:
                raise ValueError(
                    "Cached contributions are unavailable. Provide X and groups explicitly."
                )
            contribs = self.cached_contribs_
        else:
            _, contribs = self.predict_and_contrib(X, groups=groups)
        per_feature = contribs["per_feature"]
        importance = np.abs(per_feature).mean(axis=0)
        assert self.feature_names is not None
        return {
            name: float(importance[idx])
            for idx, name in enumerate(self.feature_names)
        }

    def _cache_training_contribs(self, X_full: np.ndarray,
                                 group_labels: Sequence[str]) -> None:
        if not self._is_fitted:
            self.cached_contribs_ = None
            return
        try:
            _, contribs = self.predict_and_contrib(X_full, groups=group_labels)
        except Exception as exc:  # pragma: no cover - best effort
            if self.config.verbose:
                print(
                    f"[MixedGAM] Failed to cache training contributions: {exc}"
                )
            self.cached_contribs_ = None
        else:
            self.cached_contribs_ = contribs

    def get_feature_shape(
        self,
        feature: Union[int, str],
        grid: Optional[np.ndarray] = None,
        num: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the shared base shape function :math:`f_j(x_j)`.

        Example:
            >>> xs, fx = model.get_feature_shape("age")
            >>> ax.plot(xs, fx)
        """
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
        """Return group-calibrated per-feature contributions (affine mode only)."""
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
        """Plot the global base shape with optional group overlays.

        In ``group_residual`` mode, group overlays use the adjusted
        per-feature contributions (base + residual for the selected feature).
        """
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
        ax.set_ylabel("Contribution")
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
                "target_scaler":
                self.target_scaler,
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
                "cached_contribs":
                self.cached_contribs_,
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
        model.target_scaler = extras.get("target_scaler")
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
        model.cached_contribs_ = extras.get("cached_contribs")
        model._build_networks()

        checkpoint = torch.load(str(model_path), map_location=model.device)
        assert model.base_gam is not None and model.head is not None
        model.base_gam.load_state_dict(checkpoint["base_state_dict"])
        model.head.load_state_dict(checkpoint["head_state_dict"])
        model.base_gam.eval()
        model.head.eval()
        model._is_fitted = True
        return model


class ConditionalSNNRegressor(MixedGAMRegressor):
    """Alias for ``MixedGAMRegressor`` with branding aligned to ConditionalSNN."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def ray_tune_search(
        cls,
        X: ArrayLike,
        y: Sequence[float],
        *,
        groups: Optional[Sequence[Any]] = None,
        param_space: Optional[Dict[str, Any]] = None,
        num_samples: int = 20,
        metric: str = "val_loss",
        mode: str = "min",
        val_ratio: float = 0.2,
        scheduler: Optional[Any] = None,
        resources_per_trial: Optional[Dict[str, int]] = None,
        max_epochs: Optional[int] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Hyper-parameter search with Ray Tune + ASHAScheduler.

        Example:
            >>> search = MixedGAMRegressor.ray_tune_search(X, y, groups=g)
            >>> best_cfg = search["best_config"]
            >>> model = MixedGAMRegressor(**best_cfg)
            >>> model.fit(X, y, groups=g)
        """
        try:
            from ray import tune
            from ray.tune import Tuner
            from ray.tune.schedulers import ASHAScheduler
        except ImportError as exc:
            raise ImportError(
                "Ray Tune is required. Install with `pip install ray[tune]`."
            ) from exc

        legacy_choice_key = "__mixed_gam_choice__"

        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1 for Ray Tune.")

        base_cfg = MixedGAMConfig()
        max_epochs_value = max_epochs or base_cfg.n_epochs

        X_array, y_array = cls._ensure_array_inputs(X, y)
        if groups is None:
            group_labels = np.array(["__global__"] * len(y_array))
        else:
            group_labels = np.asarray(groups).astype(str)

        stratify = group_labels if len(np.unique(group_labels)) > 1 else None
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
            group_labels,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify,
        )

        resolved_space: Optional[Dict[str, Any]] = None
        if param_space is None:
            resolved_space = None
        elif isinstance(param_space, dict):
            resolved_space = dict(param_space)
        elif isinstance(param_space, Sequence):
            candidates = list(param_space)
            if candidates and all(
                    isinstance(item, dict) for item in candidates):
                resolved_space = {legacy_choice_key: tune.choice(candidates)}
            else:
                raise TypeError(
                    "param_space sequences must contain dictionaries for Ray Tune."
                )
        else:
            raise TypeError(
                "param_space must be a Ray Tune searchable dict or a sequence of dicts."
            )

        if resolved_space is None:
            resolved_space = default_ray_search_space(tune, max_epochs_value)
        param_space = resolved_space

        if resources_per_trial is None:
            resources_per_trial = {
                "cpu": 2,
                "gpu": 1 if torch.cuda.is_available() else 0,
            }

        if scheduler is None:
            scheduler = ASHAScheduler(
                time_attr="training_iteration",
                max_t=max_epochs_value,
                grace_period=max(1, max_epochs_value // 10),
                reduction_factor=2,
            )

        def _coerce_cfg(ray_cfg: Dict[str, Any]) -> MixedGAMConfig:
            cfg_kwargs: Dict[str, Any] = {}
            flat_cfg = dict(ray_cfg)
            legacy_payload = flat_cfg.pop(legacy_choice_key, None)
            if isinstance(legacy_payload, dict):
                flat_cfg.update(legacy_payload)
            for key, value in flat_cfg.items():
                if not hasattr(base_cfg, key):
                    continue
                if key in {"base_hidden_units", "residual_hidden_units"}:
                    if isinstance(value, tuple):
                        cfg_kwargs[key] = value
                    elif isinstance(value, Sequence):
                        cfg_kwargs[key] = tuple(int(v) for v in value)
                    else:
                        cfg_kwargs[key] = (int(value), )
                else:
                    cfg_kwargs[key] = value
            return replace(base_cfg, **cfg_kwargs)

        def _trainable(ray_cfg: Dict[str, Any]) -> None:
            project_root = Path(__file__).resolve().parents[2]
            project_str = str(project_root)
            if project_str not in sys.path:
                sys.path.append(project_str)
            cfg = _coerce_cfg(ray_cfg)
            pl.seed_everything(cfg.random_state, workers=True)
            scaler = cls._make_scaler(cfg.scaler)
            target_scaler = cls._make_scaler(cfg.scaler)
            X_tr_scaled = scaler.fit_transform(X_train).astype(np.float32)
            X_va_scaled = scaler.transform(X_val).astype(np.float32)
            y_tr_scaled = target_scaler.fit_transform(y_train.reshape(
                -1, 1)).astype(np.float32).reshape(-1)
            y_va_scaled = target_scaler.transform(y_val.reshape(-1, 1)).astype(
                np.float32).reshape(-1)

            encoder = GroupCrossEncoder()
            encoder.fit(g_train.tolist())
            g_tr_idx = encoder.transform(g_train.tolist())
            g_va_idx = encoder.transform(g_val.tolist())

            n_features = X_tr_scaled.shape[1]
            base_gam = BaseGAM(
                n_features=n_features,
                hidden_units=_ensure_tuple(cfg.base_hidden_units, (96, 48)),
                activation=cfg.base_activation,
                dropout=cfg.base_dropout,
                parallel_features=cfg.parallel_feature_mlp,
                norm=cfg.base_norm,
            )
            if cfg.mode == "group_residual":
                head: nn.Module = GroupResidualHead(
                    input_dim=n_features,
                    hidden_units=_ensure_tuple(cfg.residual_hidden_units,
                                               (48, )),
                    n_groups=encoder.n_groups_,
                    n_features=n_features,
                    activation=cfg.residual_activation,
                    dropout=cfg.residual_dropout,
                    unknown_index=encoder.unknown_index,
                    norm=cfg.residual_norm,
                )
            elif cfg.mode == "group_affine":
                head = GroupAffineHead(
                    n_features=n_features,
                    n_groups=encoder.n_groups_,
                    unknown_index=encoder.unknown_index,
                )
            else:
                raise ValueError(
                    f"Unsupported mode '{cfg.mode}' inside Ray Tune.")

            train_ds = TensorDataset(
                torch.from_numpy(X_tr_scaled),
                torch.from_numpy(y_tr_scaled),
                torch.from_numpy(g_tr_idx),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_va_scaled),
                torch.from_numpy(y_va_scaled),
                torch.from_numpy(g_va_idx),
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=False,
            )
            module = ConditionalSNN(base_gam, head, cfg, cfg.mode,
                                    encoder.n_groups_)
            trainer = pl.Trainer(
                max_epochs=cfg.n_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                enable_checkpointing=False,
                logger=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )
            trainer.fit(module, train_loader, val_loader)
            metric_value = trainer.callback_metrics.get(metric)
            if isinstance(metric_value, torch.Tensor):
                metric_value = float(metric_value.item())
            elif metric_value is None:
                metric_value = float("inf")
            tune.report(**{metric: float(metric_value)})

        tuner = Tuner(
            tune.with_resources(_trainable, resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                num_samples=num_samples,
                scheduler=scheduler,
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}"),
            param_space=param_space,
        )
        results = tuner.fit()
        best = results.get_best_result(metric=metric, mode=mode)
        trial_summaries = [{
            "config":
            result.config,
            metric:
            result.metrics.get(metric),
            "training_iteration":
            result.metrics.get("training_iteration"),
        } for result in results]
        return {
            "result_grid": results,
            "best_config": best.config,
            "best_metric": best.metrics.get(metric),
            "trials": trial_summaries,
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
            print("[MixedGAM] Running Ray Tune auto-search...")

        try:
            search_result = self.ray_tune_search(
                X,
                y,
                groups=[str(g) for g in groups],
                param_space=self.search_space,
                num_samples=self.search_num_samples,
                metric=self.search_metric,
                mode=self.search_mode,
                val_ratio=self.search_val_ratio,
                random_state=self.random_state,
            )
        except ImportError:
            if self.config.verbose:
                print(
                    "[MixedGAM] Skipping auto-search because Ray Tune is not installed."
                )
            return
        best_params = search_result.get("best_config") or {}
        for key, value in best_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self._init_kwargs[key] = value
        self.mode = self.config.mode
        self.scaler_type = self.config.scaler
        self.search_summary = {
            "best_config": best_params,
            "best_metric": search_result.get("best_metric"),
        }
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

    @staticmethod
    def _make_scaler(scaler_type: Optional[str]) -> Any:
        if scaler_type is None or scaler_type == "none":
            return _IdentityScaler()
        if scaler_type == "robust":
            return RobustScaler()
        if scaler_type == "standard":
            return StandardScaler()
        raise ValueError(
            f"Unsupported scaler '{scaler_type}'. Expected 'standard', 'robust', or 'none'."
        )

    def _build_scaler(self) -> Any:
        return self._make_scaler(self.scaler_type)

    def _build_networks(self) -> None:
        if self._n_features is None or self._n_groups is None:
            raise RuntimeError("Fit the model before building networks.")
        self.base_gam = BaseGAM(
            n_features=self._n_features,
            hidden_units=_ensure_tuple(self.config.base_hidden_units,
                                       (96, 48)),
            activation=self.config.base_activation,
            dropout=self.config.base_dropout,
            parallel_features=self.config.parallel_feature_mlp,
            norm=self.config.base_norm,
        ).to(self.device)
        if self.mode == "group_residual":
            self.head = GroupResidualHead(
                input_dim=self._n_features,
                hidden_units=_ensure_tuple(self.config.residual_hidden_units,
                                           (48, )),
                n_groups=self._n_groups,
                n_features=self._n_features,
                activation=self.config.residual_activation,
                dropout=self.config.residual_dropout,
                unknown_index=self.group_encoder.unknown_index
                if self.group_encoder else None,
                norm=self.config.residual_norm,
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
            contribs, _ = self.base_gam(xb)
            if self.mode == "group_residual":
                residual = self.head(xb, gb)  # type: ignore[arg-type]
                adjusted = contribs + residual
                values = adjusted[:, idx].cpu().numpy()
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
