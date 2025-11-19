"""Generic categorical feature encoder with auto-strategy selection.

This module exposes SmartCategoricalEncoder, a pluggable transformer that
supports a broad set of categorical encoding schemes, automatic strategy
selection based on observed statistics, and a registry that allows external
encoders to be plugged in without editing the core class.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


class EncoderConfigurationError(ValueError):
    """Raised when an encoder cannot be configured with the provided parameters."""


class EncoderFitError(RuntimeError):
    """Raised when an encoder cannot be fitted because of invalid inputs."""


def _to_float_series(target: Optional[Union[pd.Series, Sequence[Any]]]) -> Optional[pd.Series]:
    """Convert target arrays to float series when needed."""
    if target is None:
        return None
    if isinstance(target, pd.Series):
        return target.astype(float)
    return pd.Series(target, dtype=float)


@dataclass
class ColumnStats:
    """Container for descriptive statistics computed per column."""

    cardinality: int
    unique_ratio: float
    rare_share: float
    missing_rate: float
    most_common_freq: float


class _BaseColumnEncoder:
    """Minimal interface every concrete encoder must honor.

    Subclasses operate on a single pandas Series and must implement ``fit`` and
    ``transform``. To expose a custom encoder through SmartCategoricalEncoder,
    inherit from this base class and register it via
    ``SmartCategoricalEncoder.register_encoder('name', CustomEncoder)``.
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.feature_names_: List[str] = []

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_BaseColumnEncoder":
        raise NotImplementedError

    def transform(self, series: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

    def get_feature_names(self) -> List[str]:
        return self.feature_names_


class _OneHotColumnEncoder(_BaseColumnEncoder):
    """Applies sklearn's OneHotEncoder to a single column.

    Usage: set ``strategy='onehot'`` or map ``{'col': 'onehot'}`` when
    instantiating SmartCategoricalEncoder. Optional params: ``drop`` (for
    example ``'if_binary'``) forwarded via ``params['onehot']``.
    """
    def __init__(self, column: str, drop: Optional[str] = None) -> None:
        super().__init__(column)
        self.drop = drop
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, drop=drop)

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_OneHotColumnEncoder":
        self.encoder.fit(series.to_frame())
        self.feature_names_ = [
            f"{self.column}__{cat}" for cat in self.encoder.categories_[0]
        ]
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        transformed = self.encoder.transform(series.to_frame())
        return pd.DataFrame(transformed, columns=self.feature_names_, index=series.index)


class _OrdinalColumnEncoder(_BaseColumnEncoder):
    """Maps categories to integers while reserving ``-1`` for unseen values.

    Usage: choose ``strategy='ordinal'`` or per-column mapping.
    Works best when approximate ordering is acceptable or downstream models
    (tree ensembles) can handle arbitrary integer labels.
    """
    def __init__(self, column: str) -> None:
        super().__init__(column)
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_OrdinalColumnEncoder":
        self.encoder.fit(series.to_frame())
        self.feature_names_ = [f"{self.column}__ordinal"]
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        encoded = self.encoder.transform(series.to_frame())
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _TargetMeanEncoder(_BaseColumnEncoder):
    """Classic mean target encoding with logistic smoothing and optional noise.

    Usage: ``strategy='target'`` (requires a target vector). Configure
    smoothing/noise via ``params['target']`` for SmartCategoricalEncoder.
    """
    def __init__(self, column: str, smoothing: float = 10.0, noise: float = 0.0) -> None:
        super().__init__(column)
        self.smoothing = smoothing
        self.noise = noise
        self.global_mean_: float = 0.0
        self.mapping_: Dict[Any, float] = {}
        self.feature_names_ = [f"{self.column}__target_mean"]

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_TargetMeanEncoder":
        if y is None:
            raise EncoderFitError("Target encoding requires 'y'.")
        y_series = _to_float_series(y)
        df = pd.DataFrame({"feature": series, "target": y_series})
        grouped = df.groupby("feature", dropna=False)["target"]
        stats = grouped.agg(["mean", "count"])
        self.global_mean_ = float(y_series.mean())
        smoothing = 1.0 / (1.0 + np.exp(-(stats["count"] - self.smoothing)))
        smooth = self.global_mean_ * (1 - smoothing) + stats["mean"] * smoothing
        self.mapping_ = smooth.to_dict()
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        values = series.map(self.mapping_).fillna(self.global_mean_)
        if self.noise > 0.0:
            noise = np.random.normal(0, self.noise, size=len(values))
            values = values + noise
        return pd.DataFrame(values, columns=self.feature_names_, index=series.index)


class _LeaveOneOutEncoder(_BaseColumnEncoder):
    """Row-wise target mean that excludes the current observation.

    Usage: ``strategy='leave_one_out'`` with a supplied ``y`` series. Helpful
    when leakage must be minimized; noise and epsilon are set via
    ``params['leave_one_out']``.
    """
    def __init__(self, column: str, noise: float = 0.0, epsilon: float = 1e-6) -> None:
        super().__init__(column)
        self.noise = noise
        self.epsilon = epsilon
        self.global_mean_: float = 0.0
        self.sum_: Dict[Any, float] = {}
        self.count_: Dict[Any, int] = {}
        self.feature_names_ = [f"{self.column}__loo"]

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_LeaveOneOutEncoder":
        if y is None:
            raise EncoderFitError("Leave-one-out encoding requires 'y'.")
        y_series = _to_float_series(y)
        df = pd.DataFrame({"feature": series, "target": y_series})
        grouped = df.groupby("feature", dropna=False)["target"]
        stats = grouped.agg(["sum", "count"])
        self.sum_ = stats["sum"].to_dict()
        self.count_ = stats["count"].astype(int).to_dict()
        self.global_mean_ = float(y_series.mean())
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        encoded = []
        for value in series:
            total = self.sum_.get(value, self.global_mean_)
            count = self.count_.get(value, 0)
            if count <= 1:
                encoded_value = self.global_mean_
            else:
                encoded_value = (total - self.global_mean_) / (count - 1 + self.epsilon)
            if self.noise > 0:
                encoded_value += np.random.normal(0, self.noise)
            encoded.append(encoded_value)
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _CatBoostEncoder(_BaseColumnEncoder):
    """CatBoost-style ordered target statistics for a single column.

    Usage: ``strategy='catboost'`` with a target vector. Parameters ``prior``,
    ``noise`` and ``random_state`` can be passed via ``params['catboost']``.
    """
    def __init__(
        self,
        column: str,
        prior: float = 0.5,
        noise: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(column)
        self.prior = prior
        self.noise = noise
        self.random_state = random_state
        self.feature_names_ = [f"{self.column}__catboost"]
        self.mapping_: Dict[Any, float] = {}

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_CatBoostEncoder":
        if y is None:
            raise EncoderFitError("CatBoost encoding requires 'y'.")
        df = pd.DataFrame({"feature": series, "target": _to_float_series(y)})
        df = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        running_sum: Dict[Any, float] = {}
        running_count: Dict[Any, int] = {}
        encoded_values: Dict[Any, List[float]] = {}

        for value, target in zip(df["feature"], df["target"]):
            sum_before = running_sum.get(value, 0.0)
            count_before = running_count.get(value, 0)
            encoded = (sum_before + self.prior) / (count_before + 1)
            encoded_values.setdefault(value, []).append(encoded)
            running_sum[value] = sum_before + float(target)
            running_count[value] = count_before + 1

        self.mapping_ = {k: float(np.mean(v)) for k, v in encoded_values.items()}
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        default = np.mean(list(self.mapping_.values())) if self.mapping_ else self.prior
        encoded = series.map(self.mapping_).fillna(default)
        if self.noise > 0.0:
            encoded = encoded + np.random.normal(0, self.noise, size=len(encoded))
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _JamesSteinEncoder(_BaseColumnEncoder):
    """James-Stein shrinkage toward the global mean for regression targets.

    Usage: ``strategy='james_stein'`` (requires ``y``). Control the
    ``prior_weight`` through ``params['james_stein']``.
    """
    def __init__(self, column: str, prior_weight: float = 5.0) -> None:
        super().__init__(column)
        self.prior_weight = prior_weight
        self.global_mean_: float = 0.0
        self.mapping_: Dict[Any, float] = {}
        self.feature_names_ = [f"{self.column}__james_stein"]

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_JamesSteinEncoder":
        if y is None:
            raise EncoderFitError("James-Stein encoding requires 'y'.")
        y_series = _to_float_series(y)
        df = pd.DataFrame({"feature": series, "target": y_series})
        grouped = df.groupby("feature", dropna=False)["target"]
        stats = grouped.agg(["mean", "count"])
        self.global_mean_ = float(y_series.mean())
        self.mapping_ = {
            key: (self.global_mean_ * self.prior_weight + row["mean"] * row["count"])
            / (self.prior_weight + row["count"])
            for key, row in stats.iterrows()
        }
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        encoded = series.map(self.mapping_).fillna(self.global_mean_)
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _WoEEncoder(_BaseColumnEncoder):
    """Weight-of-Evidence encoder for binary classification.

    Usage: ``strategy='woe'`` with a binary target. Adjust Laplace smoothing
    through ``params['woe']['regularization']``.
    """
    def __init__(self, column: str, regularization: float = 1.0) -> None:
        super().__init__(column)
        self.regularization = regularization
        self.mapping_: Dict[Any, float] = {}
        self.feature_names_ = [f"{self.column}__woe"]

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_WoEEncoder":
        if y is None:
            raise EncoderFitError("Weight of Evidence encoding requires 'y'.")
        y_series = _to_float_series(y)
        classes = sorted(y_series.dropna().unique())
        if len(classes) != 2:
            raise EncoderFitError("Weight of Evidence encoding requires binary target.")

        df = pd.DataFrame({"feature": series, "target": y_series})
        grouped = df.groupby("feature", dropna=False)["target"]
        pos = grouped.sum()
        count = grouped.count()
        neg = count - pos

        total_pos = pos.sum()
        total_neg = neg.sum()
        for category in count.index:
            pos_val = pos.get(category, 0.0) + self.regularization
            neg_val = neg.get(category, 0.0) + self.regularization
            ratio = (pos_val / (total_pos + self.regularization)) / (
                neg_val / (total_neg + self.regularization)
            )
            self.mapping_[category] = float(np.log(ratio))
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        default = float(np.mean(list(self.mapping_.values()))) if self.mapping_ else 0.0
        encoded = series.map(self.mapping_).fillna(default)
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _CountFrequencyEncoder(_BaseColumnEncoder):
    """Maps categories to raw counts or normalized frequencies.

    Usage: choose ``strategy='count'`` or ``strategy='frequency'``. The
    ``normalize`` flag is set automatically based on the requested strategy,
    but can be overridden via ``params`` if needed.
    """
    def __init__(self, column: str, normalize: bool = False) -> None:
        super().__init__(column)
        self.normalize = normalize
        suffix = "freq" if normalize else "count"
        self.mapping_: Dict[Any, float] = {}
        self.feature_names_ = [f"{self.column}__{suffix}"]

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_CountFrequencyEncoder":
        counts = series.value_counts(normalize=self.normalize, dropna=False)
        self.mapping_ = counts.to_dict()
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        encoded = series.map(self.mapping_).fillna(0.0)
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _HashingEncoder(_BaseColumnEncoder):
    """Applies sklearn's FeatureHasher to very high-cardinality columns.

    Usage: set ``strategy='hashing'`` (no target required). Control the output
    dimension via ``params['hashing']['n_features']``.
    """
    def __init__(self, column: str, n_features: int = 32) -> None:
        super().__init__(column)
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type="string")
        self.feature_names_ = [f"{self.column}__hash_{i}" for i in range(self.n_features)]

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_HashingEncoder":
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        values = series.fillna("__nan__").astype(str)
        inputs = [{value: 1.0} for value in values]
        matrix = self.hasher.transform(inputs).toarray()
        return pd.DataFrame(matrix, columns=self.feature_names_, index=series.index)


class _BinaryEncoder(_BaseColumnEncoder):
    """Encodes categories as binary digits of their integer indices.

    Usage: ``strategy='binary'`` for mid-cardinality unsupervised columns.
    Automatically determines the number of required bits during ``fit``.
    """
    def __init__(self, column: str) -> None:
        super().__init__(column)
        self.category_mapping_: Dict[Any, int] = {}
        self.num_bits_: int = 0

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_BinaryEncoder":
        categories = pd.Series(series).fillna("__nan__").astype(str).unique()
        self.category_mapping_ = {cat: idx for idx, cat in enumerate(categories)}
        self.num_bits_ = int(math.ceil(math.log2(len(self.category_mapping_) + 1)))
        self.feature_names_ = [f"{self.column}__bin_{i}" for i in range(self.num_bits_)]
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        values = series.fillna("__nan__").astype(str)
        encoded = np.zeros((len(values), self.num_bits_))
        for row_idx, val in enumerate(values):
            idx = self.category_mapping_.get(val, -1)
            if idx < 0:
                continue
            binary_str = format(idx, f"0{self.num_bits_}b")
            encoded[row_idx] = [int(bit) for bit in binary_str]
        return pd.DataFrame(encoded, columns=self.feature_names_, index=series.index)


class _HelmertEncoder(_BaseColumnEncoder):
    """Produces Helmert (backward difference) contrasts for categorical levels.

    Usage: ``strategy='helmert'``; requires at least two observed categories
    and is typically used for ANOVA-style models.
    """
    def __init__(self, column: str) -> None:
        super().__init__(column)
        self.category_order_: List[Any] = []
        self.contrasts_: Dict[Any, List[float]] = {}

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_HelmertEncoder":
        categories = list(pd.Series(series).dropna().unique())
        if len(categories) <= 1:
            raise EncoderFitError("Helmert encoding requires at least two categories.")
        self.category_order_ = categories
        k = len(categories)
        self.feature_names_ = [f"{self.column}__helmert_{i}" for i in range(k - 1)]
        for idx, category in enumerate(categories):
            row = []
            for j in range(k - 1):
                if j < idx:
                    row.append(-1.0 / (idx + 1))
                elif j == idx:
                    row.append((k - idx - 1) / (idx + 1))
                else:
                    row.append(0.0)
            self.contrasts_[category] = row
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        default = [0.0] * len(self.feature_names_)
        matrix = [self.contrasts_.get(value, default) for value in series]
        return pd.DataFrame(matrix, columns=self.feature_names_, index=series.index)


class _RareCategoryEncoder(_BaseColumnEncoder):
    """Groups infrequent categories into a single indicator column.

    Usage: ``strategy='rare_grouping'`` and specify ``min_freq`` via
    ``params['rare_grouping']['min_freq']``. Often combined with a follow-up
    One-Hot or target encoder.
    """
    def __init__(self, column: str, min_freq: float = 0.01) -> None:
        super().__init__(column)
        self.min_freq = min_freq
        self.keep_categories_: List[Any] = []
        self.keep_index_: Dict[Any, int] = {}

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_RareCategoryEncoder":
        counts = series.value_counts(normalize=True, dropna=False)
        self.keep_categories_ = counts[counts >= self.min_freq].index.tolist()
        self.keep_index_ = {cat: idx for idx, cat in enumerate(self.keep_categories_)}
        self.feature_names_ = [
            f"{self.column}__top_{str(cat)}" for cat in self.keep_categories_
        ] + [f"{self.column}__rare"]
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        data = np.zeros((len(series), len(self.feature_names_)))
        for idx, value in enumerate(series):
            if value in self.keep_index_:
                data[idx, self.keep_index_[value]] = 1
            else:
                data[idx, -1] = 1
        return pd.DataFrame(data, columns=self.feature_names_, index=series.index)


class _EntityEmbeddingEncoder(_BaseColumnEncoder):
    """Creates dense embeddings via One-Hot + TruncatedSVD compression.

    Usage: ``strategy='embedding'`` (unsupervised). The ``embedding_dim`` and
    ``random_state`` options are configurable under ``params['embedding']``.
    """
    def __init__(self, column: str, embedding_dim: int = 4, random_state: Optional[int] = None) -> None:
        super().__init__(column)
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.reducer: Optional[TruncatedSVD] = TruncatedSVD(
            n_components=embedding_dim, random_state=random_state
        )

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_EntityEmbeddingEncoder":
        values = series.fillna("__nan__").astype(str)
        self.ohe.fit(values.to_frame())
        oh_matrix = self.ohe.transform(values.to_frame())
        if oh_matrix.shape[1] <= self.embedding_dim:
            self.reducer = None
            self.feature_names_ = [
                f"{self.column}__embed_{i}" for i in range(oh_matrix.shape[1])
            ]
        else:
            assert self.reducer is not None
            self.reducer.fit(oh_matrix)
            self.feature_names_ = [
                f"{self.column}__embed_{i}" for i in range(self.embedding_dim)
            ]
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        values = series.fillna("__nan__").astype(str)
        oh_matrix = self.ohe.transform(values.to_frame())
        if self.reducer is None:
            dense = oh_matrix.toarray()
        else:
            dense = self.reducer.transform(oh_matrix)
        return pd.DataFrame(dense, columns=self.feature_names_, index=series.index)


class _MultiHashEmbeddingEncoder(_BaseColumnEncoder):
    """Multi-hash embedding bag suitable for recommender identifiers.

    Usage: ``strategy='multi_hash'``. Tune ``n_hashes``, ``hash_dim`` and
    ``embedding_dim`` via ``params['multi_hash']``. Optionally uses the target
    to nudge embeddings during ``fit``.
    """
    def __init__(
        self,
        column: str,
        n_hashes: int = 2,
        hash_dim: int = 64,
        embedding_dim: int = 4,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(column)
        self.n_hashes = n_hashes
        self.hash_dim = hash_dim
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.feature_names_ = [
            f"{self.column}__mh_{h}_{d}"
            for h in range(n_hashes)
            for d in range(embedding_dim)
        ]
        self.embeddings_: np.ndarray = np.zeros((self.n_hashes, self.hash_dim, self.embedding_dim))

    def fit(self, series: pd.Series, y: Optional[pd.Series] = None) -> "_MultiHashEmbeddingEncoder":
        rng = np.random.default_rng(self.random_state)
        self.embeddings_ = rng.normal(scale=0.01, size=self.embeddings_.shape)
        if y is not None:
            targets = _to_float_series(y).to_numpy()
            values = series.fillna("__nan__").astype(str).to_numpy()
            for idx, value in enumerate(values):
                for h in range(self.n_hashes):
                    bucket = self._hash(value, h)
                    self.embeddings_[h, bucket] += targets[idx]
        return self

    def _hash(self, value: str, seed: int) -> int:
        digest = hashlib.blake2b(f"{value}:{seed}".encode("utf-8"), digest_size=4)
        return int.from_bytes(digest.digest(), "little") % self.hash_dim

    def transform(self, series: pd.Series) -> pd.DataFrame:
        values = series.fillna("__nan__").astype(str)
        data = np.zeros((len(series), len(self.feature_names_)))
        for row_idx, value in enumerate(values):
            cursor = 0
            for h in range(self.n_hashes):
                bucket = self._hash(value, h)
                embedding_vector = self.embeddings_[h, bucket]
                data[row_idx, cursor : cursor + self.embedding_dim] = embedding_vector
                cursor += self.embedding_dim
        return pd.DataFrame(data, columns=self.feature_names_, index=series.index)


ColumnStrategy = Union[str, Dict[str, str]]


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Flexible categorical encoder with automatic strategy discovery."""

    BUILTIN_ENCODERS: ClassVar[Dict[str, Type[_BaseColumnEncoder]]] = {
        "onehot": _OneHotColumnEncoder,
        "ordinal": _OrdinalColumnEncoder,
        "target": _TargetMeanEncoder,
        "catboost": _CatBoostEncoder,
        "james_stein": _JamesSteinEncoder,
        "woe": _WoEEncoder,
        "count": _CountFrequencyEncoder,
        "frequency": _CountFrequencyEncoder,
        "hashing": _HashingEncoder,
        "binary": _BinaryEncoder,
        "helmert": _HelmertEncoder,
        "leave_one_out": _LeaveOneOutEncoder,
        "rare_grouping": _RareCategoryEncoder,
        "embedding": _EntityEmbeddingEncoder,
        "multi_hash": _MultiHashEmbeddingEncoder,
    }

    def __init__(
        self,
        strategy: ColumnStrategy = "auto",
        params: Optional[Dict[str, Any]] = None,
        auto_survey: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.strategy = strategy
        self.params = params or {}
        self.auto_survey = auto_survey
        self.random_state = random_state

        self.column_stats_: Dict[str, ColumnStats] = {}
        self.column_encoders_: Dict[str, _BaseColumnEncoder] = {}
        self.column_strategies_: Dict[str, str] = {}
        self.fitted_feature_names_: List[str] = []
        self.decision_log_: Dict[str, str] = {}

    @classmethod
    def register_encoder(cls, name: str, encoder_class: Type[_BaseColumnEncoder]) -> None:
        """Register custom encoder classes for later use."""
        if not issubclass(encoder_class, _BaseColumnEncoder):
            raise EncoderConfigurationError(
                f"Encoder '{name}' must inherit from _BaseColumnEncoder."
            )
        cls.BUILTIN_ENCODERS[name] = encoder_class

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, Sequence[Any]]] = None,
    ) -> "SmartCategoricalEncoder":
        """Fit encoders for each categorical column."""
        X_df = self._validate_input(X)
        y_series = _to_float_series(y) if y is not None else None
        categorical_columns = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if not categorical_columns:
            raise EncoderFitError("No categorical columns detected.")

        self.column_stats_.clear()
        self.column_encoders_.clear()
        self.column_strategies_.clear()
        self.decision_log_.clear()

        for column in categorical_columns:
            series = X_df[column]
            stats = self._survey_column(series)
            self.column_stats_[column] = stats
            strategy = self._resolve_strategy(column, stats, y_series)
            params = self._build_params_for_strategy(strategy)
            encoder = self._instantiate_encoder(column, strategy, params)
            encoder.fit(series, y_series)
            self.column_encoders_[column] = encoder
            self.column_strategies_[column] = strategy

        self._collect_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X using the learned encoders."""
        check_is_fitted(self, "column_encoders_")
        X_df = self._validate_input(X, fit_stage=False)
        transformed_frames: List[pd.DataFrame] = []
        for column, encoder in self.column_encoders_.items():
            if column not in X_df.columns:
                raise EncoderFitError(f"Column '{column}' missing at transform time.")
            transformed_frames.append(encoder.transform(X_df[column]))
        return pd.concat(transformed_frames, axis=1)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, Sequence[Any]]] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Persist the fitted encoder to path using joblib."""
        check_is_fitted(self, "column_encoders_")
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SmartCategoricalEncoder":
        loaded = joblib.load(path)
        if not isinstance(loaded, SmartCategoricalEncoder):
            raise EncoderConfigurationError("Loaded object is not a SmartCategoricalEncoder.")
        return loaded

    def get_feature_names(self) -> List[str]:
        check_is_fitted(self, "fitted_feature_names_")
        return list(self.fitted_feature_names_)

    def _validate_input(self, X: pd.DataFrame, fit_stage: bool = True) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise EncoderFitError("SmartCategoricalEncoder expects a pandas DataFrame.")
        if fit_stage and X.empty:
            raise EncoderFitError("Cannot fit encoder on empty DataFrame.")
        return X

    def _survey_column(self, series: pd.Series) -> ColumnStats:
        non_missing = series.dropna()
        cardinality = non_missing.nunique()
        total = len(series)
        unique_ratio = cardinality / max(len(non_missing), 1)
        counts = series.value_counts(dropna=False)
        rare_threshold = self.params.get("general", {}).get("rare_threshold", 0.02)
        rare_mask = (counts / max(total, 1)) < rare_threshold
        rare_share = float(counts[rare_mask].sum() / max(total, 1)) if total else 0.0
        missing_rate = float(series.isna().mean())
        most_common_freq = float(counts.max() / total) if total else 0.0
        return ColumnStats(
            cardinality=int(cardinality),
            unique_ratio=float(unique_ratio),
            rare_share=float(rare_share),
            missing_rate=float(missing_rate),
            most_common_freq=most_common_freq,
        )

    def _resolve_strategy(self, column: str, stats: ColumnStats, y: Optional[pd.Series]) -> str:
        if isinstance(self.strategy, dict):
            if column not in self.strategy:
                raise EncoderConfigurationError(f"No strategy provided for column '{column}'.")
            return self.strategy[column]
        if isinstance(self.strategy, str) and self.strategy != "auto":
            return self.strategy

        if not self.auto_survey:
            raise EncoderConfigurationError(
                "Auto strategy selection disabled but strategy='auto' requested."
            )

        thresholds = self.params.get(
            "general",
            {
                "max_onehot_cardinality": 12,
                "max_target_cardinality": 60,
                "max_hash_cardinality": 200,
                "rare_threshold": 0.02,
            },
        )
        has_target = y is not None

        if stats.cardinality <= thresholds.get("max_onehot_cardinality", 12):
            decision = "onehot"
        elif stats.cardinality <= thresholds.get("max_target_cardinality", 60):
            decision = "catboost" if has_target else "ordinal"
        elif stats.cardinality <= thresholds.get("max_hash_cardinality", 200):
            decision = "target" if has_target else "hashing"
        else:
            decision = "multi_hash" if has_target else "embedding"

        if stats.missing_rate > 0.2 and has_target:
            decision = "catboost"
        if stats.unique_ratio > 0.9 and has_target:
            decision = "multi_hash"
        if stats.rare_share > thresholds.get("rare_threshold", 0.02):
            decision = "rare_grouping"

        self.decision_log_[column] = decision
        return decision

    def _build_params_for_strategy(self, strategy: str) -> Dict[str, Any]:
        key = strategy if strategy in self.params else strategy.lower()
        params = self.params.get(key, {}).copy()
        if strategy in {"count", "frequency"}:
            params["normalize"] = strategy == "frequency"
        if "random_state" not in params and self.random_state is not None:
            params["random_state"] = self.random_state
        return params

    def _instantiate_encoder(self, column: str, strategy: str, params: Dict[str, Any]) -> _BaseColumnEncoder:
        key = strategy if strategy in self.BUILTIN_ENCODERS else strategy.lower()
        encoder_cls = self.BUILTIN_ENCODERS.get(key)
        if encoder_cls is None:
            raise EncoderConfigurationError(f"Unknown encoder strategy '{strategy}'.")
        try:
            encoder = encoder_cls(column=column, **params)
        except TypeError as exc:
            raise EncoderConfigurationError(
                f"Failed to initialize encoder '{strategy}' for column '{column}': {exc}"
            ) from exc
        return encoder

    def _collect_feature_names(self) -> None:
        names: List[str] = []
        for encoder in self.column_encoders_.values():
            names.extend(encoder.get_feature_names())
        self.fitted_feature_names_ = names


__all__ = ["SmartCategoricalEncoder"]
