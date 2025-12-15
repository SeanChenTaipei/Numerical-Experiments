from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =========================
# Utilities: validation & preprocessing
# =========================

def _validate_columns(df: pd.DataFrame, y_col: str, feature_cols: List[str]) -> None:
    missing = [c for c in [y_col] + feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")


def _to_1d_float(x: Union[pd.Series, np.ndarray]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D after reshape.")
    return arr


def _standardize_1d(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        # Constant feature: keep as zeros to avoid NaNs; HSIC should become ~0 after centering anyway.
        return np.zeros_like(x)
    return (x - mu) / sd


def _impute_1d(x: np.ndarray, strategy: str = "median") -> np.ndarray:
    # Minimal imputation to avoid kernel NaNs.
    if not np.any(np.isnan(x)):
        return x
    if strategy == "median":
        fill = float(np.nanmedian(x))
    elif strategy == "mean":
        fill = float(np.nanmean(x))
    elif strategy == "zero":
        fill = 0.0
    else:
        raise ValueError(f"Unsupported impute strategy: {strategy}")
    out = x.copy()
    out[np.isnan(out)] = fill
    return out


# =========================
# HSIC core: kernels + Gram matrices + HSIC estimator
# =========================

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: Kc = H K H."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n), dtype=float) / n
    return H @ K @ H


def _pairwise_sq_dists_1d(x: np.ndarray) -> np.ndarray:
    """Pairwise squared distances for 1D vector x (n,)."""
    x = x.reshape(-1, 1)
    x2 = x ** 2
    return x2 + x2.T - 2.0 * (x @ x.T)


def _median_heuristic_sigma_1d(x: np.ndarray, eps: float = 1e-12) -> float:
    """Median heuristic for RBF sigma using 1D vector x."""
    D2 = _pairwise_sq_dists_1d(x)
    tri = D2[np.triu_indices(D2.shape[0], k=1)]
    med = float(np.median(tri)) if tri.size > 0 else 1.0
    return float(np.sqrt(max(med, eps)))


def _gram_1d(
    x: np.ndarray,
    kernel: str,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Compute 1D Gram matrix for x.
    Returns (K, used_sigma).
    """
    x = _to_1d_float(x)

    if kernel == "linear":
        return np.outer(x, x), None

    if kernel == "rbf":
        if sigma is None:
            sigma = _median_heuristic_sigma_1d(x)
        D2 = _pairwise_sq_dists_1d(x)
        K = np.exp(-D2 / (2.0 * sigma * sigma))
        return K, sigma

    raise ValueError(f"Unsupported kernel: {kernel}")


@dataclass
class HSICConfig:
    """
    Configuration for HSIC computation.

    scale:
      - If True, standardize each variable before kernel computation.
      - Recommended for RBF kernels.
    impute:
      - 'median' / 'mean' / 'zero' for NaN handling.
    """
    x_kernel: str = "rbf"
    y_kernel: str = "rbf"
    x_sigma: Optional[float] = None
    y_sigma: Optional[float] = None
    scale: bool = True
    impute: str = "median"


@dataclass
class HSICComputer:
    """
    HSIC computer using empirical centered-Gram estimator:

      HSIC_n(X,Y) = tr(Kc Lc) / (n-1)^2

    Notes:
      - This is a practical estimator (commonly used).
      - For constant vectors, standardization returns zeros -> HSIC ~ 0.
    """
    cfg: HSICConfig

    def hsic(self, x: np.ndarray, y: np.ndarray) -> float:
        x = _to_1d_float(x)
        y = _to_1d_float(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length.")

        x = _impute_1d(x, self.cfg.impute)
        y = _impute_1d(y, self.cfg.impute)

        if self.cfg.scale:
            x = _standardize_1d(x)
            y = _standardize_1d(y)

        K, _ = _gram_1d(x, kernel=self.cfg.x_kernel, sigma=self.cfg.x_sigma)
        L, _ = _gram_1d(y, kernel=self.cfg.y_kernel, sigma=self.cfg.y_sigma)
        Kc = _center_gram(K)
        Lc = _center_gram(L)

        n = x.shape[0]
        denom = (n - 1) ** 2 if n > 1 else 1.0
        return float(np.trace(Kc @ Lc) / denom)


# =========================
# Selector 1: HSIC relevance ranking (fast, recommended baseline)
# =========================

@dataclass
class HSICRelevanceSelector:
    """
    Select features by HSIC relevance score: a_j = HSIC(X_j, Y).

    Typical use:
      - Stage 1.5 filter: keep Top-N features before RFE / LOFO / SHAP.
    """
    y_col: str
    feature_cols: List[str]

    hsic_cfg: HSICConfig = field(default_factory=HSICConfig)

    top_k: Optional[int] = 100
    min_hsic: float = 0.0
    drop_constant_features: bool = True

    # fitted
    relevance_: Optional[pd.Series] = None
    selected_features_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "HSICRelevanceSelector":
        _validate_columns(df, self.y_col, self.feature_cols)

        y = _to_1d_float(df[self.y_col].to_numpy())
        comp = HSICComputer(self.hsic_cfg)

        scores: Dict[str, float] = {}
        for col in self.feature_cols:
            x = _to_1d_float(df[col].to_numpy())

            if self.drop_constant_features:
                x_imp = _impute_1d(x, self.hsic_cfg.impute)
                if np.nanstd(x_imp) < 1e-12:
                    scores[col] = 0.0
                    continue

            scores[col] = comp.hsic(x, y)

        s = pd.Series(scores, dtype=float).sort_values(ascending=False)

        if self.min_hsic > 0:
            s = s[s >= float(self.min_hsic)]

        if self.top_k is not None:
            s = s.head(int(self.top_k))

        self.relevance_ = s
        self.selected_features_ = s.index.tolist()
        return self

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        # Allow small overrides at call-time
        old_top_k, old_min = self.top_k, self.min_hsic
        self.top_k = kwargs.get("top_k", self.top_k)
        self.min_hsic = float(kwargs.get("min_hsic", self.min_hsic))
        try:
            self.fit(df)
            return {
                "selected_features": self.selected_features_,
                "relevance": self.relevance_,
            }
        finally:
            self.top_k, self.min_hsic = old_top_k, old_min


# =========================
# Selector 2: HSIC-Lasso (global redundancy control, heavier)
# =========================

def _spectral_norm_power(A: np.ndarray, iters: int = 60, eps: float = 1e-12) -> float:
    """Approximate spectral norm for symmetric matrix by power iteration."""
    n = A.shape[0]
    if n == 0:
        return 0.0
    v = np.random.randn(n)
    v /= (np.linalg.norm(v) + eps)
    for _ in range(iters):
        v = A @ v
        v /= (np.linalg.norm(v) + eps)
    # Rayleigh quotient
    return float(np.sqrt(max(v @ (A @ v), 0.0)))


@dataclass
class HSICLassoSelector:
    """
    HSIC-Lasso-like selector (prototype, but production-usable for moderate p).

    Objective (min form):
      min_{beta >= 0}  beta^T R beta - a^T beta + lam * ||beta||_1

    Where:
      a_j = HSIC(X_j, Y)         (relevance)
      R_jk = HSIC(X_j, X_k)      (redundancy)

    Algorithm:
      - Projected proximal gradient
      - Stopping: objective convergence OR support-set stability OR max_iter
      - Lambda path: scan lam_list to hit target_nnz or pick closest.

    Complexity:
      - Building R is O(p^2 * n^2) in naive kernel HSIC; for large p you should
        run relevance selector first to reduce candidate set.
    """
    y_col: str
    feature_cols: List[str]

    hsic_cfg_xy: HSICConfig = field(default_factory=lambda: HSICConfig(x_kernel="rbf", y_kernel="rbf"))
    hsic_cfg_xx: Optional[HSICConfig] = None  # if None, reuse x_kernel as y_kernel for X-X

    # Lambda path controls sparsity
    lam_list: Optional[List[float]] = None
    target_nnz: Optional[Tuple[int, int]] = (30, 80)

    # Optimizer params
    max_iter: int = 800
    tol: float = 1e-6
    support_threshold: float = 1e-8
    support_patience: int = 20
    step_size: Optional[float] = None  # if None auto from Lipschitz

    verbose: bool = False

    # fitted
    relevance_: Optional[np.ndarray] = None
    redundancy_: Optional[np.ndarray] = None
    coef_: Optional[np.ndarray] = None
    chosen_lam_: Optional[float] = None
    selected_features_: Optional[List[str]] = None

    def _make_xx_cfg(self) -> HSICConfig:
        if self.hsic_cfg_xx is not None:
            return self.hsic_cfg_xx
        # default: same kernel family for X-X
        return HSICConfig(
            x_kernel=self.hsic_cfg_xy.x_kernel,
            y_kernel=self.hsic_cfg_xy.x_kernel,
            x_sigma=self.hsic_cfg_xy.x_sigma,
            y_sigma=self.hsic_cfg_xy.x_sigma,
            scale=self.hsic_cfg_xy.scale,
            impute=self.hsic_cfg_xy.impute,
        )

    def _compute_a_R(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = X.shape[1]
        comp_xy = HSICComputer(self.hsic_cfg_xy)
        comp_xx = HSICComputer(self._make_xx_cfg())

        a = np.zeros(p, dtype=float)
        for j in range(p):
            a[j] = comp_xy.hsic(X[:, j], y)

        R = np.zeros((p, p), dtype=float)
        for j in range(p):
            # diagonal is HSIC(Xj, Xj) > 0 (can be used as scaling)
            R[j, j] = comp_xx.hsic(X[:, j], X[:, j])
            for k in range(j + 1, p):
                val = comp_xx.hsic(X[:, j], X[:, k])
                R[j, k] = val
                R[k, j] = val
        return a, R

    def _solve_single_lambda(self, a: np.ndarray, R: np.ndarray, lam: float) -> np.ndarray:
        p = a.shape[0]
        beta = np.zeros(p, dtype=float)

        if self.step_size is None:
            # Lipschitz of grad for beta^T R beta is 2||R||2
            spec = max(_spectral_norm_power(R), 1e-12)
            step = 1.0 / (2.0 * spec)
        else:
            step = float(self.step_size)

        prev_obj = np.inf
        prev_support: Optional[Tuple[int, ...]] = None
        stable = 0

        for t in range(self.max_iter):
            grad = 2.0 * (R @ beta) - a
            z = beta - step * grad

            # proximal for L1 + projection beta>=0
            beta_new = np.maximum(0.0, z - step * lam)

            obj = float(beta_new @ (R @ beta_new) - a @ beta_new + lam * np.sum(beta_new))
            support = tuple(np.where(beta_new > self.support_threshold)[0].tolist())

            if support == prev_support:
                stable += 1
            else:
                stable = 0
                prev_support = support

            if self.verbose and (t % 50 == 0 or t == self.max_iter - 1):
                nnz = int((beta_new > self.support_threshold).sum())
                print(f"[lam={lam:.3e} iter={t}] obj={obj:.6e} nnz={nnz} stable={stable}")

            # stopping conditions
            if abs(prev_obj - obj) < self.tol:
                beta = beta_new
                break
            if stable >= self.support_patience:
                beta = beta_new
                break

            beta = beta_new
            prev_obj = obj

        return beta

    def _auto_lam_list(self, a: np.ndarray) -> List[float]:
        # Heuristic: scale lambda by max relevance
        amax = float(np.max(a)) if a.size else 1.0
        # from small (dense) to large (sparse), but we will scan large->small or vice versa by choice
        lams = np.logspace(np.log10(amax * 1e-3 + 1e-12), np.log10(amax * 1e1 + 1e-12), 14)
        # Prefer scanning from large to small to find target nnz sooner (sparser first)
        return list(sorted(lams, reverse=True))

    def fit(self, df: pd.DataFrame) -> "HSICLassoSelector":
        _validate_columns(df, self.y_col, self.feature_cols)

        X = df[self.feature_cols].to_numpy(dtype=float)
        y = _to_1d_float(df[self.y_col].to_numpy())

        a, R = self._compute_a_R(X, y)
        self.relevance_ = a
        self.redundancy_ = R

        lam_list = self.lam_list or self._auto_lam_list(a)

        best_beta: Optional[np.ndarray] = None
        best_lam: Optional[float] = None

        if self.target_nnz is None:
            lo, hi = 1, len(self.feature_cols)
        else:
            lo, hi = int(self.target_nnz[0]), int(self.target_nnz[1])
            lo = max(1, lo)
            hi = min(len(self.feature_cols), hi)

        target_mid = (lo + hi) // 2

        def nnz(beta: np.ndarray) -> int:
            return int((beta > self.support_threshold).sum())

        best_dist = float("inf")

        for lam in lam_list:
            beta = self._solve_single_lambda(a, R, float(lam))
            k = nnz(beta)

            # Primary: hit target nnz range
            if lo <= k <= hi:
                best_beta, best_lam = beta, float(lam)
                break

            # Fallback: keep the closest to target_mid
            dist = abs(k - target_mid)
            if dist < best_dist:
                best_dist = dist
                best_beta, best_lam = beta, float(lam)

        if best_beta is None:
            # should not happen, but keep safe
            best_beta = np.zeros_like(a)
            best_lam = float(lam_list[-1])

        self.coef_ = best_beta
        self.chosen_lam_ = best_lam

        idx = np.where(self.coef_ > self.support_threshold)[0].tolist()
        self.selected_features_ = [self.feature_cols[i] for i in idx]
        return self

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        # Allow limited overrides without polluting state permanently
        old_lam_list = self.lam_list
        old_target_nnz = self.target_nnz
        old_verbose = self.verbose

        self.lam_list = kwargs.get("lam_list", self.lam_list)
        self.target_nnz = kwargs.get("target_nnz", self.target_nnz)
        self.verbose = bool(kwargs.get("verbose", self.verbose))

        try:
            self.fit(df)
            return {
                "selected_features": self.selected_features_,
                "coef": self.coef_,
                "relevance": self.relevance_,
                "chosen_lam": self.chosen_lam_,
            }
        finally:
            self.lam_list = old_lam_list
            self.target_nnz = old_target_nnz
            self.verbose = old_verbose