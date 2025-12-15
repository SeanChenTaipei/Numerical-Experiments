from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: Kc = H K H."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    """Compute pairwise squared distances for a 1D vector x (n,)."""
    x = x.reshape(-1, 1)
    # (xi - xj)^2 = xi^2 + xj^2 - 2 xi xj
    x2 = x ** 2
    return x2 + x2.T - 2.0 * (x @ x.T)


def _median_heuristic_sigma(x: np.ndarray, eps: float = 1e-12) -> float:
    """Median heuristic for RBF bandwidth sigma using 1D vector x."""
    D2 = _pairwise_sq_dists(x)
    # take upper triangle without diagonal
    triu = D2[np.triu_indices(D2.shape[0], k=1)]
    med = np.median(triu)
    return float(np.sqrt(max(med, eps)))


def _gram_matrix_1d(
    x: np.ndarray,
    kernel: str,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Compute Gram matrix for a 1D feature vector."""
    x = np.asarray(x).astype(float).reshape(-1)
    if kernel == "linear":
        # For 1D, linear kernel is x x^T
        return np.outer(x, x)
    if kernel == "rbf":
        if sigma is None:
            sigma = _median_heuristic_sigma(x)
        D2 = _pairwise_sq_dists(x)
        return np.exp(-D2 / (2.0 * (sigma ** 2)))
    raise ValueError(f"Unsupported kernel: {kernel}")


def hsic_unbiased_empirical(
    x: np.ndarray,
    y: np.ndarray,
    x_kernel: str = "rbf",
    y_kernel: str = "rbf",
    x_sigma: Optional[float] = None,
    y_sigma: Optional[float] = None,
) -> float:
    """
    Empirical HSIC via centered Gram matrices:
      HSIC = tr(Kc Lc) / (n-1)^2
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    K = _center_gram(_gram_matrix_1d(x, kernel=x_kernel, sigma=x_sigma))
    L = _center_gram(_gram_matrix_1d(y, kernel=y_kernel, sigma=y_sigma))
    n = x.shape[0]
    denom = (n - 1) ** 2 if n > 1 else 1.0
    return float(np.trace(K @ L) / denom)


def _power_iteration_spectral_norm(A: np.ndarray, iters: int = 50, eps: float = 1e-12) -> float:
    """Approximate spectral norm ||A||2 for symmetric PSD-ish matrices."""
    n = A.shape[0]
    v = np.random.randn(n)
    v /= (np.linalg.norm(v) + eps)
    for _ in range(iters):
        v = A @ v
        v /= (np.linalg.norm(v) + eps)
    return float(np.sqrt((v @ (A @ v))))


@dataclass
class HSICLassoSelector:
    """
    HSIC-Lasso-like feature selector (prototype).

    Objective (minimization form):
        min_{beta >= 0}  beta^T R beta - a^T beta + lam * ||beta||_1

    where:
        a_j = HSIC(X_j, Y)  (relevance)
        R_jk = HSIC(X_j, X_k) (redundancy)

    Notes:
    - This is a practical prototype (projected proximal gradient).
    - Suitable as a Stage-1 filter for downstream model-based selection.
    """

    y_col: str
    feature_cols: List[str]

    # HSIC kernels
    x_kernel: str = "rbf"
    y_kernel: str = "rbf"
    x_sigma: Optional[float] = None
    y_sigma: Optional[float] = None

    # Optimization / selection
    lam: float = 1e-3
    max_iter: int = 500
    tol: float = 1e-6
    step_size: Optional[float] = None  # if None, auto from spectral norm
    top_k: Optional[int] = None
    coef_threshold: float = 1e-8
    verbose: bool = False

    # Fitted attributes
    coef_: Optional[np.ndarray] = None
    relevance_: Optional[np.ndarray] = None
    redundancy_: Optional[np.ndarray] = None
    selected_features_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "HSICLassoSelector":
        """Fit selector on df and compute selected features."""
        X = df[self.feature_cols].to_numpy(dtype=float)
        y = df[self.y_col].to_numpy(dtype=float).reshape(-1)
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("df[y_col] length mismatch with feature rows.")

        # 1) Relevance vector a_j = HSIC(X_j, y)
        a = np.zeros(p, dtype=float)
        for j in range(p):
            a[j] = hsic_unbiased_empirical(
                X[:, j],
                y,
                x_kernel=self.x_kernel,
                y_kernel=self.y_kernel,
                x_sigma=self.x_sigma,
                y_sigma=self.y_sigma,
            )

        # 2) Redundancy matrix R_jk = HSIC(X_j, X_k)
        R = np.zeros((p, p), dtype=float)
        for j in range(p):
            R[j, j] = hsic_unbiased_empirical(
                X[:, j],
                X[:, j],
                x_kernel=self.x_kernel,
                y_kernel=self.x_kernel,  # same kernel family for features
                x_sigma=self.x_sigma,
                y_sigma=self.x_sigma,
            )
            for k in range(j + 1, p):
                val = hsic_unbiased_empirical(
                    X[:, j],
                    X[:, k],
                    x_kernel=self.x_kernel,
                    y_kernel=self.x_kernel,
                    x_sigma=self.x_sigma,
                    y_sigma=self.x_sigma,
                )
                R[j, k] = val
                R[k, j] = val

        # 3) Solve: min beta^T R beta - a^T beta + lam ||beta||_1, beta>=0
        beta = np.zeros(p, dtype=float)

        # Lipschitz constant of grad( beta^T R beta - a^T beta ) is 2||R||2
        if self.step_size is None:
            spec = _power_iteration_spectral_norm(R, iters=50)
            L = max(2.0 * spec, 1e-12)
            step = 1.0 / L
        else:
            step = float(self.step_size)

        prev_obj = np.inf
        for t in range(self.max_iter):
            grad = 2.0 * (R @ beta) - a  # gradient of smooth part
            # Prox step for L1 with non-negativity:
            # z = beta - step*grad
            # beta <- max(0, z - step*lam)
            z = beta - step * grad
            beta_new = np.maximum(0.0, z - step * self.lam)

            # Objective
            obj = float(beta_new @ (R @ beta_new) - a @ beta_new + self.lam * np.sum(beta_new))

            if self.verbose and (t % 50 == 0 or t == self.max_iter - 1):
                print(f"[iter={t}] obj={obj:.6e} |beta|_0={(beta_new > self.coef_threshold).sum()}")

            if abs(prev_obj - obj) < self.tol:
                beta = beta_new
                break

            beta = beta_new
            prev_obj = obj

        # 4) Select features
        if self.top_k is not None:
            idx = np.argsort(-beta)[: int(self.top_k)]
            idx = idx[beta[idx] > self.coef_threshold]
        else:
            idx = np.where(beta > self.coef_threshold)[0]

        self.coef_ = beta
        self.relevance_ = a
        self.redundancy_ = R
        self.selected_features_ = [self.feature_cols[i] for i in idx.tolist()]
        return self

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Run selection. kwargs can override a subset of init params at call-time.

        Returns:
            dict with keys:
              - selected_features
              - coef
              - relevance
        """
        # Lightweight overrides (common use)
        top_k = kwargs.get("top_k", self.top_k)
        lam = kwargs.get("lam", self.lam)

        # Temporarily override for this call
        old_top_k, old_lam = self.top_k, self.lam
        self.top_k, self.lam = top_k, lam
        try:
            self.fit(df)
            return {
                "selected_features": self.selected_features_,
                "coef": self.coef_,
                "relevance": self.relevance_,
            }
        finally:
            # restore
            self.top_k, self.lam = old_top_k, old_lam