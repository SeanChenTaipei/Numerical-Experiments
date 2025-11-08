"""Data loading utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .config import DatasetConfig


def load_dataset(
    cfg: DatasetConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset specified by the config and return train/test splits."""
    if cfg.name == "iris":
        raw = datasets.load_iris(as_frame=True)
        features = raw.data
        target = raw.target
    else:
        raise ValueError(
            f"Unsupported dataset '{cfg.name}'. Try 'iris' or extend `load_dataset`."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=target,
    )

    # Ensure contiguous arrays for downstream sklearn usage.
    return (
        pd.DataFrame(np.asarray(X_train), columns=features.columns),
        pd.DataFrame(np.asarray(X_test), columns=features.columns),
        pd.Series(np.asarray(y_train), name=target.name),
        pd.Series(np.asarray(y_test), name=target.name),
    )
