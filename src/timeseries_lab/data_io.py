from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np
import pandas as pd

from .settings import PROJECT_ROOT
from .utils import hash_pandas_frame


class DataLoader(Protocol):
    """Protocol for all loaders."""

    def load(self) -> pd.DataFrame: ...


@dataclass(slots=True)
class SyntheticDatasetBundle:
    """Container for synthetic data + factors."""

    df_data: pd.DataFrame
    df_factors: pd.DataFrame
    metadata: Dict[str, Any]


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_feather(path: Path) -> pd.DataFrame:
    return pd.read_feather(path)


class DataLoaderFactory:
    """Factory that dispatches file readers and synthetic generators."""

    registry: Dict[str, Callable[[Path], pd.DataFrame]] = {
        ".csv": _load_csv,
        ".feather": _load_feather,
        ".ft": _load_feather,
    }

    @classmethod
    def from_path(cls, path: str | Path) -> pd.DataFrame:
        """Load a dataframe from csv/feather."""

        file_path = Path(path)
        reader = cls.registry.get(file_path.suffix.lower())
        if not reader:
            raise ValueError(f"Unsupported extension: {file_path.suffix}")
        return reader(file_path)

    @classmethod
    def from_bytes(cls, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """Load uploaded file based on filename extension."""

        extension = Path(filename).suffix.lower()
        buffer = io.BytesIO(file_bytes)
        if extension == ".csv":
            return pd.read_csv(buffer)
        if extension in {".feather", ".ft"}:
            return pd.read_feather(buffer)
        raise ValueError(f"Unsupported upload extension {extension}")

    @classmethod
    def fallback_synthetic(cls, *, rows: int = 3000, groups: int = 12) -> SyntheticDatasetBundle:
        """Generate a synthetic dataset with time + group structure."""

        return generate_synthetic_dataset(rows=rows, groups=groups)


@lru_cache(maxsize=8)
def generate_synthetic_dataset(*, rows: int = 5000, groups: int = 15) -> SyntheticDatasetBundle:
    """Create reproducible synthetic dataset for demos/tests."""

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2020-01-01", periods=rows, freq="H")
    machine_ids = rng.integers(0, groups, size=rows)
    batch_ids = rng.integers(0, groups // 3 + 1, size=rows)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": machine_ids.astype(str),
            "batch_id": batch_ids.astype(str),
            "runtime_hours": rng.gamma(shape=2.0, scale=3.0, size=rows) + machine_ids * 0.05,
            "temperature": 20 + rng.normal(0, 3, size=rows) + batch_ids * 0.1,
            "pressure": 5 + rng.normal(0, 1, size=rows) + machine_ids * 0.02,
            "vibration": rng.normal(0, 1, size=rows),
            "operator": rng.choice(list("ABCDE"), size=rows),
        }
    )
    df["target_main"] = (
        0.3 * df["runtime_hours"]
        + 0.8 * df["temperature"]
        - 0.5 * df["pressure"]
        + 0.2 * np.sin(np.linspace(0, 12 * np.pi, rows))
        + rng.normal(0, 0.5, size=rows)
    )
    df["target_secondary"] = df["target_main"] * 0.8 + rng.normal(0, 0.3, size=rows)

    df_factors = pd.DataFrame({"FactorName": ["target_main", "target_secondary"]})
    metadata = {
        "hash": hash_pandas_frame(df),
        "rows": rows,
        "groups": groups,
        "source": "synthetic",
    }
    return SyntheticDatasetBundle(df_data=df, df_factors=df_factors, metadata=metadata)


def persist_bundle(bundle: SyntheticDatasetBundle, data_path: Path | None = None) -> None:
    """Persist the synthetic bundle to disk for reproducibility."""

    target_data = data_path or PROJECT_ROOT / "data" / "synthetic_data.feather"
    target_factors = target_data.with_name("synthetic_factors.feather")
    target_data.parent.mkdir(parents=True, exist_ok=True)
    bundle.df_data.reset_index(drop=True).to_feather(target_data)
    bundle.df_factors.to_feather(target_factors)
