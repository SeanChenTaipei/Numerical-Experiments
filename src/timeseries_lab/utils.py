from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def hash_pandas_frame(df: pd.DataFrame) -> str:
    """Return a hex hash for a dataframe that is stable across sessions."""

    hashed = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def stable_hash(obj: object) -> str:
    """Hash any JSON-serializable object deterministically."""

    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8", "ignore")
    return hashlib.md5(payload).hexdigest()


def build_cache_key(*parts: str) -> str:
    """Concatenate cache key parts safely."""

    return "|".join(parts)


def maybe_sample_df(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Return df sampled to max_rows if necessary."""

    if len(df) <= max_rows:
        df.attrs["sampled"] = False
        return df.copy()
    sample = df.sample(n=max_rows, random_state=random_state)
    sample.attrs["sampled"] = True
    return sample.sort_index()


def ensure_datetime(series: pd.Series) -> pd.Series:
    """Coerce a series to datetime64."""

    return pd.to_datetime(series, errors="coerce").sort_values()


def listify(value: str | Sequence[str]) -> List[str]:
    """Ensure the input is a list of strings."""

    if isinstance(value, str):
        return [value]
    return list(value)


def top_k_jaccard(selections: Sequence[Sequence[str]], k: int) -> float:
    """Compute mean pairwise Jaccard for the provided selections."""

    if not selections:
        return 0.0
    clipped = [set(seq[:k]) for seq in selections]
    if len(clipped) == 1:
        return 1.0
    scores: List[float] = []
    for i in range(len(clipped)):
        for j in range(i + 1, len(clipped)):
            inter = len(clipped[i] & clipped[j])
            union = len(clipped[i] | clipped[j])
            scores.append(inter / union if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


@dataclass(slots=True)
class GroupCoverageEntry:
    group: str
    split: str
    share: float


def compute_group_coverage(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    """Return coverage table for group distribution across splits."""

    total = (
        pd.concat(
            [
                df_train.assign(split="train"),
                df_val.assign(split="val"),
                df_test.assign(split="test"),
            ],
            axis=0,
        )
        .groupby([group_col, "split"])
        .size()
        .reset_index(name="count")
    )
    total["share"] = (
        total.groupby("split")["count"].transform(lambda x: x / x.sum()).round(4)
    )
    return total.sort_values(["split", "share"], ascending=[True, False])


def ensure_directory(path: str | Iterable[str]) -> None:
    """Create a directory (or directories) if missing."""

    if isinstance(path, str):
        Path(path).mkdir(parents=True, exist_ok=True)
        return
    for part in path:
        Path(part).mkdir(parents=True, exist_ok=True)
