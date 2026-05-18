"""Artifact serialization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
