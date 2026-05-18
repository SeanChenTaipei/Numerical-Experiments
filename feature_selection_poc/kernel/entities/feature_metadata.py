"""Feature metadata helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureMetadata:
    feature_name: str
    data_source: str = "unknown"
    station: str = "unknown"
    process_step: str = "unknown"
    tool: str | None = None
    recipe: str | None = None


REQUIRED_METADATA_COLUMNS = ["feature_name", "data_source", "station", "process_step", "tool", "recipe"]


def ensure_feature_metadata(features: list[str], metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return metadata with one row per feature and all graph-oriented columns."""
    if metadata is None or metadata.empty:
        metadata = pd.DataFrame({"feature_name": features})
    metadata = metadata.copy()
    if "feature_name" not in metadata.columns:
        metadata = metadata.rename(columns={metadata.columns[0]: "feature_name"})
    metadata = metadata.drop_duplicates("feature_name")
    metadata = pd.DataFrame({"feature_name": features}).merge(metadata, on="feature_name", how="left")
    for col in REQUIRED_METADATA_COLUMNS:
        if col not in metadata.columns:
            metadata[col] = None
    for col in ["data_source", "station", "process_step"]:
        metadata[col] = metadata[col].fillna("unknown")
    return metadata[REQUIRED_METADATA_COLUMNS]
