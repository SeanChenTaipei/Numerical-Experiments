"""Markdown summary report builder."""
from __future__ import annotations

from pathlib import Path

from feature_selection_poc.kernel.entities import FeatureSelectionResult


def build_summary_report(result: FeatureSelectionResult, output_path: str | Path) -> None:
    lines = [
        "# Feature Selection POC Summary",
        f"Selected feature count: {len(result.selected_features)}",
        f"Removed feature count: {len(result.removed_features)}",
        "",
        "## Selected Features",
        *[f"- {f}" for f in result.selected_features],
        "",
        "## Model Evaluation",
        *[f"- {k}: {v}" for k, v in result.model_evaluation.items()],
    ]
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines))
