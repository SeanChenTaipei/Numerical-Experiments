from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .utils import hash_pandas_frame


@dataclass(slots=True)
class DriftReport:
    table: pd.DataFrame
    coverage: Dict[str, str]
    compare_label: str


class DriftAnalyzer:
    """Compute drift metrics between reference and comparison splits."""

    def __init__(self, cfg: Dict[str, float]):
        self.cfg = cfg

    def compare(
        self,
        *,
        reference: pd.DataFrame,
        comparison: pd.DataFrame,
        features: List[str],
        label: str,
    ) -> DriftReport:
        rows = []
        for feature in features:
            ref = reference[feature].dropna()
            comp = comparison[feature].dropna()
            if ref.empty or comp.empty:
                continue
            if pd.api.types.is_numeric_dtype(ref):
                metrics = self._numeric_metrics(ref, comp)
            else:
                metrics = self._categorical_metrics(ref, comp)
            risk = self._evaluate_risk(metrics)
            rows.append(
                {
                    "feature": feature,
                    **metrics,
                    "risk_level": risk["level"],
                    "note": risk["note"],
                }
            )

        table = pd.DataFrame(rows).sort_values("risk_level", ascending=False)
        coverage = {
            "ref_hash": hash_pandas_frame(reference[features]),
            "cmp_hash": hash_pandas_frame(comparison[features]),
        }
        return DriftReport(table=table, coverage=coverage, compare_label=label)

    def _numeric_metrics(self, ref: pd.Series, comp: pd.Series) -> Dict[str, float]:
        ks = stats.ks_2samp(ref, comp, alternative="two-sided").statistic
        cvm = stats.cramervonmises_2samp(ref, comp).statistic
        psi = _population_stability(ref, comp)
        wass = stats.wasserstein_distance(ref, comp)
        js = _jensen_shannon(ref, comp, bins=20)
        tv = _total_variation(ref, comp, bins=20)
        return {"ks": ks, "cvm": cvm, "psi": psi, "wasserstein": wass, "js": js, "tv": tv}

    def _categorical_metrics(self, ref: pd.Series, comp: pd.Series) -> Dict[str, float]:
        ref_counts = ref.value_counts(normalize=True)
        comp_counts = comp.value_counts(normalize=True).reindex(ref_counts.index, fill_value=0.0)
        psi = ((ref_counts - comp_counts) * np.log((ref_counts + 1e-9) / (comp_counts + 1e-9))).sum()
        js = stats.entropy(ref_counts, qk=comp_counts) + stats.entropy(comp_counts, qk=ref_counts)
        tv = 0.5 * np.abs(ref_counts - comp_counts).sum()
        return {"ks": np.nan, "cvm": np.nan, "psi": psi, "wasserstein": np.nan, "js": js, "tv": tv}

    def _evaluate_risk(self, metrics: Dict[str, float]) -> Dict[str, str]:
        thresholds = self.cfg.get("thresholds", {})
        risk_level = "low"
        notes = []
        for metric, value in metrics.items():
            if np.isnan(value):
                continue
            threshold = thresholds.get(metric)
            if threshold and value >= threshold:
                risk_level = "high"
                notes.append(f"{metric}>{threshold}")
        note = ", ".join(notes) if notes else "ok"
        return {"level": risk_level, "note": note}


def _population_stability(ref: pd.Series, comp: pd.Series, bins: int = 20) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = ref.quantile(quantiles).drop_duplicates().values
    if len(breaks) < 2:
        min_val, max_val = ref.min(), ref.max() + 1e-6
        breaks = np.linspace(min_val, max_val, bins + 1)
    ref_counts, _ = np.histogram(ref, bins=breaks)
    comp_counts, _ = np.histogram(comp, bins=breaks)
    ref_perc = ref_counts / ref_counts.sum()
    comp_perc = comp_counts / max(comp_counts.sum(), 1)
    psi = np.sum((ref_perc - comp_perc) * np.log((ref_perc + 1e-6) / (comp_perc + 1e-6)))
    return float(psi)


def _jensen_shannon(ref: pd.Series, comp: pd.Series, bins: int) -> float:
    hist_ref, edges = np.histogram(ref, bins=bins, density=True)
    hist_comp, _ = np.histogram(comp, bins=edges, density=True)
    m = 0.5 * (hist_ref + hist_comp)
    return 0.5 * (stats.entropy(hist_ref, m) + stats.entropy(hist_comp, m))


def _total_variation(ref: pd.Series, comp: pd.Series, bins: int) -> float:
    hist_ref, edges = np.histogram(ref, bins=bins, density=True)
    hist_comp, _ = np.histogram(comp, bins=edges, density=True)
    return 0.5 * np.abs(hist_ref - hist_comp).sum()
