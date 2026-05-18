from typing import Protocol
import pandas as pd

class FeatureScorer(Protocol):
    def score(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: ...
