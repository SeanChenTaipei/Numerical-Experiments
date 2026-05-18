from typing import Protocol
import pandas as pd

class FeatureSelector(Protocol):
    def select(self, scorebook: pd.DataFrame) -> list[str]: ...
