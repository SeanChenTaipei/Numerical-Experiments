from typing import Protocol
import numpy as np

class Evaluator(Protocol):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]: ...
