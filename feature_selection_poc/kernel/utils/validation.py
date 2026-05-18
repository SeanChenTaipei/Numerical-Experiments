import pandas as pd


def validate_regression_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if X.empty:
        raise ValueError("X must contain at least one feature")
