import logging

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from kedro_datasets.pandas import ParquetDataset
from typing import Dict
import numpy as np

def train_baseline_model(X_train: ParquetDataset, y_train: ParquetDataset) -> DummyRegressor:
    """Trains a baseline model using DummyRegressor.

    Args:
        X_train: Training features
        y_train: Training target
        parameters: Parameters defined in parameters.yml

    Returns:
        Trained baseline model.
    """

    # y_train is a pd.ParquetDataset, so we need to convert it to a Series
    y_train = y_train.squeeze()
    
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)

    return baseline_model


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Calculates and returns regression performance metrics.

    Args:
        regressor: Trained regression model.
        X_test: Testing data of independent features.
        y_test: True values of the target variable.

    Returns:
        Dictionary with MSE, RMSE, and R2 scores.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
        "model_type": str(type(model).__name__),
    }
