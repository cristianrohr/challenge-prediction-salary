import logging

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from kedro_datasets.pandas import ParquetDataset
from typing import Dict
import numpy as np
from sklearn.model_selection import cross_val_score
import scipy.stats as st

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

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains a linear regression model.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained linear regression model.
    """
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    return linear_regression


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


def evaluate_model_with_cv(
    model, X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> dict[str, float]:
    """
    Calculates and returns regression performance metrics using cross-validation 
    and calculates 95% confidence intervals for each metric.

    Args:
        model: Trained regression model.
        X_train: Training data of independent features.
        y_train: True values of the target variable.
        cv: Number of folds for cross-validation.

    Returns:
        Dictionary with MSE, RMSE, and R2 scores along with 95% confidence intervals.
    """
    mse_scores = cross_val_score(model, X_train, y_train, cv=parameters["cv"], scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-mse_scores)  # Negate MSE for RMSE
    r2_scores = cross_val_score(model, X_train, y_train, cv=parameters["cv"], scoring="r2")

    mse_mean = -mse_scores.mean()
    rmse_mean = rmse_scores.mean()
    r2_mean = r2_scores.mean()

    mse_conf_int = st.t.interval(parameters["ci"], len(mse_scores)-1, loc=mse_mean, scale=st.sem(mse_scores))
    rmse_conf_int = st.t.interval(parameters["ci"], len(rmse_scores)-1, loc=rmse_mean, scale=st.sem(rmse_scores))
    r2_conf_int = st.t.interval(parameters["ci"], len(r2_scores)-1, loc=r2_mean, scale=st.sem(r2_scores))

    return {
        "mse": mse_mean,
        "rmse": rmse_mean,
        "r2_score": r2_mean,
        "mse_conf_interval": mse_conf_int,
        "rmse_conf_interval": rmse_conf_int,
        "r2_score_conf_interval": r2_conf_int,
        "model_type": str(type(model).__name__),
    }