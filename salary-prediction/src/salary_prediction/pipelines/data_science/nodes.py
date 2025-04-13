import logging

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from kedro_datasets.pandas import ParquetDataset
from typing import Dict
import numpy as np
from sklearn.model_selection import cross_val_score
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
import shap
import shap.plots
import os
import matplotlib.pyplot as plt


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

def train_linear_regression(X_train:ParquetDataset, y_train: ParquetDataset) -> LinearRegression:
    """Trains a linear regression model.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained linear regression model.
    """

    y_train = y_train.squeeze()

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    return linear_regression

def train_randomforestregressor_model(X_train: ParquetDataset, y_train: ParquetDataset, parameters: Dict) -> RandomForestRegressor:
    """Trains a Random Forest Regressor model.

    Args:
        X_train: Training features
        y_train: Training target
        parameters: Parameters defined in parameters.yml

    Returns:
        Trained Random Forest Regressor model.
    """

    y_train = y_train.squeeze()

    rf_model = RandomForestRegressor(
        n_estimators=parameters["rf_params"]["n_estimators"],
        max_depth=parameters["rf_params"]["max_depth"],
        random_state=parameters["random_state"],
    )
    rf_model.fit(X_train, y_train)

    return rf_model


def train_lasso_regression(X_train: ParquetDataset, y_train: ParquetDataset, parameters: Dict) -> Lasso:
    """Trains a Lasso Regression model.

    Args:
        X_train: Training features
        y_train: Training target
        parameters: Parameters defined in parameters.yml

    Returns:
        Trained Lasso Regression model.
    """

    y_train = y_train.squeeze()

    lasso_model = Lasso(alpha=parameters["lasso_params"]["alpha"])
    lasso_model.fit(X_train, y_train)

    return lasso_model


def evaluate_model(
    model, X_test: ParquetDataset, y_test: ParquetDataset
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
    model, X_train: ParquetDataset, y_train: ParquetDataset, parameters: dict
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

    y_train = y_train.squeeze()


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



def shap_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, model, parameters: dict, output_dir: str = "data/08_reporting") -> pd.DataFrame:
    """Selects important features using SHAP and generates a report and plot.

    Args:
        X_train: Training features.
        y_train: Training target.
        model: Trained machine learning model.
        parameters: Parameters from yml file
        output_dir: Directory where the report and plot will be saved.

    Returns:
        X_train with only the selected features.
    """
    
    # Train a model (e.g., RandomForest) to get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    model_name = str(type(model).__name__)

    # If shap_values is a list (classification case with multiple classes), select the first class' SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # For binary/multi-class classification, select the first class

    # If shap_values is 3D (multi-output regression), reduce it to 2D by selecting the first output
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]  # Take the first output for multi-output regression

    # Compute mean absolute SHAP values for each feature
    shap_importance = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.Series(shap_importance, index=X_train.columns)

    # Select features based on the method specied in the parameters
    selection_method = parameters["feature_selection"]["method"]

    if selection_method == "custom":
        custom_features = parameters.get("feature_selection", {}).get("custom_features", [])
        selected_features = [f for f in custom_features if f in X_train.columns]
    elif selection_method == "threshold":
        threshold = parameters.get("feature_selection", {}).get("importance_threshold", 1000)
        selected_features = feature_importance[feature_importance >= threshold].index.tolist()
    else:  # Default to "top_n"
        top_n = parameters.get("feature_selection", {}).get("top_n", 10)
        selected_features = feature_importance.nlargest(top_n).index.tolist()

    # Generate SHAP summary plot (global feature importance)
    plt.figure(figsize=(10, 8))  # Set figure size explicitly
    shap.summary_plot(shap_values, X_train, show=False)  # Add show=False to prevent immediate display
    plt.tight_layout()  # Adjust layout
    shap_plot_path = os.path.join(output_dir, f"{model_name}_shap_summary_plot.png")
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=150)  # Increase DPI and use tight bbox
    plt.close()  # Close the plot

    # Generate SHAP bar plot (specific feature importance)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    shap_bar_plot_path = os.path.join(output_dir, f"{model_name}_shap_bar_plot.png")
    plt.savefig(shap_bar_plot_path)
    plt.close()  # Close the plot to avoid displaying it immediately
    
    # Create the report
    report_file_path = os.path.join(output_dir, f"{model_name}_shap_feature_importance_report.txt")
    with open(report_file_path, 'w') as f:
        f.write("SHAP Feature Importance Report\n")
        f.write("="*30 + "\n")
        f.write(f"Top 10 selected features based on SHAP values:\n\n")
        
        for feature, importance in feature_importance.nlargest(10).items():
            f.write(f"{feature}: {importance}\n")

    # Return the dataset with selected features
    return X_train[selected_features]


def train_randomforestregressor_with_shap(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    parameters: Dict
) -> RandomForestRegressor:
    """Train RandomForestRegressor with feature selection based on SHAP values.

    Args:
        X_train: Training features.
        y_train: Training target.
        parameters: Parameters from Kedro's catalog.

    Returns:
        Trained Random Forest model.
    """

     # Convert y_train to 1D array if it's a DataFrame or column vector
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.ravel()  # Convert DataFrame to 1D numpy array
    elif isinstance(y_train, pd.Series):
        y_train = y_train.values  # Convert Series to 1D numpy array
    elif hasattr(y_train, 'shape') and len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()  # Convert numpy column vector to 1D array

    # Train a preliminary RandomForestRegressor to use for SHAP
    rf_model = RandomForestRegressor(
        n_estimators=parameters["rf_params"]["n_estimators"],
        max_depth=parameters["rf_params"]["max_depth"],
        random_state=parameters["random_state"]
    )
    rf_model.fit(X_train, y_train)

    # Perform feature selection using SHAP
    X_train_selected = shap_feature_selection(X_train, y_train, rf_model, parameters)

    # Train the final model on the selected features
    rf_model_selected = RandomForestRegressor(
        n_estimators=parameters["rf_params"]["n_estimators"],
        max_depth=parameters["rf_params"]["max_depth"],
        random_state=parameters["random_state"]
    )
    rf_model_selected.fit(X_train_selected, y_train)

    return rf_model_selected, X_train_selected