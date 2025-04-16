import logging
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
import optuna
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
import random
from sklearn.base import clone

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

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

def optimize_randomforest_hyperparameters(X_train: ParquetDataset, y_train: ParquetDataset, parameters: Dict) -> Dict:
    """Optimizes hyperparameters for RandomForestRegressor using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        parameters: Parameters defined in parameters.yml
        
    Returns:
        Dictionary with optimized hyperparameters
    """   
    y_train = y_train.squeeze()
    
    def objective(trial):
        # Define the hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        # Create and evaluate model with current hyperparameters
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=parameters["random_state"]
        )
        
        # Use cross-validation to evaluate the model
        cv_scores = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=parameters["cv"],
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        # Return the mean negative MSE (Optuna minimizes the objective)
        return cv_scores.mean()
    
    # Create and run the study
    sampler = optuna.samplers.TPESampler(seed=parameters["random_state"])
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective, 
        n_trials=parameters["rf_optuna"]["n_trials"],
        timeout=parameters["rf_optuna"].get("timeout", None)
    )
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best CV score: {best_value}")
    
    # Return best parameters
    return {
        "rf_params": {
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "min_samples_split": best_params["min_samples_split"],
            "min_samples_leaf": best_params["min_samples_leaf"],
            "max_features": best_params["max_features"],
        },
        "random_state": parameters["random_state"]
    }

from xgboost import XGBRegressor

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> XGBRegressor:
    """Trains an XGBoost Regressor model.

    Args:
        X_train: Training features.
        y_train: Training target.
        parameters: Parameters defined in parameters.yml.

    Returns:
        Trained XGBoost Regressor model.
    """
    y_train = y_train.squeeze()

    xgb_model = XGBRegressor(
        n_estimators=parameters["xgb_params"]["n_estimators"],
        max_depth=parameters["xgb_params"]["max_depth"],
        learning_rate=parameters["xgb_params"]["learning_rate"],
        subsample=parameters["xgb_params"]["subsample"],
        colsample_bytree=parameters["xgb_params"]["colsample_bytree"],
        random_state=parameters["random_state"],
        tree_method="hist",          # Ensures deterministic hist-based tree
        enable_categorical=False,    # Avoid automatic detection
        n_jobs=1                     # Forces single-threaded execution
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def optimize_xgboost_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> Dict:
    """Optimizes hyperparameters for XGBoost Regressor using Optuna."""

    y_train = y_train.squeeze()

    def objective(trial):
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = XGBRegressor(
            **param_grid,
            random_state=parameters["random_state"],
            tree_method="hist",          # Ensures deterministic hist-based tree
            enable_categorical=False,    # Avoid automatic detection
            n_jobs=1                     # Forces single-threaded execution
        )

        scores = cross_val_score(
            model, X_train, y_train,
            cv=parameters["cv"],
            scoring="neg_mean_squared_error",
            n_jobs=1
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=parameters["random_state"])
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective,
        n_trials=parameters["xgb_optuna"]["n_trials"],
        timeout=parameters["xgb_optuna"].get("timeout", None)
    )

    best_params = study.best_params
    logging.info(f"Best XGBoost params: {best_params}")

    return {
        "xgb_params": best_params,
        "random_state": parameters["random_state"]
    }


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

def train_elastic_net(X_train: ParquetDataset, y_train: ParquetDataset, parameters: Dict) -> ElasticNet:
    """Trains a Elastic Net Regression model.

    Args:
        X_train: Training features
        y_train: Training target
        parameters: Parameters defined in parameters.yml

    Returns:
        Trained Elastic Net Regression model.
    """

    y_train = y_train.squeeze()

    elastic_net_model = ElasticNet(
        alpha=parameters["elastic_net_params"]["alpha"],
        l1_ratio=parameters["elastic_net_params"]["l1_ratio"],
        selection='cyclic'
    )
    elastic_net_model.fit(X_train, y_train)

    return elastic_net_model

def optimize_elastic_net_hyperparameters(X_train: ParquetDataset, y_train: ParquetDataset, parameters: Dict) -> Dict:
    """Optimizes hyperparameters for Elastic Net using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        parameters: Parameters defined in parameters.yml
        
    Returns:
        Dictionary with optimized hyperparameters
    """
    
    y_train = y_train.squeeze()
    
    def objective(trial):
        # Define the hyperparameter search space
        alpha = trial.suggest_float('alpha', 0.001, 1.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.01, 1.0)
        
        # Create and evaluate model with current hyperparameters
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=parameters["random_state"],
            selection = 'cyclic'
        )
        
        # Use cross-validation to evaluate the model
        cv_scores = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=parameters["cv"],
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        # Return the mean negative MSE (Optuna minimizes the objective)
        return cv_scores.mean()
    
    # Create and run the study
    sampler = optuna.samplers.TPESampler(seed=parameters["random_state"])
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective, 
        n_trials=parameters["elastic_net_optuna"]["n_trials"],
        timeout=parameters["elastic_net_optuna"].get("timeout", None)
    )
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best CV score: {best_value}")
    
    # Return best parameters
    return {
        "elastic_net_params": {
            "alpha": best_params["alpha"],
            "l1_ratio": best_params["l1_ratio"]
        }
    }

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
        model_name: Name of the model for which the metrics are calculated.

    Returns:
        Dictionary with MSE, RMSE, and R2 scores along with 95% confidence intervals.
    """

    y_train = y_train.squeeze()
    
    try:
        model = clone(model)
    except Exception as e:
        raise ValueError(f"Model {model} cannot be cloned for CV. Use unfitted model instead. Error: {e}")

    mse_scores = cross_val_score(model, X_train, y_train, cv=parameters["cv"], scoring="neg_mean_squared_error", n_jobs=1)
    rmse_scores = np.sqrt(-mse_scores)  # Negate MSE for RMSE
    r2_scores = cross_val_score(model, X_train, y_train, cv=parameters["cv"], scoring="r2", n_jobs=1)

    mse_mean = -mse_scores.mean()
    rmse_mean = rmse_scores.mean()
    r2_mean = r2_scores.mean()

    mse_conf_int = st.t.interval(parameters["ci"], len(mse_scores)-1, loc=mse_mean, scale=st.sem(mse_scores))
    rmse_conf_int = st.t.interval(parameters["ci"], len(rmse_scores)-1, loc=rmse_mean, scale=st.sem(rmse_scores))
    r2_conf_int = st.t.interval(parameters["ci"], len(r2_scores)-1, loc=r2_mean, scale=st.sem(r2_scores))

    return {
        "model_type": str(type(model).__name__),
        #"model_name": str(type(model).__name__),
        "mse": mse_mean,
        "rmse": rmse_mean,
        "r2_score": r2_mean,
        "mse_conf_interval": mse_conf_int,
        "rmse_conf_interval": rmse_conf_int,
        "r2_score_conf_interval": r2_conf_int,
        "mse_folds": list(-mse_scores),
        "rmse_folds": list(rmse_scores),
        "r2_folds": list(r2_scores),
    }


def select_features_with_shap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict
) -> pd.DataFrame:
    """
    Computes SHAP values from a RandomForest model and selects features based on importance.

    Args:
        X_train: Training data of independent features.
        X_test: Test data of independent features.
        y_train: True values of the target variable.
        parameters: Parameters defined in parameters.yml

    Returns:
        Tuple of (X_train_selected, X_test_selected) with selected features.
    """
    from sklearn.ensemble import RandomForestRegressor
    import shap
    import os
    import matplotlib.pyplot as plt

    y_train = y_train.squeeze()

    rf_model = RandomForestRegressor(
        n_estimators=parameters["rf_params"]["n_estimators"],
        max_depth=parameters["rf_params"]["max_depth"],
        random_state=parameters["random_state"]
    )
    rf_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):  # multiclass
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(shap_importance, index=X_train.columns)

    selection_method = parameters["feature_selection"]["method"]

    if selection_method == "custom":
        selected_features = parameters["feature_selection"].get("custom_features", [])
        selected_features = [f for f in selected_features if f in X_train.columns]
    elif selection_method == "threshold":
        threshold = parameters["feature_selection"]["importance_threshold"]
        selected_features = feature_importance[feature_importance >= threshold].index.tolist()
    else:
        top_n = parameters["feature_selection"].get("top_n", 10)
        selected_features = feature_importance.nlargest(top_n).index.tolist()

    # Optional report/plots
    report_dir = "data/08_reporting"
    os.makedirs(report_dir, exist_ok=True)

    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "shap_summary_plot.png"), bbox_inches='tight', dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig(os.path.join(report_dir, "shap_bar_plot.png"), bbox_inches='tight', dpi=150)
    plt.close()

    return X_train[selected_features], X_test[selected_features]

def train_randomforestregressor_model_on_selected(
    X_train_selected: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor using SHAP-selected features.
    """
    y_train = y_train.squeeze()

    model = RandomForestRegressor(
        n_estimators=parameters["rf_params"]["n_estimators"],
        max_depth=parameters["rf_params"]["max_depth"],
        random_state=parameters["random_state"]
    )
    model.fit(X_train_selected, y_train)

    return model
