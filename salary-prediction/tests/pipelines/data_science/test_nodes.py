# tests/pipelines/data_science/test_nodes.py

import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests to avoid GUI errors
import matplotlib.pyplot as plt
import os
from unittest.mock import patch, MagicMock, ANY, call # Import ANY and call for flexible mock assertions
import re # For checking file content
from pathlib import Path # Import Path for cleaner path handling

# Import functions from the nodes.py file
# Assuming nodes.py is in src/salary_prediction/pipelines/data_science/
from salary_prediction.pipelines.data_science.nodes import (
    train_baseline_model,
    train_linear_regression,
    train_randomforestregressor_model,
    train_lasso_regression,
    evaluate_model,
    evaluate_model_with_cv,
    shap_feature_selection,
    train_randomforestregressor_with_shap,
)

# --- Fixtures ---

@pytest.fixture
def sample_data():
    """Creates sample data for testing."""
    np.random.seed(42)  # for reproducibility
    X = pd.DataFrame({
        'Age': np.random.randint(22, 60, 100),
        'Years_of_Experience': np.random.uniform(0, 30, 100),
        'Feature3': np.random.randn(100) * 10,
        'Gender_Male': np.random.randint(0, 2, 100),
        'Edu_Bachelor': np.random.randint(0, 2, 100),
        'Edu_Master': np.random.randint(0, 2, 100),
    })
    # Create a somewhat realistic target variable related to features
    y = pd.Series(
        50000
        + X['Age'] * 500
        + X['Years_of_Experience'] * 1500
        + X['Gender_Male'] * 5000
        + X['Edu_Master'] * 10000
        + np.random.normal(0, 15000, 100) # Add some noise
    ).rename("Salary")

    # Ensure no negative salaries
    y[y < 10000] = 10000

    # Simulate Kedro's ParquetDataset loading behavior (returns pandas objects)
    # We'll use these directly in tests as the node functions handle them.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def sample_parameters():
    """Creates sample parameters dictionary."""
    return {
        "random_state": 42,
        "cv": 3, # Use fewer folds for faster tests
        "ci": 0.95,
        "rf_params": {
            "n_estimators": 10, # Use fewer estimators for faster tests
            "max_depth": 5,
        },
        "lasso_params": {
            "alpha": 0.1,
        },
        "feature_selection": {
             "method": "top_n", # Default method for testing
             "top_n": 3,
             "importance_threshold": 1000,
             "custom_features": ["Age", "Years_of_Experience"]
        }
    }

@pytest.fixture
def trained_baseline(sample_data):
    """Provides a trained baseline model."""
    X_train, _, y_train, _ = sample_data
    return train_baseline_model(X_train, y_train)

@pytest.fixture
def trained_linear_regression(sample_data):
    """Provides a trained linear regression model."""
    X_train, _, y_train, _ = sample_data
    return train_linear_regression(X_train, y_train)

@pytest.fixture
def trained_lasso_regression(sample_data, sample_parameters):
    """Provides a trained Lasso regression model."""
    X_train, _, y_train, _ = sample_data
    return train_lasso_regression(X_train, y_train, sample_parameters)

@pytest.fixture
def trained_random_forest(sample_data, sample_parameters):
    """Provides a trained Random Forest model."""
    X_train, _, y_train, _ = sample_data
    return train_randomforestregressor_model(X_train, y_train, sample_parameters)


# --- Test Functions ---

# 1. Test Training Functions
# -------------------------

def test_train_baseline_model(sample_data):
    """Test baseline model training."""
    X_train, _, y_train, _ = sample_data
    model = train_baseline_model(X_train, y_train.to_frame()) # Test with DataFrame y_train
    assert isinstance(model, DummyRegressor)
    # Check if the model predicts the mean
    assert model.predict(X_train).mean() == pytest.approx(y_train.mean())

    model_series = train_baseline_model(X_train, y_train) # Test with Series y_train
    assert isinstance(model_series, DummyRegressor)
    assert model_series.predict(X_train).mean() == pytest.approx(y_train.mean())


def test_train_linear_regression(sample_data):
    """Test linear regression model training."""
    X_train, _, y_train, _ = sample_data
    model = train_linear_regression(X_train, y_train.to_frame()) # Test with DataFrame y_train
    assert isinstance(model, LinearRegression)
    # Check if model is fitted (has coef_ attribute)
    assert hasattr(model, 'coef_')

    model_series = train_linear_regression(X_train, y_train) # Test with Series y_train
    assert isinstance(model_series, LinearRegression)
    assert hasattr(model_series, 'coef_')


def test_train_randomforestregressor_model(sample_data, sample_parameters):
    """Test Random Forest model training."""
    X_train, _, y_train, _ = sample_data
    params = sample_parameters

    model = train_randomforestregressor_model(X_train, y_train.to_frame(), params) # Test with DataFrame y_train
    assert isinstance(model, RandomForestRegressor)
    # Check if parameters are set correctly
    assert model.n_estimators == params["rf_params"]["n_estimators"]
    assert model.max_depth == params["rf_params"]["max_depth"]
    assert model.random_state == params["random_state"]
    # Check if model is fitted (has feature_importances_ attribute)
    assert hasattr(model, 'feature_importances_')

    model_series = train_randomforestregressor_model(X_train, y_train, params) # Test with Series y_train
    assert isinstance(model_series, RandomForestRegressor)
    assert hasattr(model_series, 'feature_importances_')


def test_train_lasso_regression(sample_data, sample_parameters):
    """Test Lasso regression model training."""
    X_train, _, y_train, _ = sample_data
    params = sample_parameters

    model = train_lasso_regression(X_train, y_train.to_frame(), params) # Test with DataFrame y_train
    assert isinstance(model, Lasso)
    # Check if parameters are set correctly
    assert model.alpha == params["lasso_params"]["alpha"]
    # Check if model is fitted (has coef_ attribute)
    assert hasattr(model, 'coef_')

    model_series = train_lasso_regression(X_train, y_train, params) # Test with Series y_train
    assert isinstance(model_series, Lasso)
    assert hasattr(model_series, 'coef_')

# 2. Test Evaluation Functions
# ---------------------------

def test_evaluate_model(trained_linear_regression, sample_data):
    """Test model evaluation without CV."""
    X_train, X_test, y_train, y_test = sample_data
    model = trained_linear_regression # Use a simple fitted model

    metrics = evaluate_model(model, X_test, y_test)

    # Calculate expected metrics manually for verification
    y_pred = model.predict(X_test)
    expected_mse = mean_squared_error(y_test, y_pred)
    expected_rmse = np.sqrt(expected_mse)
    expected_r2 = r2_score(y_test, y_pred)

    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert "model_type" in metrics

    assert metrics["mse"] == pytest.approx(expected_mse)
    assert metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics["r2_score"] == pytest.approx(expected_r2)
    assert metrics["model_type"] == "LinearRegression"


@patch('salary_prediction.pipelines.data_science.nodes.cross_val_score')
def test_evaluate_model_with_cv(mock_cv_score, trained_linear_regression, sample_data, sample_parameters):
    """Test model evaluation with CV and confidence intervals."""
    X_train, _, y_train, _ = sample_data
    model = trained_linear_regression
    params = sample_parameters

    # Define mock return values for cross_val_score
    # Make sure the number of scores matches params["cv"]
    mock_mse_scores = np.array([-2.5e8, -2.2e8, -2.8e8]) # Example negative MSE scores
    mock_r2_scores = np.array([0.75, 0.78, 0.72])        # Example R2 scores

    # Configure the mock to return different values based on the 'scoring' argument
    def cv_side_effect(*args, **kwargs):
        # Args: model, X, y, cv, scoring
        # Check the scoring argument to return the correct mock scores
        scoring = kwargs.get('scoring')
        y_arg = args[2] # Get the y argument passed to cross_val_score

        # Simulate sklearn's internal handling which extracts the Series/array
        # Note: We don't modify y_arg here, just check its type for correctness later
        # cross_val_score expects a 1D array-like structure for y.
        # The node should handle the conversion if y_train is a DataFrame.

        if scoring == 'neg_mean_squared_error':
            return mock_mse_scores
        elif scoring == 'r2':
            return mock_r2_scores
        else:
            pytest.fail(f"Unexpected scoring metric: {scoring}")

    mock_cv_score.side_effect = cv_side_effect

    # --- Test with Series y_train ---
    metrics = evaluate_model_with_cv(model, X_train, y_train, params)

    # Expected calculations based on mock scores
    expected_mse_mean = -mock_mse_scores.mean()
    expected_rmse_scores = np.sqrt(-mock_mse_scores)
    expected_rmse_mean = expected_rmse_scores.mean()
    expected_r2_mean = mock_r2_scores.mean()

    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert "mse_conf_interval" in metrics
    assert "rmse_conf_interval" in metrics
    assert "r2_score_conf_interval" in metrics
    assert "model_type" in metrics

    assert metrics["mse"] == pytest.approx(expected_mse_mean)
    assert metrics["rmse"] == pytest.approx(expected_rmse_mean)
    assert metrics["r2_score"] == pytest.approx(expected_r2_mean)
    assert isinstance(metrics["mse_conf_interval"], tuple)
    assert len(metrics["mse_conf_interval"]) == 2
    assert isinstance(metrics["rmse_conf_interval"], tuple)
    assert len(metrics["rmse_conf_interval"]) == 2
    assert isinstance(metrics["r2_score_conf_interval"], tuple)
    assert len(metrics["r2_score_conf_interval"]) == 2
    assert metrics["model_type"] == "LinearRegression"

    # Verify cross_val_score was called correctly (twice: once for mse, once for r2)
    assert mock_cv_score.call_count == 2
    # Use ANY for y_train comparison as sklearn might convert it internally
    expected_calls = [
        call(model, X_train, ANY, cv=params["cv"], scoring='neg_mean_squared_error'),
        call(model, X_train, ANY, cv=params["cv"], scoring='r2')
    ]
    mock_cv_score.assert_has_calls(expected_calls, any_order=True)

    # Check y_train passed was the original Series before potential internal conversion
    # Need to check the actual calls captured by the mock
    first_call_args = mock_cv_score.call_args_list[0][0]
    second_call_args = mock_cv_score.call_args_list[1][0]
    pd.testing.assert_series_equal(first_call_args[2], y_train)
    pd.testing.assert_series_equal(second_call_args[2], y_train)

    # --- Test with DataFrame y_train ---
    mock_cv_score.reset_mock()
    y_train_df = y_train.to_frame() # Create DataFrame version
    metrics_df = evaluate_model_with_cv(model, X_train, y_train_df, params)
    assert metrics_df["mse"] == pytest.approx(expected_mse_mean) # Check one metric for brevity
    assert mock_cv_score.call_count == 2 # Called twice more

    # Verify calls with DataFrame y_train
    expected_calls_df = [
        call(model, X_train, ANY, cv=params["cv"], scoring='neg_mean_squared_error'),
        call(model, X_train, ANY, cv=params["cv"], scoring='r2')
    ]
    mock_cv_score.assert_has_calls(expected_calls_df, any_order=True)
    # Check y_train passed to the mock was the Series extracted from the DataFrame
    first_call_args_df = mock_cv_score.call_args_list[0][0]
    second_call_args_df = mock_cv_score.call_args_list[1][0]

    # FIX: Assert that the argument passed to cross_val_score is the Series
    # This assumes the node function correctly extracts the Series from the DataFrame
    # before passing it to cross_val_score.
    assert isinstance(first_call_args_df[2], pd.Series)
    pd.testing.assert_series_equal(first_call_args_df[2], y_train, check_names=False)
    assert isinstance(second_call_args_df[2], pd.Series)
    pd.testing.assert_series_equal(second_call_args_df[2], y_train, check_names=False)


# 3. Test SHAP Functions
# ----------------------

# NOTE: We are *not* mocking 'builtins.open' anymore to avoid interfering with matplotlib/PIL internals.
# Instead, we check for file existence and content where needed.
@patch('salary_prediction.pipelines.data_science.nodes.shap.TreeExplainer')
@patch('salary_prediction.pipelines.data_science.nodes.shap.summary_plot')
@patch('salary_prediction.pipelines.data_science.nodes.plt.savefig')
@patch('salary_prediction.pipelines.data_science.nodes.plt.close')
@patch('salary_prediction.pipelines.data_science.nodes.plt.figure')
@patch('salary_prediction.pipelines.data_science.nodes.plt.tight_layout')
def test_shap_feature_selection_top_n(
    mock_tight_layout, mock_figure, mock_close, mock_savefig, mock_summary_plot, mock_tree_explainer,
    trained_random_forest, sample_data, sample_parameters, tmp_path
):
    """Test SHAP feature selection with top_n method and file generation."""
    X_train, _, y_train, _ = sample_data
    model = trained_random_forest
    params = sample_parameters.copy() # Use copy to avoid modifying fixture
    params["feature_selection"]["method"] = "top_n"
    output_dir = tmp_path / "shap_output" # Create a subdirectory in tmp_path
    output_dir.mkdir()
    output_dir_str = str(output_dir)

    # --- Mock Configuration ---
    mock_explainer_instance = MagicMock()
    mock_shap_vals = np.random.rand(*X_train.shape)
    feature_names = X_train.columns.tolist()
    # Assign importance: make 'Years_of_Experience', 'Age', 'Edu_Master' most important
    yoe_idx = feature_names.index('Years_of_Experience')
    age_idx = feature_names.index('Age')
    edu_idx = feature_names.index('Edu_Master')
    mock_shap_vals[:, yoe_idx] *= 5 # Mock mean abs ~ 2.5
    mock_shap_vals[:, age_idx] *= 4  # Mock mean abs ~ 2.0
    mock_shap_vals[:, edu_idx] *= 3  # Mock mean abs ~ 1.5

    mock_explainer_instance.shap_values.return_value = mock_shap_vals
    mock_tree_explainer.return_value = mock_explainer_instance
    # --- End Mock Configuration ---

    X_selected = shap_feature_selection(X_train, y_train, model, params, output_dir=output_dir_str)

    # --- Assertions ---
    # Check return type and shape (CORE LOGIC)
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == X_train.shape[0]
    assert X_selected.shape[1] == params["feature_selection"]["top_n"] # Should select top N features

    # Check selected features (based on mocked SHAP values) (CORE LOGIC)
    # Order might vary depending on exact mean(|SHAP|) calculation, check membership and count
    expected_top_features = ['Years_of_Experience', 'Age', 'Edu_Master']
    assert len(X_selected.columns) == len(expected_top_features)
    assert set(X_selected.columns) == set(expected_top_features)

    # Check SHAP interactions
    mock_tree_explainer.assert_called_once_with(model)
    # FIX: Remove check_additivity=ANY as it's not passed by the node
    mock_explainer_instance.shap_values.assert_called_once_with(X_train)


    # Check Plotting calls (summary and bar)
    assert mock_summary_plot.call_count == 2
    # Call 1: Default summary plot
    mock_summary_plot.assert_any_call(mock_shap_vals, X_train, show=False)
    # Call 2: Bar plot
    mock_summary_plot.assert_any_call(mock_shap_vals, X_train, plot_type="bar", show=False)

    assert mock_figure.call_count >= 1 # figure might be called internally by summary_plot too
    assert mock_tight_layout.call_count >= 1
    assert mock_savefig.call_count == 2
    assert mock_close.call_count >= 2 # Might be called more if summary_plot creates figures

    # Check file saving paths and existence
    model_name = "RandomForestRegressor"
    expected_summary_path = Path(output_dir_str) / f"{model_name}_shap_summary_plot.png"
    expected_bar_path = Path(output_dir_str) / f"{model_name}_shap_bar_plot.png"
    expected_report_path = Path(output_dir_str) / f"{model_name}_shap_feature_importance_report.txt"

    # Check if savefig was called with the correct paths
    savefig_calls = mock_savefig.call_args_list
    saved_files = [Path(c[0][0]) for c in savefig_calls] # Get the first positional argument (filepath) from each call
    assert expected_summary_path in saved_files
    assert expected_bar_path in saved_files

    # Check report file was created (we don't mock open anymore, so the file should actually be written)
    assert expected_report_path.is_file()

    # Check report file content (basic check - focusing on core info)
    # Note: Report content checks might still fail if the report generation in the node is buggy.
    # These tests focus on whether the *core selection logic* works and files are generated.
    report_content = expected_report_path.read_text()
    assert "SHAP Feature Importance Report" in report_content
    # Check if top features (based on mock importance) are mentioned in the report
    assert re.search(r"Years_of_Experience:\s+\d+\.\d+", report_content) is not None
    assert re.search(r"Age:\s+\d+\.\d+", report_content) is not None
    assert re.search(r"Edu_Master:\s+\d+\.\d+", report_content) is not None


@patch('salary_prediction.pipelines.data_science.nodes.shap.TreeExplainer')
@patch('salary_prediction.pipelines.data_science.nodes.plt.savefig') # Still mock savefig to avoid actual saving
@patch('salary_prediction.pipelines.data_science.nodes.shap.summary_plot') # Mock plots
@patch('salary_prediction.pipelines.data_science.nodes.plt.close')
@patch('salary_prediction.pipelines.data_science.nodes.plt.figure')
@patch('salary_prediction.pipelines.data_science.nodes.plt.tight_layout')
def test_shap_feature_selection_threshold(
    mock_tight, mock_fig, mock_close, mock_summary, mock_savefig, mock_tree_explainer,
    trained_random_forest, sample_data, sample_parameters, tmp_path
):
    """Test SHAP feature selection with threshold method."""
    X_train, _, y_train, _ = sample_data
    model = trained_random_forest
    params = sample_parameters.copy()
    # Set high threshold first to select fewer features
    params["feature_selection"]["method"] = "threshold"
    # Calculate expected mean abs SHAP based on mock values
    # Mocking: YOE ~ 5 * 0.5 = 2.5; Age ~ 4 * 0.5 = 2.0; Edu ~ 3 * 0.5 = 1.5; Others ~ 0.5
    params["feature_selection"]["importance_threshold"] = 1.8 # Should select YOE and Age
    output_dir = tmp_path / "shap_output_thresh"
    output_dir.mkdir()
    output_dir_str = str(output_dir)


    # Mock explainer and shap_values like before
    mock_explainer_instance = MagicMock()
    mock_shap_vals = np.random.rand(*X_train.shape) # Base values ~ mean 0.5
    feature_names = X_train.columns.tolist()
    yoe_idx = feature_names.index('Years_of_Experience')
    age_idx = feature_names.index('Age')
    edu_idx = feature_names.index('Edu_Master')
    mock_shap_vals[:, yoe_idx] *= 5 # Mock mean abs ~ 2.5
    mock_shap_vals[:, age_idx] *= 4  # Mock mean abs ~ 2.0
    mock_shap_vals[:, edu_idx] *= 3 # Mock mean abs ~ 1.5
    mock_explainer_instance.shap_values.return_value = mock_shap_vals
    mock_tree_explainer.return_value = mock_explainer_instance

    # --- Act ---
    X_selected = shap_feature_selection(X_train, y_train, model, params, output_dir=output_dir_str)

    # --- Assert ---
    # Check core selection logic
    assert isinstance(X_selected, pd.DataFrame)
    # Expecting only 'Years_of_Experience' and 'Age' to pass the threshold > 1.8
    expected_features = ['Years_of_Experience', 'Age']
    assert len(X_selected.columns) == len(expected_features)
    assert set(X_selected.columns) == set(expected_features)

    # Check report file was created and has correct content (basic check)
    model_name = "RandomForestRegressor"
    expected_report_path = Path(output_dir_str) / f"{model_name}_shap_feature_importance_report.txt"
    assert expected_report_path.is_file()
    report_content = expected_report_path.read_text()
    assert "SHAP Feature Importance Report" in report_content
    assert "Years_of_Experience" in report_content # Check selected feature is mentioned
    assert "Age" in report_content # Check selected feature is mentioned


@patch('salary_prediction.pipelines.data_science.nodes.shap.TreeExplainer')
@patch('salary_prediction.pipelines.data_science.nodes.plt.savefig') # Still mock savefig
@patch('salary_prediction.pipelines.data_science.nodes.shap.summary_plot') # Mock plots
@patch('salary_prediction.pipelines.data_science.nodes.plt.close')
@patch('salary_prediction.pipelines.data_science.nodes.plt.figure')
@patch('salary_prediction.pipelines.data_science.nodes.plt.tight_layout')
def test_shap_feature_selection_custom(
    mock_tight, mock_fig, mock_close, mock_summary, mock_savefig, mock_tree_explainer,
    trained_random_forest, sample_data, sample_parameters, tmp_path
):
    """Test SHAP feature selection with custom method."""
    X_train, _, y_train, _ = sample_data
    model = trained_random_forest
    params = sample_parameters.copy()
    params["feature_selection"]["method"] = "custom"
    # Use a subset of existing features for custom list
    custom_list = ["Age", "Gender_Male", "NonExistentFeature"]
    params["feature_selection"]["custom_features"] = custom_list
    output_dir = tmp_path / "shap_output_custom"
    output_dir.mkdir()
    output_dir_str = str(output_dir)

    # Mock explainer (SHAP values aren't strictly needed for custom logic, but function needs them for report/plots)
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.shap_values.return_value = np.random.rand(*X_train.shape)
    mock_tree_explainer.return_value = mock_explainer_instance

    # --- Act ---
    X_selected = shap_feature_selection(X_train, y_train, model, params, output_dir=output_dir_str)

    # --- Assert ---
    # Check core selection logic
    assert isinstance(X_selected, pd.DataFrame)
    # Expecting only the existing custom features
    expected_features = ["Age", "Gender_Male"]
    assert len(X_selected.columns) == len(expected_features)
    assert set(X_selected.columns) == set(expected_features)
    assert "NonExistentFeature" not in X_selected.columns # Ensure non-existent ones are ignored

    # Check report file was created and has correct content (basic check)
    model_name = "RandomForestRegressor"
    expected_report_path = Path(output_dir_str) / f"{model_name}_shap_feature_importance_report.txt"
    assert expected_report_path.is_file()
    report_content = expected_report_path.read_text()
    assert "SHAP Feature Importance Report" in report_content
    # Check that the selected features are mentioned in the report
    assert "Age" in report_content
    assert "Gender_Male" in report_content


@patch('salary_prediction.pipelines.data_science.nodes.shap_feature_selection')
@patch('salary_prediction.pipelines.data_science.nodes.RandomForestRegressor')
def test_train_randomforestregressor_with_shap(
    mock_rfr, mock_shap_select, sample_data, sample_parameters
):
    """Test training RF with SHAP feature selection integration."""
    X_train, _, y_train, _ = sample_data
    params = sample_parameters

    # --- Mock Configuration for Series y_train ---
    mock_rfr_initial_s = MagicMock(spec=RandomForestRegressor)
    mock_rfr_final_s = MagicMock(spec=RandomForestRegressor)
    mock_rfr.side_effect = [mock_rfr_initial_s, mock_rfr_final_s] # Return initial, then final

    # Mock shap_feature_selection to return a subset of columns
    selected_feature_names = ["Age", "Years_of_Experience"]
    X_train_selected_mock = X_train[selected_feature_names]
    mock_shap_select.return_value = X_train_selected_mock
    # --- End Mock Configuration ---

    # --- Act (Series y_train) ---
    final_model_s, X_selected_output_s = train_randomforestregressor_with_shap(
        X_train, y_train, params
    )

    # --- Assertions (Series y_train) ---
    # Check return values
    assert final_model_s == mock_rfr_final_s # Ensure the second (final) model is returned
    pd.testing.assert_frame_equal(X_selected_output_s, X_train_selected_mock) # Ensure the selected features df is returned

    # Check RFR instantiation calls
    assert mock_rfr.call_count == 2
    expected_rfr_kwargs = {
        "n_estimators": params["rf_params"]["n_estimators"],
        "max_depth": params["rf_params"]["max_depth"],
        "random_state": params["random_state"]
    }
    # Check both calls used the same parameters
    mock_rfr.assert_has_calls([call(**expected_rfr_kwargs), call(**expected_rfr_kwargs)])

    # Check fit calls
    # Initial fit on full data
    mock_rfr_initial_s.fit.assert_called_once()
    call_args_initial_s, _ = mock_rfr_initial_s.fit.call_args
    pd.testing.assert_frame_equal(call_args_initial_s[0], X_train)
    np.testing.assert_array_equal(call_args_initial_s[1], y_train.values) # Should be 1D numpy array

    # Final fit on selected data
    mock_rfr_final_s.fit.assert_called_once()
    call_args_final_s, _ = mock_rfr_final_s.fit.call_args
    pd.testing.assert_frame_equal(call_args_final_s[0], X_train_selected_mock)
    np.testing.assert_array_equal(call_args_final_s[1], y_train.values) # Should be 1D numpy array

    # Check shap_feature_selection call
    # FIX: Remove output_dir=None as it's not passed by the node
    mock_shap_select.assert_called_once_with(X_train, ANY, mock_rfr_initial_s, params)
    # Check y_train type passed to shap_select (should be converted to numpy array)
    y_arg_shap_s = mock_shap_select.call_args[0][1]
    assert isinstance(y_arg_shap_s, np.ndarray)
    assert y_arg_shap_s.ndim == 1
    np.testing.assert_array_equal(y_arg_shap_s, y_train.values)


    # --- Test with DataFrame y_train ---
    mock_rfr.reset_mock()
    mock_shap_select.reset_mock()

    # Setup mocks for DataFrame test
    mock_rfr_initial_df = MagicMock(spec=RandomForestRegressor)
    mock_rfr_final_df = MagicMock(spec=RandomForestRegressor)
    mock_rfr.side_effect = [mock_rfr_initial_df, mock_rfr_final_df]
    mock_shap_select.return_value = X_train_selected_mock # Reuse the selected data mock

    # Act
    final_model_df, X_selected_output_df = train_randomforestregressor_with_shap(
        X_train, y_train.to_frame(), params
    )

    # Assertions (DataFrame y_train)
    assert final_model_df == mock_rfr_final_df
    pd.testing.assert_frame_equal(X_selected_output_df, X_train_selected_mock)
    assert mock_rfr.call_count == 2
    mock_rfr.assert_has_calls([call(**expected_rfr_kwargs), call(**expected_rfr_kwargs)])

    # Check fit calls (initial)
    mock_rfr_initial_df.fit.assert_called_once()
    call_args_initial_df, _ = mock_rfr_initial_df.fit.call_args
    pd.testing.assert_frame_equal(call_args_initial_df[0], X_train)
    # Check y passed to fit is a 1D numpy array extracted from the DataFrame
    y_arg_fit_initial = call_args_initial_df[1]
    assert isinstance(y_arg_fit_initial, np.ndarray)
    assert y_arg_fit_initial.ndim == 1
    np.testing.assert_array_equal(y_arg_fit_initial, y_train.values)

    # Check fit calls (final)
    mock_rfr_final_df.fit.assert_called_once()
    call_args_final_df, _ = mock_rfr_final_df.fit.call_args
    pd.testing.assert_frame_equal(call_args_final_df[0], X_train_selected_mock)
     # Check y passed to fit is a 1D numpy array extracted from the DataFrame
    y_arg_fit_final = call_args_final_df[1]
    assert isinstance(y_arg_fit_final, np.ndarray)
    assert y_arg_fit_final.ndim == 1
    np.testing.assert_array_equal(y_arg_fit_final, y_train.values)


    # Check shap_feature_selection call
    # FIX: Remove output_dir=None as it's not passed by the node
    mock_shap_select.assert_called_once_with(X_train, ANY, mock_rfr_initial_df, params)
    # Check y_train type passed to shap_select (should be converted to numpy array)
    y_arg_shap_df = mock_shap_select.call_args[0][1]
    assert isinstance(y_arg_shap_df, np.ndarray)
    assert y_arg_shap_df.ndim == 1
    np.testing.assert_array_equal(y_arg_shap_df, y_train.values)