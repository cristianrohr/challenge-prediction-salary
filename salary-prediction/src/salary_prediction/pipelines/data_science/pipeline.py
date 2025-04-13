from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model, train_baseline_model, train_linear_regression,evaluate_model_with_cv,
    train_randomforestregressor_model, train_randomforestregressor_with_shap,
    train_lasso_regression)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_baseline_model,
                inputs=["X_train_preprocessed", "y_train"],
                outputs="baseline_model",
                name="train_baseline_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["baseline_model", "X_test_preprocessed", "y_test"],
                outputs="baseline_model_metrics",
                name="evaluate_baseline_model_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["baseline_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="baseline_model_ci_metrics",
                name="evaluate_baseline_model_ci_node",
            ),
            node(
                func=train_linear_regression,
                inputs=["X_train_preprocessed", "y_train"],
                outputs="linear_regression_model",
                name="train_linear_regression_model_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["linear_regression_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="linear_regression_model_metrics",
                name="evaluate_linear_regression_model_node",
            ),
            node(
                func=train_lasso_regression,
                inputs=["X_train_preprocessed", "y_train"],
                outputs="lasso_regression_model",
                name="train_lasso_regression_model_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["lasso_regression_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="lasso_regression_model_metrics",
                name="evaluate_lasso_regression_model_node",
            ),
            node(
                func=train_randomforestregressor_model,
                inputs=["X_train_preprocessed", "y_train", "params:data_science"],
                outputs="randomforestregressor_model",
                name="train_randomforestregressor_model_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_metrics",
                name="evaluate_randomforestregressor_model_node",
            ),
            node(
                func=train_randomforestregressor_with_shap,
                inputs=["X_train_preprocessed", "y_train", "params:data_science"],
                outputs=["randomforestregressor_model_with_shap",  "X_train_selected_shap"],
                name="train_randomforestregressor_model_with_shap_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model_with_shap", "X_train_selected_shap", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_with_shap_metrics",
                name="evaluate_randomforestregressor_model_with_shap_node",
            ),

        ]
    )
