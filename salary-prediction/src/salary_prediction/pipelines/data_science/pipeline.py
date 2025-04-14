from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model, train_baseline_model, train_elastic_net, train_linear_regression,evaluate_model_with_cv,
    train_randomforestregressor_model, train_randomforestregressor_with_shap,
    train_lasso_regression, train_elastic_net, optimize_elastic_net_hyperparameters)


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
                inputs=["X_train_preprocessed", "y_train", "params:data_science"],
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
                func=train_lasso_regression,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="lasso_regression_model_v2",
                name="train_lasso_regression_model_v2_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["lasso_regression_model_v2", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="lasso_regression_model_v2_metrics",
                name="evaluate_lasso_regression_model_v2_node",
            ),
            node(
                func=train_elastic_net,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="elastic_net_model",
                name="train_elastic_net_model_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["elastic_net_model", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="elastic_net_model_metrics",
                name="evaluate_elastic_net_model_node",
            ),
            node(
                func=optimize_elastic_net_hyperparameters,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="optimized_elastic_net_params",
                name="optimize_elastic_net_hyperparameters_node",
            ),
            node(
                func=train_elastic_net,
                inputs=["X_train_preprocessed_v2", "y_train", "optimized_elastic_net_params"],
                outputs="elastic_net_model_optimized",
                name="train_elastic_net_model_optimized_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["elastic_net_model_optimized", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="elastic_net_model_optimized_metrics",
                name="evaluate_elastic_net_model_optimized_node",
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
                func=train_randomforestregressor_model,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_v2",
                name="train_randomforestregressor_model_v2_node",
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model_v2", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_v2_metrics",
                name="evaluate_randomforestregressor_model_v2_node",
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
