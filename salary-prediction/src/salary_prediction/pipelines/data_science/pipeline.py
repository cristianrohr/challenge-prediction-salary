from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model, train_baseline_model, train_elastic_net, train_linear_regression,evaluate_model_with_cv,
    train_randomforestregressor_model, select_features_with_shap, train_randomforestregressor_model_on_selected,
    train_lasso_regression, train_elastic_net, optimize_elastic_net_hyperparameters,
    optimize_randomforest_hyperparameters, optimize_xgboost_hyperparameters, train_xgboost_model)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_baseline_model,
                inputs=["X_train_preprocessed", "y_train"],
                outputs="baseline_model",
                name="train_baseline_model_node",
                tags=["data_science", "baseline", "uses_pp_v1"],
            ),
            node(
                func=evaluate_model,
                inputs=["baseline_model", "X_test_preprocessed", "y_test"],
                outputs="baseline_model_metrics",
                name="evaluate_baseline_model_node",
                tags=["data_science", "baseline", "eval", "uses_pp_v1"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["baseline_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="baseline_model_ci_metrics",
                name="evaluate_baseline_model_ci_node",
                tags=["data_science", "baseline", "eval", "uses_pp_v1"],
            ),
            node(
                func=train_linear_regression,
                inputs=["X_train_preprocessed", "y_train"],
                outputs="linear_regression_model",
                name="train_linear_regression_model_node",
                tags=["data_science", "linear", "uses_pp_v1"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["linear_regression_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="linear_regression_model_metrics",
                name="evaluate_linear_regression_model_node",
                tags=["data_science", "linear", "eval", "uses_pp_v1"],
            ),
            node(
                func=train_lasso_regression,
                inputs=["X_train_preprocessed", "y_train", "params:data_science"],
                outputs="lasso_regression_model",
                name="train_lasso_regression_model_node",
                tags=["data_science", "lasso", "uses_pp_v1"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["lasso_regression_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="lasso_regression_model_metrics",
                name="evaluate_lasso_regression_model_node",
                tags=["data_science", "lasso", "eval", "uses_pp_v1"],
            ),
            node(
                func=train_lasso_regression,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="lasso_regression_model_v2",
                name="train_lasso_regression_model_v2_node",
                tags=["data_science", "lasso", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["lasso_regression_model_v2", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="lasso_regression_model_v2_metrics",
                name="evaluate_lasso_regression_model_v2_node",
                tags=["data_science", "lasso", "eval", "uses_pp_v2"],
            ),
            node(
                func=train_elastic_net,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="elastic_net_model",
                name="train_elastic_net_model_node",
                tags=["data_science", "elastic", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["elastic_net_model", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="elastic_net_model_metrics",
                name="evaluate_elastic_net_model_node",
                tags=["data_science", "elastic", "eval", "uses_pp_v2"],
            ),
            node(
                func=optimize_elastic_net_hyperparameters,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="optimized_elastic_net_params",
                name="optimize_elastic_net_hyperparameters_node",
                tags=["data_science", "elastic", "optimized", "uses_pp_v2"],
            ),
            node(
                func=train_elastic_net,
                inputs=["X_train_preprocessed_v2", "y_train", "optimized_elastic_net_params"],
                outputs="elastic_net_model_optimized",
                name="train_elastic_net_model_optimized_node",
                tags=["data_science", "elastic", "optimized", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["elastic_net_model_optimized", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="elastic_net_model_optimized_metrics",
                name="evaluate_elastic_net_model_optimized_node",
                tags=["data_science", "elastic", "optimized", "eval", "uses_pp_v2"],
            ),
            node(
                func=train_randomforestregressor_model,
                inputs=["X_train_preprocessed", "y_train", "params:data_science"],
                outputs="randomforestregressor_model",
                name="train_randomforestregressor_model_node",
                tags=["data_science", "rf", "uses_pp_v1"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model", "X_train_preprocessed", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_metrics",
                name="evaluate_randomforestregressor_model_node",
                tags=["data_science", "rf", "eval", "uses_pp_v1"],
            ),
            node(
                func=train_randomforestregressor_model,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_v2",
                name="train_randomforestregressor_model_v2_node",
                tags=["data_science", "rf", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model_v2", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_v2_metrics",
                name="evaluate_randomforestregressor_model_v2_node",
                tags=["data_science", "rf", "eval", "uses_pp_v2"],
            ),
            node(
                func=optimize_randomforest_hyperparameters,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="optimized_rf_params",
                name="optimize_randomforest_hyperparameters_node",
                tags=["data_science", "rf", "optimized", "uses_pp_v2"],
            ),
            node(
                func=train_randomforestregressor_model,
                inputs=["X_train_preprocessed_v2", "y_train", "optimized_rf_params"],
                outputs="randomforestregressor_model_optimized",
                name="train_randomforestregressor_model_optimized_node",
                tags=["data_science", "rf", "optimized", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model_optimized", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_optimized_metrics",
                name="evaluate_randomforestregressor_model_optimized_node",
                tags=["data_science", "rf", "optimized", "eval", "uses_pp_v2"],
            ),
            node(
                func=optimize_xgboost_hyperparameters,
                inputs=["X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="optimized_xgb_params",
                name="optimize_xgboost_hyperparameters_node",
                tags=["data_science", "xgb", "optimized", "uses_pp_v2"],
            ),
            node(
                func=train_xgboost_model,
                inputs=["X_train_preprocessed_v2", "y_train", "optimized_xgb_params"],
                outputs="xgboost_model_optimized",
                name="train_xgboost_model_optimized_node",
                tags=["data_science", "xgb", "optimized", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["xgboost_model_optimized", "X_train_preprocessed_v2", "y_train", "params:data_science"],
                outputs="xgboost_model_optimized_metrics",
                name="evaluate_xgboost_model_optimized_node",
                tags=["data_science", "xgb", "optimized", "eval", "uses_pp_v2"],
            ),
            node(
                func=select_features_with_shap,
                inputs=["X_train_preprocessed_v2", "X_test_preprocessed_v2", "y_train", "params:data_science"],
                outputs=["X_train_selected_shap", "X_test_selected_shap"],
                name="select_features_with_shap_node",
                tags=["data_science", "shap", "selector", "uses_pp_v2"]
            ),
            node(
                func=train_randomforestregressor_model_on_selected,
                inputs=["X_train_selected_shap", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_with_shap",
                name="train_randomforestregressor_model_with_shap_node",
                tags=["data_science", "rf", "shap", "uses_pp_v2"]
            ),
            node(
                func=evaluate_model_with_cv,
                inputs=["randomforestregressor_model_with_shap", "X_train_selected_shap", "y_train", "params:data_science"],
                outputs="randomforestregressor_model_with_shap_metrics",
                name="evaluate_randomforestregressor_model_with_shap_node",
                tags=["data_science", "rf", "shap", "eval", "uses_pp_v2"],
            ),
            node(
                func=evaluate_model,
                inputs=["xgboost_model_optimized", "X_test_preprocessed_v2", "y_test"],
                outputs="best_model_metrics_on_test",
                name="evaluate_best_model_on_test_node",
                tags=["data_science", "xgb", "optimized", "eval", "uses_pp_v2", "best_model"],
            ),
        ]
    )


## Add final model test here, this is not elegant, but i'm running out of time for the deadline