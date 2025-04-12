from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_baseline_model


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
        ]
    )
