from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_metrics
)


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=plot_metrics,
                inputs=["randomforestregressor_model_metrics", "params:reporting.rf_plot_filepath"],  # Inputs for the plot
                outputs=None,  # No Kedro dataset output for the plot node
                name="plot_metrics_node",
            ),
        ]
    )
