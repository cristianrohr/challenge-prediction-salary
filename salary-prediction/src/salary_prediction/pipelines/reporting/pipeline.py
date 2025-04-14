from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_metrics, select_best_model, generate_comparison_report, load_all_metrics_from_folder, generate_sorted_metrics_csv
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

            node(
                func=load_all_metrics_from_folder,
                inputs="params:reporting.metrics_folder",
                outputs="all_model_metrics",
                name="load_all_metrics_node",
            ),
            node(
                func=select_best_model,
                inputs=["all_model_metrics", "params:reporting.selection_metric", "params:reporting.selection_ascending"],
                outputs="best_model_result",
                name="select_best_model_node",
            ),
            node(
                func=generate_comparison_report,
                inputs=["all_model_metrics", "params:reporting.full_metrics_report_path"],
                outputs=None,
                name="generate_model_comparison_report_node",
            ),
            node(
                func=generate_sorted_metrics_csv,
                inputs=["all_model_metrics", "params:reporting.selection_metric", "params:reporting.selection_ascending"],
                outputs="sorted_model_metrics",
                name="generate_sorted_metrics_csv_node"
            )
        ]
    )
