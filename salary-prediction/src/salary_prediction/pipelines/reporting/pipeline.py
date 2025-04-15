from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_metrics, load_all_metrics_from_folder, generate_comparison_report, generate_sorted_metrics_csv,
    select_best_model_by_multiple_metrics, get_top_n_models, save_top_n_metrics, plot_model_comparison_scatter
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
            func=select_best_model_by_multiple_metrics,
            inputs=[
                "all_model_metrics",
                "params:reporting.metrics_list",
                "params:reporting.ascending_flags"
            ],
            outputs="best_model_result",
            name="select_best_model_by_multiple_metrics_node",
            ),
            node(
                func=get_top_n_models,
                inputs=[
                    "all_model_metrics",
                    "params:reporting.selection_metric",
                    "params:reporting.top_n"
                ],
                outputs="top_models",
                name="get_top_n_models_node",
            ),
            node(
                func=save_top_n_metrics,
                inputs=["top_models", "params:reporting.top_n_metrics_path"],
                outputs=None,
                name="save_top_n_metrics_node",
            ),
            node(
                func=plot_model_comparison_scatter,
                inputs=["all_model_metrics", "params:reporting.scatter_plot_path"],
                outputs=None,
                name="plot_model_comparison_scatter_node",
        )
        ]
    )
