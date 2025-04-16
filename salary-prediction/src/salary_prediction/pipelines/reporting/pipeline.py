from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_metrics,
    load_all_metrics_from_folder,
    generate_comparison_report,
    generate_sorted_metrics_csv,
    select_best_model_by_multiple_metrics,
    get_top_n_models,
    save_top_n_metrics,
    plot_model_comparison_scatter,
    generate_detailed_metrics_table,  # <--- NEW
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=plot_metrics,
                inputs=["randomforestregressor_model_metrics", "params:reporting.rf_plot_filepath"],
                outputs=None,
                name="plot_metrics_node",
            ),
            node(
                func=load_all_metrics_from_folder,
                inputs=["params:reporting.metrics_folder", "randomforestregressor_model_with_shap_metrics"],
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
            ),
            node(
                func=generate_detailed_metrics_table,
                inputs=[
                    "all_model_metrics",
                    "params:reporting.include_conf_intervals",
                    "params:reporting.include_fold_scores",
                    "params:reporting.output_markdown_path",
                    "params:reporting.output_html_path"
                ],
                outputs="detailed_metrics_df",
                name="generate_detailed_metrics_table_node"
            )
        ]
    )
