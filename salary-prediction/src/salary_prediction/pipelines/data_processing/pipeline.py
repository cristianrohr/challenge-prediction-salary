from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, merge_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_datasets,
                inputs=["salary_raw", "people_raw", "descriptions_raw"],
                outputs="merged_data",
                name="merge_datasets_node",
            ),
            node(
                func=clean_data,
                inputs="merged_data",
                outputs="cleaned_data",
                name="clean_data_node",
            )
        ]
    )
