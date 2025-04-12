from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, merge_datasets, split_data, build_preprocessing_pipeline


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
                inputs=["merged_data",  "params:data_processing"],
                outputs="cleaned_data",
                name="clean_data_node",
            ),
            node(
                func=split_data,
                inputs=["cleaned_data", "params:data_processing"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=build_preprocessing_pipeline,
                inputs=["X_train", "X_test", "params:data_processing"],
                outputs=["X_train_preprocessed", "X_test_preprocessed", "preprocessor"],
                name="build_preprocessing_pipeline_node",
            ),
        ]
    )
