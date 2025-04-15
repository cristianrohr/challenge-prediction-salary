from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, merge_datasets, split_data, build_preprocessing_pipeline, build_preprocessing_pipeline_v2


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_datasets,
                inputs=["salary_raw", "people_raw"],
                outputs="merged_data",
                name="merge_datasets_node",
                tags=["data_processing", "merge"],
            ),
            node(
                func=clean_data,
                inputs=["merged_data", "params:data_processing"],
                outputs="cleaned_data",
                name="clean_data_node",
                tags=["data_processing", "clean"],
            ),
            node(
                func=split_data,
                inputs=["cleaned_data", "params:data_processing"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["data_processing", "split"],
            ),
            node(
                func=build_preprocessing_pipeline,
                inputs=["X_train", "X_test", "params:data_processing"],
                outputs=["X_train_preprocessed", "X_test_preprocessed", "preprocessor"],
                name="build_preprocessing_pipeline_node",
                tags=["data_processing", "preprocessing_v1"],
            ),
            node(
                func=build_preprocessing_pipeline_v2,
                inputs=["X_train", "X_test", "params:data_processing"],
                outputs=["X_train_preprocessed_v2", "X_test_preprocessed_v2", "preprocessor_v2"],
                name="build_preprocessing_pipeline_v2_node",
                tags=["data_processing", "preprocessing_v2"],
            ),
        ]
    )
