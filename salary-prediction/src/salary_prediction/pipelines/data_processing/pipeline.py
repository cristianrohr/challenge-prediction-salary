from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, merge_datasets, split_data, build_preprocessing_pipeline, build_preprocessing_pipeline_v2


# src/your_project_name/pipelines/preprocessing/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_data,
    merge_datasets,
    split_data,
    build_preprocessing_pipeline,
    build_preprocessing_pipeline_v2,
)

def create_common_preprocessing_pipeline(**kwargs) -> Pipeline:
    """Pipeline for steps common to all preprocessing versions."""
    return pipeline(
        [
            node(
                func=merge_datasets,
                inputs=["salary_raw", "people_raw"],
                outputs="merged_data",
                name="merge_datasets_node",
                tags=["pp_common", "data_processing", "merge"],
            ),
            node(
                func=clean_data,
                inputs=["merged_data", "params:data_processing"],
                outputs="cleaned_data",
                name="clean_data_node",
                tags=["pp_common", "data_processing", "clean"],
            ),
            node(
                func=split_data,
                inputs=["cleaned_data", "params:data_processing"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["pp_common", "data_processing", "split"],
            ),
        ]
    )

def create_preprocessing_v1_pipeline(**kwargs) -> Pipeline:
    """Pipeline for preprocessing version 1."""
    return pipeline(
        [
            node(
                func=build_preprocessing_pipeline,
                inputs=["X_train", "X_test", "params:data_processing"],
                outputs=[
                    "X_train_preprocessed",
                    "X_test_preprocessed",
                    "preprocessor",
                ],
                name="build_preprocessing_pipeline_node",
                tags=["pp_v1", "data_processing", "preprocessing"], # Tag specific to v1
            ),
        ]
    )

def create_preprocessing_v2_pipeline(**kwargs) -> Pipeline:
    """Pipeline for preprocessing version 2."""
    return pipeline(
        [
            node(
                func=build_preprocessing_pipeline_v2,
                inputs=["X_train", "X_test", "params:data_processing"],
                outputs=[
                    "X_train_preprocessed_v2",
                    "X_test_preprocessed_v2",
                    "preprocessor_v2",
                ],
                name="build_preprocessing_pipeline_v2_node",
                tags=["pp_v2", "data_processing", "preprocessing"], # Tag specific to v2
            ),
        ]
    )

# Keep a function that might create the combined pipeline if needed,
# but we'll primarily use the components in the registry.
def create_pipeline(**kwargs) -> Pipeline:
    """
    DEPRECATED or used for full runs only.
    Prefer assembling pipelines in pipeline_registry.py
    based on parameters.
    """
    # This function is less useful now, registry handles composition.
    # You could potentially have it return pp_common + pp_v1 + pp_v2
    # return (
    #     create_common_preprocessing_pipeline()
    #     + create_preprocessing_v1_pipeline()
    #     + create_preprocessing_v2_pipeline()
    # )
    # Or return just the common part, assuming v1/v2 are added later
    return create_common_preprocessing_pipeline()