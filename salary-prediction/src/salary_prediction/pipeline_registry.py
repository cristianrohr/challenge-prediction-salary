# src/your_project_name/pipeline_registry.py

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline # Ensure pipeline is imported if used for combining

# Import individual pipeline creation functions
from .pipelines.data_processing.pipeline import (
    create_common_preprocessing_pipeline,
    create_preprocessing_v1_pipeline,
    create_preprocessing_v2_pipeline,
)
# Assuming create_ds_pipeline creates the *full* ds pipeline object
from .pipelines.data_science.pipeline import create_pipeline as create_ds_pipeline
from .pipelines.reporting.pipeline import create_pipeline as create_reporting_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A dictionary mapping pipeline names to Pipeline objects.
    """
    # Create instances of the pipeline components
    common_pp_pipeline = create_common_preprocessing_pipeline()
    pp_v1_pipeline = create_preprocessing_v1_pipeline()
    pp_v2_pipeline = create_preprocessing_v2_pipeline()
    # This holds the full pipeline object with all ds nodes/tags
    ds_pipeline_full_object = create_ds_pipeline()
    reporting_pipeline_object = create_reporting_pipeline()

    # --- Define Named Pipelines for Specific Runs ---
    pipelines = {} # Initialize the dictionary

    # Specific Preprocessing pipelines
    pipelines["pp_common"] = common_pp_pipeline
    pipelines["pp_v1"] = pp_v1_pipeline
    pipelines["pp_v2"] = pp_v2_pipeline

    # Combine all preprocessing steps under this name
    pipelines["data_processing"] = (
        common_pp_pipeline + pp_v1_pipeline + pp_v2_pipeline
    )

    # Specific Data Science pipelines
    pipelines["ds_full"] = ds_pipeline_full_object # All DS nodes
    pipelines["ds_v1"] = ds_pipeline_full_object.only_nodes_with_tags("uses_pp_v1")
    pipelines["ds_v2"] = ds_pipeline_full_object.only_nodes_with_tags("uses_pp_v2")

    # Assign the full data science pipeline object to this name
    pipelines["data_science"] = ds_pipeline_full_object

    # Reporting pipeline
    pipelines["reporting"] = reporting_pipeline_object

    # --- Define Composite Pipelines for End-to-End Runs ---
    pipelines["full_v1"] = pipeline( # Use pipeline() to combine if needed
        pipelines["data_processing"].only_nodes_with_tags("pp_common", "pp_v1") # Run common + v1 pp
        + pipelines["ds_v1"] # Run v1 ds nodes
        + pipelines["reporting"]
    )

    pipelines["full_v2"] = (
         common_pp_pipeline
         + pp_v2_pipeline
         + pipelines["ds_v2"] # Use the filtered v2 DS pipeline
         + pipelines["reporting"]
    )

    pipelines["full_both"] = (
        pipelines["data_processing"] # Use the combined DP pipeline
        + pipelines["data_science"]  # Use the combined DS pipeline
        + pipelines["reporting"]
    )

    # --- Define the Default Pipeline ---
    pipelines["__default__"] = pipelines["full_both"]

    return pipelines