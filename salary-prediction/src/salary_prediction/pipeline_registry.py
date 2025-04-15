"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Import individual pipeline creation functions
from .pipelines.data_processing.pipeline import (
    create_pipeline as create_dp_pipeline,
    create_common_preprocessing_pipeline,
    create_preprocessing_v1_pipeline,
    create_preprocessing_v2_pipeline,
)
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
    ds_pipeline = create_ds_pipeline()
    reporting_pipeline = create_reporting_pipeline()

    # --- Define Named Pipelines for Specific Runs ---

    # Preprocessing pipelines
    pipelines = {
        "pp_common": common_pp_pipeline,
        "pp_v1": pp_v1_pipeline,
        "pp_v2": pp_v2_pipeline,
    }

    # Data science pipelines (potentially filtered)
    # Full DS pipeline (contains nodes tagged for v1 and v2)
    pipelines["ds_full"] = ds_pipeline
    # Filtered DS pipelines
    pipelines["ds_v1"] = ds_pipeline.only_nodes_with_tags("uses_pp_v1")
    pipelines["ds_v2"] = ds_pipeline.only_nodes_with_tags("uses_pp_v2")

    # Reporting pipeline
    pipelines["reporting"] = reporting_pipeline

    # --- Define Composite Pipelines for End-to-End Runs ---

    # Run only V1 preprocessing and corresponding DS models
    pipelines["full_v1"] = (
        common_pp_pipeline
        + pp_v1_pipeline
        + pipelines["ds_v1"] # Use the filtered v1 DS pipeline
        + reporting_pipeline # Reporting will process available metrics
    )

    # Run only V2 preprocessing and corresponding DS models
    pipelines["full_v2"] = (
        common_pp_pipeline
        + pp_v2_pipeline
        + pipelines["ds_v2"] # Use the filtered v2 DS pipeline
        + reporting_pipeline # Reporting will process available metrics
    )

    # Run common steps, both preprocessing versions, and all DS models
    pipelines["full_both"] = (
        common_pp_pipeline
        + pp_v1_pipeline
        + pp_v2_pipeline
        + pipelines["ds_full"] # Use the full DS pipeline
        + reporting_pipeline # Reporting will process all metrics
    )

    # --- Define the Default Pipeline ---
    pipelines["__default__"] = pipelines["full_both"]

    return pipelines