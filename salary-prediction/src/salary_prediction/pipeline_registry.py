"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from salary_prediction.pipelines import data_processing

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    data_processing_pipeline = data_processing.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "__default__": data_processing_pipeline
    }
