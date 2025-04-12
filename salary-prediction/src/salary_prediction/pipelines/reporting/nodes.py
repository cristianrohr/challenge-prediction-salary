import os
import json
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots  # Import make_subplots

def plot_metrics(metrics: dict, output_filepath: str):
    """
    Generates individual plots for MSE, RMSE, and R² scores with their confidence intervals.
    
    Args: 
        metrics:  json file with metrics
        output_filepath: path to save the file

    Returns: 
        None
    """
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Extracting the necessary metrics from the dictionary
    mse = metrics["mse"]
    rmse = metrics["rmse"]
    r2_score = metrics["r2_score"]
    
    mse_conf_int = metrics["mse_conf_interval"]
    rmse_conf_int = metrics["rmse_conf_interval"]
    r2_conf_int = metrics["r2_score_conf_interval"]

    model_name = metrics["model_type"]

    # Create the figure with multiple subplots (facets)
    fig = make_subplots(
        rows=1, cols=3,  # 1 row, 3 columns (one for each metric)
        subplot_titles=["MSE", "RMSE", "R²"]
    )

    # MSE plot
    fig.add_trace(
        go.Bar(
            x=["MSE"],
            y=[mse],
            error_y=dict(type="data", array=[mse_conf_int[1] - mse], arrayminus=[mse - mse_conf_int[0]]),
            name="MSE",
            marker=dict(color="skyblue")
        ), row=1, col=1
    )

    # RMSE plot
    fig.add_trace(
        go.Bar(
            x=["RMSE"],
            y=[rmse],
            error_y=dict(type="data", array=[rmse_conf_int[1] - rmse], arrayminus=[rmse - rmse_conf_int[0]]),
            name="RMSE",
            marker=dict(color="lightgreen")
        ), row=1, col=2
    )

    # R² plot
    fig.add_trace(
        go.Bar(
            x=["R²"],
            y=[r2_score],
            error_y=dict(type="data", array=[r2_conf_int[1] - r2_score], arrayminus=[r2_score - r2_conf_int[0]]),
            name="R²",
            marker=dict(color="lightcoral")
        ), row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title=f"{model_name} Evaluation Metrics with 95% Confidence Intervals",
        showlegend=False,
        template="plotly_white",
        height=500,  # Adjust height to make it more compact
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Metric",
        yaxis_title="Score",
    )

    # Save the plot to the specified file path
    fig.write_image(output_filepath)  # Save as PNG
    return fig

