import os
import json
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots  # Import make_subplots
from typing import List, Dict
import pandas as pd

def select_best_model(
    all_metrics: list[dict],  # list of all model evaluation dicts
    selection_metric: str = "rmse",  # metric to select best model
    ascending: bool = True  # True for metrics like RMSE/MSE, False for R²
) -> dict:
    """
    Selects the best model based on a given metric.

    Args:
        all_metrics: List of model metrics dictionaries.
        selection_metric: Metric to base the comparison on.
        ascending: Whether lower is better (True for MSE/RMSE).

    Returns:
        A dictionary with the best model's name and metrics.
    """
    best = sorted(all_metrics, key=lambda m: m[selection_metric], reverse=not ascending)[0]
    return {
        "best_model_name": best["model_type"],
        "best_model_metrics": best
    }

def generate_comparison_report(all_metrics: list[dict], output_path: str):
    """
    Saves all model evaluation results to a JSON report.

    Args:
        all_metrics: List of evaluation dictionaries.
        output_path: Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

def generate_sorted_metrics_csv(
    all_metrics: list[dict],
    sort_by: str = "rmse",
    ascending: bool = True
):
    """
    Generates a CSV comparing model metrics, sorted by a chosen metric.

    Args:
        all_metrics: List of model metrics dictionaries.
        sort_by: Metric to sort the CSV by.
        ascending: If True, sort in ascending order (use for RMSE/MSE).
    """

    # Normalize list of dicts into a DataFrame
    df = pd.json_normalize(all_metrics)

    # Drop CI columns if they exist (or you can keep them)
    ci_cols = [col for col in df.columns if "conf_interval" in col]
    df = df.drop(columns=ci_cols, errors="ignore")

    # Sort and save
    df_sorted = df.sort_values(by=sort_by, ascending=ascending)
    return df_sorted

def load_all_metrics_from_folder(metrics_folder: str) -> List[Dict]:
    """
    Loads all JSON files from a folder and returns them as a list of dicts.

    Args:
        metrics_folder: Path to the folder containing metric JSON files.

    Returns:
        List of model metrics as dictionaries.
    """
    all_metrics = []

    for file_name in os.listdir(metrics_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(metrics_folder, file_name)
            with open(file_path, "r") as f:
                metrics = json.load(f)
                all_metrics.append(metrics)

    return all_metrics

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

