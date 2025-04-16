import os
import json
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots  # Import make_subplots
from typing import List, Dict
import pandas as pd
from scipy.stats import ttest_rel
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
import re

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

def load_all_metrics_from_folder(metrics_folder: str, _dependency_marker=None) -> List[Dict]:

    """
    Loads all JSON files from a folder and returns them as a list of dicts.

    Args:
        metrics_folder: Path to the folder containing metric JSON files.
        _dependency_marker: Marker to ensure this node is called directly and not imported.

    Returns:
        List of model metrics as dictionaries.
    """
    all_metrics = []

    for file_name in os.listdir(metrics_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(metrics_folder, file_name)
            with open(file_path, "r") as f:
                metrics = json.load(f)
                metrics["file_name"] = file_name 
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

def select_best_model_by_multiple_metrics(
    all_metrics: List[Dict], 
    metrics: List[str],
    ascending_flags: List[bool]
) -> Dict:
    """
    Selects the best model based on a given metric.

    Args:
        all_metrics: List of dictionaries containing model metrics.
        metrics: List of metrics to consider for selection.
        ascending_flags: List of booleans indicating whether to use ascending order for the metrics.

    Returns:
        Dictionary containing the name of the best model and its metrics.
    """

    df = pd.DataFrame(all_metrics)
    for metric, ascending in zip(metrics, ascending_flags):
        df[f"{metric}_rank"] = df[metric].rank(ascending=ascending)
    df["avg_rank"] = df[[f"{m}_rank" for m in metrics]].mean(axis=1)
    best_idx = df["avg_rank"].idxmin()
    best_model = df.iloc[best_idx].to_dict()
    return {
        "best_model_name": best_model["model_type"],
        "best_model_metrics": best_model
    }

def get_top_n_models(
    all_metrics: List[Dict], sort_by: str, top_n: int = 3, ascending: bool = True
) -> List[Dict]:
    """
    Returns the top N models based on a given metric, sorted in ascending or descending order.

    Args:
        all_metrics: List of dictionaries containing model metrics.
        sort_by: Metric to use for sorting.
        top_n: Number of top models to return.
        ascending: Whether to use ascending order (True for metrics like RMSE/MSE, False for R²).

    Returns:
        List of dictionaries containing top N models.
    """
    return sorted(all_metrics, key=lambda m: m[sort_by], reverse=not ascending)[:top_n]

def reject_models_with_wide_ci(
    all_metrics: List[Dict], max_ci_width: float = 10000
) -> List[Dict]:
    """
    Rejects models with confidence intervals wider than a given threshold.

    Args:
        all_metrics: List of dictionaries containing model metrics.
        max_ci_width: Maximum allowed width of confidence interval.

    Returns:
        List of dictionaries containing rejected models.
    """
    return [
        m for m in all_metrics
        if (m["rmse_conf_interval"][1] - m["rmse_conf_interval"][0]) < max_ci_width
    ]

def save_top_n_metrics(
    top_models: List[Dict], output_path: str
):
    """
    Saves a list of top models to a JSON file.

    Args:
        top_models: List of dictionaries containing model metrics.
        output_path: Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(top_models, f, indent=4)

def compare_models_statistically(
    metrics_list: List[Dict], metric: str = "rmse"
) -> pd.DataFrame:
    """
    Compares the performance of models statistically.

    Args:
        metrics_list: List of dictionaries containing model metrics.
        metric: Metric to compare. Default is "rmse".

    Returns:
        DataFrame with comparison results.
    """
    comparisons = []
    for i, m1 in enumerate(metrics_list):
        for j, m2 in enumerate(metrics_list):
            if i < j and metric + "_folds" in m1 and metric + "_folds" in m2:
                t_stat, p_value = ttest_rel(m1[metric + "_folds"], m2[metric + "_folds"])
                comparisons.append({
                    "model_1": m1["model_type"],
                    "model_2": m2["model_type"],
                    "p_value": p_value
                })
    return pd.DataFrame(comparisons)

def plot_model_comparison_scatter(
    all_metrics: List[Dict], output_path: str
):
    """
    Plots a scatter plot comparing RMSE and R² scores.

    Args:
        all_metrics: List of dictionaries containing model metrics.
        output_path: Path to save the scatter plot.
    """

    # filter models with neccesary fields
    required_keys = ["rmse_conf_interval", "r2_score_conf_interval", "rmse", "r2_score"]
    filtered = [m for m in all_metrics if all(k in m for k in required_keys)]

    if not filtered:
        raise ValueError("No models contain all required confidence interval keys.")

    df = pd.json_normalize(all_metrics)

    def extract(val, index):
        return val[index] if isinstance(val, (list, tuple)) and len(val) == 2 else None

    df["rmse_ci_high"] = df["rmse_conf_interval"].apply(lambda x: extract(x, 1))
    df["rmse_ci_low"] = df["rmse_conf_interval"].apply(lambda x: extract(x, 0))
    df["r2_ci_high"] = df["r2_score_conf_interval"].apply(lambda x: extract(x, 1))
    df["r2_ci_low"] = df["r2_score_conf_interval"].apply(lambda x: extract(x, 0))

    df["rmse_err"] = df["rmse_ci_high"] - df["rmse"]
    df["rmse_err_minus"] = df["rmse"] - df["rmse_ci_low"]
    df["r2_err"] = df["r2_ci_high"] - df["r2_score"]
    df["r2_err_minus"] = df["r2_score"] - df["r2_ci_low"]

    fig = px.scatter(
        df,
        x="rmse",
        y="r2_score",
        text="model_type",
        error_x="rmse_err",
        error_x_minus="rmse_err_minus",
        error_y="r2_err",
        error_y_minus="r2_err_minus",
        title="Model Comparison (RMSE vs R²)",
        labels={"rmse": "RMSE ↓", "r2_score": "R² ↑"},
        width=850,
        height=650,
    )

    fig.update_traces(
        textposition="top right",
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode="markers+text")
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
    )

    fig.write_image(output_path)
    return fig

def generate_detailed_metrics_table(
    all_metrics: List[Dict],
    include_conf_intervals: bool = True,
    include_fold_scores: bool = False,
    output_markdown: str = None,
    output_html: str = None
) -> pd.DataFrame:
    """
    Generates a detailed comparison table of model metrics.

    Args:
        all_metrics: List of model metrics dictionaries.
        include_conf_intervals: Whether to include confidence intervals.
        include_fold_scores: Whether to include cross-validation fold scores.
        output_markdown: Optional path to save markdown version.
        output_html: Optional path to save HTML version.

    Returns:
        Formatted pandas DataFrame.
    """
    rows = []
    for m in all_metrics:
        file_name = m.get("file_name", "")
        row = {
            "Model": m["model_type"],
            "RMSE": m.get("rmse"),
            "MSE": m.get("mse"),
            "R²": m.get("r2_score"),
            "Preprocessing": None,
            "Used SHAP": "shap" in file_name.lower(),
            "Used Optuna": "optimized" in file_name.lower(),
        }

        # Extract preprocessing version from filename
        pp_match = re.search(r"pp_v\d+", file_name.lower())
        row["Preprocessing"] = pp_match.group(0) if pp_match else None

        if include_conf_intervals:
            row["RMSE CI"] = (
                f"[{m['rmse_conf_interval'][0]:.2f}, {m['rmse_conf_interval'][1]:.2f}]"
                if "rmse_conf_interval" in m else None
            )
            row["MSE CI"] = (
                f"[{m['mse_conf_interval'][0]:.2f}, {m['mse_conf_interval'][1]:.2f}]"
                if "mse_conf_interval" in m else None
            )
            row["R² CI"] = (
                f"[{m['r2_score_conf_interval'][0]:.2f}, {m['r2_score_conf_interval'][1]:.2f}]"
                if "r2_score_conf_interval" in m else None
            )

        if include_fold_scores:
            row["RMSE folds"] = (
                ", ".join([f"{v:.2f}" for v in m["rmse_folds"]]) if "rmse_folds" in m else None
            )
            row["R² folds"] = (
                ", ".join([f"{v:.2f}" for v in m["r2_score_folds"]]) if "r2_score_folds" in m else None
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Export
    if output_markdown:
        os.makedirs(os.path.dirname(output_markdown), exist_ok=True)
        with open(output_markdown, "w") as f:
            f.write(df.to_markdown(index=False))

    if output_html:
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        df.to_html(output_html, index=False)

    return df
