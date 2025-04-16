# challenge-salary-prediction

## Overview

This project was built as a technical machine learning challenge to predict individual salaries based on structured data, applying best practices in reproducible data science using the Kedro framework.

We developed a fully modular and documented ML pipeline that includes:
- Data ingestion and preprocessing with two alternative versions to compare feature engineering strategies.
- Model training and evaluation, ranging from simple baselines to advanced tree-based regressors.
- Confidence interval estimation to assess the statistical reliability of the predictions.
- Feature importance analysis using SHAP to enhance model interpretability.

Dataset Overview
- The dataset includes:
- Demographics (age, gender, education)
- Professional features (job title, years of experience)
- Text descriptions (excluded in this implementation)

The goal is to predict salary (regression task) using only structured features.

### Mandatory Features:

The challenge have some mandatory features:
- Load and preprocess the dataset using Pandas / Polars or similar tools.
- Perform feature transformations on the input dataset to derive useful numerical features from the model.
- Develop a predictive model using a machine learning library such as Scikit-learn or TensorFlow/Keras.
- Evaluate the model's performance using appropriate metrics. Explaining why that metric was useful.
- Compare the model results against a sensible baseline – e.g. a DummyRegressor
- All the metrics should be reported as intervals, not as single estimates. See references on how to calculate confidence intervals for ML.
- Structure the code in modules . See the reference to know what is expected.
- Present the final result in a Jupyter Notebook which should:
  - Have a Table of Contents at the top.
  - Contain explanatory text in markdown, properly formatted.
  - Have no big blocks of code, instead it should import the functionality from the modules and only have initial configuration.
  - Include illustrations or images showing the results.


## Data

The dataset contains three files:
- `salary.csv`: Contains the target variable (Salary)
- `people.csv`: Contains features about individuals
- `descriptions.csv`: Contains textual descriptions ( NOT USED!!)

Dataset involves both numerical and categorical variables

## Technical Stack
- Python 3 and virtual environments
- Kedro for modular pipelines
- Scikit-learn, XGBoost, SHAP for modeling and interpretation
- Jupyter Notebooks for EDA and result presentation

## Project Structure

Kedro was used to structure the code.

The project have three different pipelines

- data_processing: responsible for loading and preprocessing the data
  - pp_common: common preprocessing steps
  - pp_v1: preprocessing steps for version 1
  - pp_v2: preprocessing steps for version 2 ( a more advanced version of the preprocessing )
- data_science: responsible for training the model
- reporting: responsible for reporting the results

## Installation & Setup

1. Clone the repository
`git clone https://github.com/cristianrohr/challenge-prediction-salary.git`
2. Create a virtual environment
`python3 -m venv venv`
3. Activate the virtual environment
`source venv/bin/activate`
4. Install dependencies
`pip install -r requirements.txt`
5. Run the EDA notebook
`jupyter notebook notebooks/01_EDA.ipynb`
6. Run the Kedro pipeline
`cd salary_prediction; kedro run`

Run V1 Only:
Execute: `kedro run --pipeline=full_v1`
This runs pp_common, pp_v1, then only the data science nodes tagged uses_pp_v1, and finally reporting.

Run V2 Only:
Execute: `kedro run --pipeline=full_v2`
This runs pp_common, pp_v2, then only the data science nodes tagged uses_pp_v2, and finally reporting.

Run Both (Default Behavior):
Execute: `kedro run` (runs __default__, which we mapped to full_both)
Or explicitly: `kedro run --pipeline=full_both`

This runs pp_common, pp_v1, pp_v2, all data science nodes (ds_full), and finally reporting.

## Results
After cross-validation and model selection, the final evaluation was performed on the held-out test set using the best model: XGBRegressor, trained with pp_v2 preprocessing and tuned via Optuna. The model achieved a Root Mean Squared Error (RMSE) of 14,468, a Mean Squared Error (MSE) of ~2.09×10⁸, and an R² score of 0.913 on the test data. 

## Full Report
- The complete modeling report is available in `salary_prediction/notebooks/02_Model_Report.ipynb`
- The EDA report is available in `salary_prediction/notebooks/01_EDA.ipynb`