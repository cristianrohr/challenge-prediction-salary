# challenge-salary-prediction

## Overview

This project develops a machine learning model to predict salaries based on features such as age, gender, education level, job title, years of experience, and job descriptions. This is a regression task.

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

#### Optional Features ( just the ones we have implemented):
- Validate the model using cross-validation techniques. (3)
- Visualize the relationships between features and the target variable (4). Included in the EDA. (4)
- Use an interpretation tool to explain the contribution of each feature - e.g. SHAP (5)
- Use a more advanced model such as Random Forest, or Neural Networks. (6)
- Lock your dependencies - e.g., using pipenv, uv or pdm. (8)


## Data

The dataset contains three files:
- `salary.csv`: Contains the target variable (Salary)
- `people.csv`: Contains features about individuals
- `descriptions.csv`: Contains textual descriptions ( NOT USED!!)

Dataset involves both numerical and categorical variables

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
`kedro run`

Run V1 Only:
Set preprocessing_version_to_run: "v1" in parameters_data_processing.yml (optional, for documentation/clarity).

Execute: `kedro run --pipeline=full_v1`
This runs pp_common, pp_v1, then only the data science nodes tagged uses_pp_v1, and finally reporting.

Run V2 Only:
Set preprocessing_version_to_run: "v2" in parameters_data_processing.yml (optional).
Execute: `kedro run --pipeline=full_v2`
This runs pp_common, pp_v2, then only the data science nodes tagged uses_pp_v2, and finally reporting.

Run Both (Default Behavior):
Set preprocessing_version_to_run: "both" in parameters_data_processing.yml.
Execute: `kedro run` (runs __default__, which we mapped to full_both)
Or explicitly: `kedro run --pipeline=full_both`

This runs pp_common, pp_v1, pp_v2, all data science nodes (ds_full), and finally reporting.

## Methodology

Kedro was used to structure de code.

### EDA

Exploratory data analsysis. First a fast EDA was performed in the notebooks/01_exploratory_data_analysis.ipynb

### Baseline model

As suggested a DummyRegressor was implemented as the baseline model

### Metrics

In order to evaluate the models some regression metrics were selected

 - MSE (Mean Squared Error): Average squared difference between the predicted and real salaries.
 - RMSE (Root Mean Squared Error): Square root of MSE.
 - R2 (R-squared): Proportion of variace in salary explained by the model.

## Results
1. Baseline model 

Test:
{
  "mse": 2403174600.7218146,
  "rmse": 49022.18478119692,
  "r2_score": -0.0023325074934903434,
  "model_type": "DummyRegressor"
}

Train with CV CI's:
{
  "mse": 2327671491.480214,
  "rmse": 48158.40404560465,
  "r2_score": -0.03205210959908009,
  "mse_conf_interval": [
    1941110039.546183,
    2714232943.414245
  ],
  "rmse_conf_interval": [
    44125.477453554595,
    52191.330637654704
  ],
  "r2_score_conf_interval": [
    -0.07021907803626393,
    0.006114858838103733
  ],
  "model_type": "DummyRegressor"
}

2. Linear regression
{
  "mse": 245295119.11827332,
  "rmse": 15550.531658798393,
  "r2_score": 0.8913156707773544,
  "mse_conf_interval": [
    165826615.19786328,
    324763623.03868335
  ],
  "rmse_conf_interval": [
    12962.293461212596,
    18138.769856384188
  ],
  "r2_score_conf_interval": [
    0.8620612942374825,
    0.9205700473172264
  ],
  "model_type": "LinearRegression"
}

3. RandomForestRegressor
{
  "mse": 237083576.69463697,
  "rmse": 15368.801466421832,
  "r2_score": 0.8934237853681604,
  "mse_conf_interval": [
    196318485.91431722,
    277848667.4749567
  ],
  "rmse_conf_interval": [
    14063.932685338918,
    16673.670247504746
  ],
  "r2_score_conf_interval": [
    0.8703175955357493,
    0.9165299752005714
  ],
  "model_type": "RandomForestRegressor"
}

4. RandomForestRegressor + SHAP feature selection
{
  "mse": 244666229.32381254,
  "rmse": 15614.111247418661,
  "r2_score": 0.889959794756901,
  "mse_conf_interval": [
    203983160.220127,
    285349298.4274981
  ],
  "rmse_conf_interval": [
    14322.423105951186,
    16905.799388886135
  ],
  "r2_score_conf_interval": [
    0.8659463970975693,
    0.9139731924162328
  ],
  "model_type": "RandomForestRegressor"
}

5. Lasso
{
  "mse": 245239583.6016566,
  "rmse": 15548.702211450833,
  "r2_score": 0.8913466435190515,
  "mse_conf_interval": [
    165741747.2817182,
    324737419.921595
  ],
  "rmse_conf_interval": [
    12959.958157532057,
    18137.44626536961
  ],
  "r2_score_conf_interval": [
    0.8621360968999312,
    0.9205571901381717
  ],
  "model_type": "Lasso"
}

## References
