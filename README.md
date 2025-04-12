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
- Visualize the relationships between features and the target variable (4). Included in the EDA.

## Data

The dataset contains three files:
- `salary.csv`: Contains the target variable (Salary)
- `people.csv`: Contains features about individuals
- `descriptions.csv`: Contains textual descriptions

Dataset involves both numerical and categorical variables

## Project Structure

Kedro was used to structure the code.

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



## Methodology

## Results

## References
