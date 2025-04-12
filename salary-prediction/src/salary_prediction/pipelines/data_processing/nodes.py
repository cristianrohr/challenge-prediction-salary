import pandas as pd
from sklearn.model_selection import train_test_split

def merge_datasets(salary_df: pd.DataFrame, people_df: pd.DataFrame, descriptions_df: pd.DataFrame):
    """
    Merges the datasets into a single one.

    Args:
        salary_df: Dataframe with the target variable.
        people_df: Dataframe with the features.
        descriptions_df: Dataframe with the descriptions.
    Returns:
        Merged dataset.
    """
    return people_df.merge(salary_df, on="id", how = "inner")

def clean_data(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Clean the data.

    Args:
        data Raw data.
    Returns:
        Cleaned dataframe.
    """
    df = data.copy()

    df = df.drop_duplicates()

    # Drop the ID column and rows were salary is NA or NULL
    df = df.drop(columns=["id"])
    df = df.dropna(subset=[parameters["target_variable"]])

    return df

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """
    Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_processing.yml.
    Returns:
        Split data.
    """
    
    # Extract the target variable
    X = data.drop(parameters["target_variable"], axis=1)
    y = data[parameters["target_variable"]]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=parameters["test_size"], 
        random_state=parameters["random_state"]
    )
    
    return X_train, X_test, y_train, y_test
    