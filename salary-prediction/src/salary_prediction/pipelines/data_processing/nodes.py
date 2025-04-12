import pandas as pd

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

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
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
    df = df.dropna(subset=["Salary"])

    return df
