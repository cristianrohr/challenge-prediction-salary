import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

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

    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, parameters: dict) -> Tuple:
    """
    Builds a preprocessing pipeline and applies it to training and test sets.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        Transformed X_train, X_test and the fitted preprocessor pipeline.
    """

    # Use X_train to detect column types
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Inspect cardinality
    low_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() < parameters["cardinality_threshold"]]
    high_cardinality_cols = [col for col in categorical_cols if col not in low_cardinality_cols]
    
    # Build preprocessing pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("low_cat", low_card_pipeline, low_cardinality_cols),
        ("high_cat", high_card_pipeline, high_cardinality_cols),
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Keep names for the features, so we can easily perfom analysis downstream
    feature_names = []

    # For numeric columns (names are preserved)
    feature_names.extend(numerical_cols)

    # For low cardinality categoricals (OneHotEncoder expands these)
    ohe = preprocessor.named_transformers_["low_cat"].named_steps["onehot"]
    low_card_features = ohe.get_feature_names_out(low_cardinality_cols)
    feature_names.extend(low_card_features)

    # For high cardinality categoricals (OrdinalEncoder gives one column per feature)
    feature_names.extend(high_cardinality_cols)

    # Build DataFrames with column names
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

    return X_train_transformed, X_test_transformed, preprocessor