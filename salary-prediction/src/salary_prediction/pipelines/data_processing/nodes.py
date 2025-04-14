import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import logging

log = logging.getLogger(__name__)


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


def group_job_titles(title):
    """
    Group job titles into meaningful categories based on seniority and department.
    
    Args:
        title: The job title string
        
    Returns:
        A string representing the job title group
    """
    if pd.isna(title):
        return "Unknown"
        
    title = str(title).lower()
    
    # Extract seniority
    if "junior" in title or "jr" in title:
        seniority = "Junior"
    elif "senior" in title or "sr" in title or "principal" in title:
        seniority = "Senior" 
    elif "director" in title or "chief" in title or "vp" in title or "ceo" in title or "cto" in title:
        seniority = "Executive"
    else:
        seniority = "Mid-level"
        
    # Extract department/function
    if any(word in title for word in ["engineer", "developer", "software", "web", "technical", "network", "it ", "data scientist"]):
        department = "Technical"
    elif any(word in title for word in ["sales", "account", "business development"]):
        department = "Sales"
    elif any(word in title for word in ["marketing", "social media", "content", "copywriter"]):
        department = "Marketing"
    elif any(word in title for word in ["hr", "human resources", "recruit"]):
        department = "HR"
    elif any(word in title for word in ["finance", "financial", "accountant"]):
        department = "Finance"
    elif any(word in title for word in ["data", "analyst", "research", "scientist"]):
        department = "Analytics"
    elif any(word in title for word in ["product", "ux", "design"]):
        department = "Product"
    elif any(word in title for word in ["operations", "supply chain"]):
        department = "Operations"
    elif any(word in title for word in ["customer", "service", "support"]):
        department = "Customer Service"
    else:
        department = "Other"
        
    return f"{seniority}_{department}"



def build_preprocessing_pipeline_v2(X_train: pd.DataFrame, X_test: pd.DataFrame, parameters: dict) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Builds a preprocessing pipeline and applies it to training and test sets,
    handling job titles with custom grouping approach (excluding original job title).
    """
    ordinal_col_name = parameters.get("ordinal_column", "Education Level")
    education_order = parameters.get("education_level_order", ["Bachelor's", "Master's", "PhD"])
    cardinality_threshold = parameters.get("cardinality_threshold", 10)
    job_title_column = parameters.get("job_title_column", "Job Title")

    log.info(f"Using '{ordinal_col_name}' as the ordinal column with order: {education_order}")
    log.info(f"Using cardinality threshold: {cardinality_threshold}")
    log.info(f"Using '{job_title_column}' for hierarchical grouping (original column will be removed)")

    # Apply job title grouping transformation first
    if job_title_column in X_train.columns:
        log.info(f"Applying hierarchical grouping to job titles")
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # Create new feature with grouped job titles
        X_train['JobTitleGroup'] = X_train[job_title_column].apply(group_job_titles)
        X_test['JobTitleGroup'] = X_test[job_title_column].apply(group_job_titles)
        
    else:
        log.warning(f"Job title column '{job_title_column}' not found in data")

    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    all_categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    explicit_ordinal_cols = []
    job_title_group_cols = []
    other_categorical_cols = []
    
    # Separate columns into different categories
    if ordinal_col_name in all_categorical_cols:
        explicit_ordinal_cols = [ordinal_col_name]
        
    if 'JobTitleGroup' in all_categorical_cols:
        job_title_group_cols = ['JobTitleGroup']
        
    # All remaining categorical columns (excluding job title)
    other_categorical_cols = [col for col in all_categorical_cols 
                             if col != ordinal_col_name 
                             and col != job_title_column 
                             and col != 'JobTitleGroup']

    # Split remaining categoricals by cardinality
    low_cardinality_cols = [col for col in other_categorical_cols if X_train[col].nunique() < cardinality_threshold]
    high_cardinality_cols = [col for col in other_categorical_cols if col not in low_cardinality_cols]

    log.info(f"Numerical columns: {numerical_cols}")
    log.info(f"Education level ordinal column: {explicit_ordinal_cols}")
    log.info(f"Job title group column (reduced cardinality): {job_title_group_cols}")
    log.info(f"Low cardinality categorical columns: {low_cardinality_cols}")
    log.info(f"Other high cardinality categorical columns: {high_cardinality_cols}")

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline for Education Level
    edu_ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[education_order], handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Pipeline for JobTitleGroup (treat as low cardinality categorical with OneHotEncoder)
    job_group_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    # Pipeline for other low cardinality categorical columns
    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    # Pipeline for other high cardinality categorical columns
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal_fallback", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # --- Column Transformer ---
    transformers = []
    if numerical_cols:
        transformers.append(("num", num_pipeline, numerical_cols))
    if explicit_ordinal_cols:
        transformers.append(("edu_ordinal", edu_ordinal_pipeline, explicit_ordinal_cols))
    if job_title_group_cols:  # Add the grouped job titles transformer
        transformers.append(("job_group", job_group_pipeline, job_title_group_cols))
    # Removed the job_title_cols transformer
    if low_cardinality_cols:
        transformers.append(("low_cat_ohe", low_card_pipeline, low_cardinality_cols))
    if high_cardinality_cols:
        transformers.append(("high_cat_ord", high_card_pipeline, high_cardinality_cols))

    if not transformers:
        log.error("No columns identified for preprocessing. Check input data.")
        raise ValueError("No columns found to preprocess.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    log.info("Fitting preprocessing pipeline on training data...")
    X_train_transformed_np = preprocessor.fit_transform(X_train)
    log.info("Transforming test data...")
    X_test_transformed_np = preprocessor.transform(X_test)

    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.split('__')[-1] for name in feature_names]
    except AttributeError:
        log.warning("Falling back to manual feature name reconstruction.")
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
             if name == 'remainder': continue
             if trans == 'drop': continue

             if name in ['num', 'edu_ordinal']:  # Removed 'job_title_orig'
                 feature_names.extend(cols)
             elif name in ['job_group', 'low_cat_ohe']:
                 ohe = trans.named_steps['onehot']
                 feature_names.extend(ohe.get_feature_names_out(cols))
             elif name == 'high_cat_ord':
                 feature_names.extend(cols)

    X_train_preprocessed = pd.DataFrame(X_train_transformed_np, columns=feature_names, index=X_train.index)
    X_test_preprocessed = pd.DataFrame(X_test_transformed_np, columns=feature_names, index=X_test.index)

    log.info(f"Preprocessing complete. Transformed train shape: {X_train_preprocessed.shape}, Transformed test shape: {X_test_preprocessed.shape}")

    return X_train_preprocessed, X_test_preprocessed, preprocessor