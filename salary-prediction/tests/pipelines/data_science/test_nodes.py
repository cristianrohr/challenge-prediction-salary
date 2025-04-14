import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline # Alias to avoid clash with Kedro Pipeline

# Import the node functions we want to test
# Assuming these exist in your project structure
# from salary_prediction.pipelines.data_processing.nodes import (
#     merge_datasets,
#     clean_data,
#     split_data,
#     build_preprocessing_pipeline,
# )

# --- Mock Node Functions (if not running within Kedro context) ---
# Replace these with your actual imports if available
# Placeholder implementations for testing purposes if needed:

def merge_datasets(salary_df, people_df, descriptions_df=None):
    """Placeholder merge function"""
    # Assuming descriptions_df is optional or handled inside
    merged_df = pd.merge(people_df, salary_df, on="id", how="inner")
    return merged_df

def clean_data(df, parameters):
    """Placeholder clean function"""
    target = parameters["target_variable"]
    # Drop ID if it exists after merge (it shouldn't based on the merge above)
    if "id" in df.columns:
         df = df.drop(columns=["id"])
    # Drop rows with NaN in the target variable
    df = df.dropna(subset=[target])
    # Drop duplicate rows
    df = df.drop_duplicates()
    return df.reset_index(drop=True)

from sklearn.model_selection import train_test_split
def split_data(df, parameters):
     """Placeholder split function"""
     target = parameters["target_variable"]
     test_size = parameters["test_size"]
     random_state = parameters["random_state"]
     X = df.drop(columns=[target])
     y = df[[target]] # Keep y as DataFrame
     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=test_size, random_state=random_state
     )
     return X_train, X_test, y_train, y_test

# Assume build_preprocessing_pipeline exists and works as intended for the test structure below
# We won't redefine it here, as its internal logic is complex and assumed correct for the test
# --- End Mock Node Functions ---


# --- Updated Fixtures ---

@pytest.fixture
def sample_salary_raw():
    """Fixture for raw salary data matching real structure."""
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Salary": [90000.0, 65000.0, 150000.0, 60000.0, 200000.0, np.nan, 120000.0, 80000.0, 45000.0] # Added NaN example
    })

@pytest.fixture
def sample_people_raw():
    """Fixture for raw people data matching real structure."""
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 6, 7, 8, 9], # ID 5 missing, ID 9 won't match salary
        "Age": [32.0, 28.0, 45.0, 36.0, 52.0, 42.0, 31.0, 26.0, 39.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's", "Master's", "Bachelor's", "Bachelor's", "PhD"],
        "Job Title": ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director", "Product Manager", "Sales Manager", "Marketing Coordinator", "Research Scientist"],
        "Years of Experience": [5.0, 3.0, 15.0, 7.0, 20.0, 12.0, 4.0, 1.0, 10.0]
    })

@pytest.fixture
def sample_descriptions_raw():
    """Fixture for raw descriptions data (keep as is if merge_datasets needs 3 args)."""
    # If your actual merge_datasets doesn't use a third df, you can remove this
    # and update the test_merge_datasets signature.
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "description": ["Desc 0", "Desc 1", "Desc 2", "Desc 3", "Desc 4", "Desc 5", "Desc 6", "Desc 7", "Desc 8", "Desc 9"]
    })

@pytest.fixture
def sample_merged_data(sample_salary_raw, sample_people_raw):
    """Fixture for expected merged data based on updated raw fixtures."""
    # Expected: Inner join on 'id'. IDs 0, 1, 2, 3, 4, 6, 7, 8 should match.
    # Salary for ID 5 is NaN in salary_raw, but ID 5 doesn't exist in people_raw for inner join.
    # ID 9 exists in people_raw but not salary_raw.
    # The placeholder merge_datasets function only merges salary and people.
    # Adjust if your real merge_datasets uses descriptions_df.
    expected = pd.merge(sample_people_raw, sample_salary_raw, on="id", how="inner")
    return expected
    # Manually constructing the expected result for clarity:
    # return pd.DataFrame({
    #     "id": [0, 1, 2, 3, 4, 6, 7, 8],
    #     "Age": [32.0, 28.0, 45.0, 36.0, 52.0, 42.0, 31.0, 26.0],
    #     "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    #     "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's", "Master's", "Bachelor's", "Bachelor's"],
    #     "Job Title": ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director", "Product Manager", "Sales Manager", "Marketing Coordinator"],
    #     "Years of Experience": [5.0, 3.0, 15.0, 7.0, 20.0, 12.0, 4.0, 1.0],
    #     "Salary": [90000.0, 65000.0, 150000.0, 60000.0, 200000.0, 120000.0, 80000.0, 45000.0]
    # })


@pytest.fixture
def sample_data_for_cleaning(sample_merged_data):
    """Fixture for data before cleaning (includes duplicates and potential NaN salary)."""
    # Add a duplicate row (e.g., the first row)
    duplicate_row = sample_merged_data.iloc[[0]]
    data_with_duplicates = pd.concat([sample_merged_data, duplicate_row], ignore_index=True)
    # Add a row with NaN salary if not already present from merge
    # Find an ID that exists in people but not salary OR modify an existing salary to NaN
    # In sample_salary_raw, ID 5 has NaN salary. Let's add ID 5 to people_raw for this test case setup.
    # Or simpler: modify salary of an existing merged row to NaN *before* adding duplicate
    data_with_nan = sample_merged_data.copy()
    # If the merge didn't result in NaN salary, uncomment below to force one
    # data_with_nan.loc[data_with_nan['id'] == 3, 'Salary'] = np.nan # Example: make ID 3 salary NaN

    # Now add duplicate *after* potential NaN introduction
    duplicate_row_for_nan = data_with_nan.iloc[[0]]
    final_data = pd.concat([data_with_nan, duplicate_row_for_nan], ignore_index=True)

    # Let's re-check the merge results based on updated raw data. ID 5 is NaN in salary,
    # but not in people, so it's excluded by inner join. So merged data has no NaNs yet.
    # Let's manually add a NaN for ID 3 to test cleaning
    final_data.loc[final_data['id'] == 3, 'Salary'] = np.nan

    return final_data


@pytest.fixture
def sample_data_processing_params():
    """Fixture for data processing parameters."""
    return {
        "target_variable": "Salary", # Updated target variable name
        "test_size": 0.25,
        "random_state": 42,
        "cardinality_threshold": 5 # Adjust based on new categorical features if needed
    }

@pytest.fixture
def sample_cleaned_data(sample_merged_data):
    """Fixture for expected cleaned data (after dropping id, duplicates, NaN target)."""
    # Start with the result of the merge
    cleaned = sample_merged_data.copy()
    # Manually apply the cleaning steps expected by clean_data node
    # 1. Drop rows with NaN in target 'Salary'
    #    In our setup, we manually made ID 3 NaN Salary in sample_data_for_cleaning
    cleaned = cleaned.dropna(subset=["Salary"]) # This removes ID 3 if we manually set it to NaN
    # Check if merge itself resulted in NaNs
    cleaned = cleaned.dropna(subset=["Salary"])

    # 2. Drop duplicates (original sample_merged_data has no duplicates yet)
    #    The sample_data_for_cleaning adds a duplicate of row 0.
    #    So, we expect the cleaned data to NOT have the duplicate.
    #    The result should look like sample_merged_data minus any rows that had NaN Salary.
    # As ID 5 (NaN Salary) was filtered by inner join, and we didn't manually add NaN here,
    # the result of merge has no NaN Salary. Let's assume the clean function drops ID 3 based
    # on sample_data_for_cleaning fixture where we manually added NaN for ID 3.
    cleaned = cleaned[cleaned['id'] != 3] # Remove row where Salary was manually set to NaN

    # 3. Drop 'id' column (assuming clean_data does this)
    if "id" in cleaned.columns:
         cleaned = cleaned.drop(columns=["id"])

    return cleaned.reset_index(drop=True)


# --- Test Functions (Updated Assertions and Setup for Preprocessing) ---

def test_merge_datasets(sample_salary_raw, sample_people_raw, sample_descriptions_raw, sample_merged_data):
    """Test the merge_datasets function."""
    # Using the placeholder merge_datasets which only uses first 2 args
    # If your actual node uses descriptions_df, adjust the call and expected output
    merged = merge_datasets(sample_salary_raw, sample_people_raw) # Pass only needed args

    # Sort by 'id' before comparing to ensure order doesn't matter
    merged_sorted = merged.sort_values(by="id").reset_index(drop=True)
    expected_sorted = sample_merged_data.sort_values(by="id").reset_index(drop=True)

    # Use check_like=True if column order might differ but names/types should match
    assert_frame_equal(merged_sorted, expected_sorted, check_like=True)

def test_clean_data(sample_data_for_cleaning, sample_data_processing_params, sample_cleaned_data):
    """Test the clean_data function."""
    cleaned = clean_data(sample_data_for_cleaning.copy(), sample_data_processing_params) # Use copy

    # Reset index for consistent comparison
    cleaned = cleaned.reset_index(drop=True)
    expected = sample_cleaned_data.reset_index(drop=True)

    # Check specific conditions
    assert "id" not in cleaned.columns, "ID column should be dropped"
    assert cleaned.duplicated().sum() == 0, "Should contain no duplicate rows"
    assert cleaned[sample_data_processing_params["target_variable"]].isnull().sum() == 0, "Should have no NaNs in target variable"

    # Compare the final dataframe structure (ignoring column order potentially)
    # Sort columns alphabetically for robust comparison
    cleaned_sorted_cols = cleaned.sort_index(axis=1)
    expected_sorted_cols = expected.sort_index(axis=1)
    assert_frame_equal(cleaned_sorted_cols, expected_sorted_cols, check_dtype=False)

def test_split_data(sample_cleaned_data, sample_data_processing_params):
    """Test the split_data function."""
    # Need enough data for split, sample_cleaned_data might be small
    # Let's create slightly larger cleaned data for this test
    if len(sample_cleaned_data) < 4: # Need at least 4 for a 75/25 split
        # Example: duplicate some rows for testing split sizes
        cleaned_for_split = pd.concat([sample_cleaned_data] * 2, ignore_index=True)
        if len(cleaned_for_split) < 4:
             cleaned_for_split = pd.concat([cleaned_for_split] * 2, ignore_index=True) # Make it bigger
    else:
        cleaned_for_split = sample_cleaned_data.copy()


    X_train, X_test, y_train, y_test = split_data(cleaned_for_split, sample_data_processing_params)

    target = sample_data_processing_params["target_variable"]
    test_size = sample_data_processing_params["test_size"]
    total_rows = len(cleaned_for_split)

    # Calculate expected sizes (handle potential rounding)
    expected_test_rows = int(round(total_rows * test_size))
    expected_train_rows = total_rows - expected_test_rows


    # Check types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame) # Changed to DataFrame
    assert isinstance(y_test, pd.DataFrame) # Changed to DataFrame

    # Check shapes
    assert X_train.shape[0] == expected_train_rows
    assert X_test.shape[0] == expected_test_rows
    assert y_train.shape[0] == expected_train_rows
    assert y_test.shape[0] == expected_test_rows

    # Check columns
    assert target not in X_train.columns
    assert target not in X_test.columns
    assert y_train.columns.tolist() == [target] # Check column name in DataFrame
    assert y_test.columns.tolist() == [target] # Check column name in DataFrame

    # Check if data is split correctly (no overlap in index)
    assert X_train.index.intersection(X_test.index).empty
    assert y_train.index.intersection(y_test.index).empty
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)

# Assume build_preprocessing_pipeline function exists and is imported
# from salary_prediction.pipelines.data_processing.nodes import build_preprocessing_pipeline
# --- Mock build_preprocessing_pipeline for testing ---
# This is a simplified placeholder. Your actual function will be more complex.
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
def build_preprocessing_pipeline(X_train, X_test, parameters):
    """Placeholder preprocessing pipeline builder"""
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    low_cardinality_features = [
        col for col in categorical_features
        if X_train[col].nunique() <= parameters['cardinality_threshold']
    ]
    high_cardinality_features = [
        col for col in categorical_features
        if X_train[col].nunique() > parameters['cardinality_threshold']
    ]

    # Create preprocessing pipelines for different feature types
    numeric_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    low_cardinality_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')) # drop='first'
    ])

    high_cardinality_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Handle unknown values explicitly for OrdinalEncoder during transform
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('low_cat', low_cardinality_transformer, low_cardinality_features),
            ('high_cat', high_cardinality_transformer, high_cardinality_features)
        ],
        remainder='passthrough' # Keep other columns if any (shouldn't be if features defined correctly)
    )

    # Fit on training data and transform both sets
    X_train_prep_np = preprocessor.fit_transform(X_train)
    X_test_prep_np = preprocessor.transform(X_test)

    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()

    # Convert back to DataFrames
    X_train_prep = pd.DataFrame(X_train_prep_np, columns=feature_names, index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_prep_np, columns=feature_names, index=X_test.index)

    return X_train_prep, X_test_prep, preprocessor
# --- End Mock ---

def test_build_preprocessing_pipeline(sample_data_processing_params):
    """Test the build_preprocessing_pipeline function with updated columns."""
    # Create sample data reflecting columns AFTER cleaning (no id, no target)
    X_train = pd.DataFrame({
        "Age": [32.0, 28.0, 45.0, np.nan, 52.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", np.nan], # Added NaN
        "Job Title": ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director"], # Assume high cardinality
        "Years of Experience": [5.0, 3.0, 15.0, 7.0, 20.0]
    })
    X_test = pd.DataFrame({
        "Age": [42.0, 31.0, 26.0],
        "Gender": ["Female", "Male", "Unknown"], # Unknown category
        "Education Level": ["Master's", "Bachelor's", "Master's"],
        "Job Title": ["Product Manager", "Sales Manager", "Data Analyst"], # Contains seen and unseen titles
        "Years of Experience": [12.0, np.nan, 1.0] # Added NaN
    })

    # Use the fixture directly (received as parameter)
    params = sample_data_processing_params.copy()  # Make a copy to avoid modifying the fixture

    # Modify the threshold specifically for this test's logic
    params['cardinality_threshold'] = 4 # Make Job Title high cardinality

    X_train_prep, X_test_prep, preprocessor = build_preprocessing_pipeline(X_train.copy(), X_test.copy(), params)
    
    # --- Checks ---
    # 1. Check output types
    assert isinstance(X_train_prep, pd.DataFrame)
    assert isinstance(X_test_prep, pd.DataFrame)
    assert isinstance(preprocessor, ColumnTransformer)

    # 2. Check shapes (rows should match input)
    assert X_train_prep.shape[0] == X_train.shape[0]
    assert X_test_prep.shape[0] == X_test.shape[0]

    # 3. Check preprocessor fitting (attempt to transform again)
    try:
        _ = preprocessor.transform(X_test)
    except Exception as e:
        pytest.fail(f"Preprocessor transform failed after initial fit/transform: {e}")

    # 4. Check for NaNs after preprocessing (should be handled by imputers)
    assert X_train_prep.isnull().sum().sum() == 0, "No NaNs expected in preprocessed training data"
    # Test data might get -1 from OrdinalEncoder for unseen, but not NaN from imputers/OHE ignore
    # Let's check if any NaN remains instead of checking for specific values like -1
    assert X_test_prep.isnull().sum().sum() == 0, "No NaNs expected in preprocessed test data with current settings"


    # 5. Check column names and count (this is crucial and depends on the transformations)
    # Expected columns based on threshold=4 and drop='first' for OHE:
    # - num__Age, num__Years of Experience (from num pipeline) - 2 cols
    # - low_cat__Gender_Male (OHE on Gender, Female dropped) - 1 col
    # - low_cat__Education Level_Master's, low_cat__Education Level_PhD (OHE on Education Level, Bachelor's dropped) - 2 cols
    # - high_cat__Job Title (from Ordinal Encoder) - 1 col
    # Total = 2 + 1 + 2 + 1 = 6 columns
    expected_col_count = 6
    assert X_train_prep.shape[1] == expected_col_count, f"Expected {expected_col_count} columns, got {X_train_prep.shape[1]}"
    assert X_test_prep.shape[1] == expected_col_count

    expected_feature_names_sorted = sorted([
        'num__Age', 'num__Years of Experience',
        'low_cat__Gender_Male',
        'low_cat__Education Level_Master\'s', 'low_cat__Education Level_PhD', # OHE names depend on imputer+encoder results
        'high_cat__Job Title'
    ])
    actual_feature_names_sorted = sorted(X_train_prep.columns.tolist())

    assert actual_feature_names_sorted == expected_feature_names_sorted, \
        f"Expected features: {expected_feature_names_sorted}\nActual features: {actual_feature_names_sorted}"
    assert sorted(X_test_prep.columns.tolist()) == expected_feature_names_sorted

    # 6. Check if preprocessor contains the right steps (optional but good)
    assert "num" in preprocessor.named_transformers_
    assert "low_cat" in preprocessor.named_transformers_
    assert "high_cat" in preprocessor.named_transformers_

    assert isinstance(preprocessor.named_transformers_["num"], SklearnPipeline)
    assert isinstance(preprocessor.named_transformers_["low_cat"], SklearnPipeline)
    assert isinstance(preprocessor.named_transformers_["high_cat"], SklearnPipeline)

    # Optionally, check the steps within each pipeline
    assert "scaler" in preprocessor.named_transformers_["num"].named_steps
    assert "imputer" in preprocessor.named_transformers_["num"].named_steps
    assert "onehot" in preprocessor.named_transformers_["low_cat"].named_steps
    assert "imputer" in preprocessor.named_transformers_["low_cat"].named_steps
    assert "ordinal" in preprocessor.named_transformers_["high_cat"].named_steps
    assert "imputer" in preprocessor.named_transformers_["high_cat"].named_steps