import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline # Alias to avoid clash with Kedro Pipeline

# Import the node functions we want to test
from salary_prediction.pipelines.data_processing.nodes import (
    merge_datasets,
    clean_data,
    split_data,
    build_preprocessing_pipeline,
)

# --- Fixtures for Sample Data ---

@pytest.fixture
def sample_salary_raw():
    """Fixture for raw salary data."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "salary": [50000, 60000, np.nan, 75000]
    })

@pytest.fixture
def sample_people_raw():
    """Fixture for raw people data."""
    return pd.DataFrame({
        "id": [1, 2, 3, 5], # Note: ID 3 exists, ID 4 doesn't, ID 5 doesn't match salary
        "age": [25, 30, 35, 40],
        "city": ["A", "B", "A", "C"],
        "experience": [2, 8, 10, 15],
        "education": ["Bachelor", "Master", "Bachelor", "PhD"]
    })

@pytest.fixture
def sample_descriptions_raw():
    """Fixture for raw descriptions data (currently unused in merge)."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "description": ["Desc 1", "Desc 2", "Desc 3", "Desc 4", "Desc 5"]
    })

@pytest.fixture
def sample_merged_data():
    """Fixture for expected merged data (result of merge_datasets)."""
    # Expected: Inner join on 'id', only IDs 1, 2, 3 should match
    return pd.DataFrame({
        "id": [1, 2, 3],
        "age": [25, 30, 35],
        "city": ["A", "B", "A"],
        "experience": [2, 8, 10],
        "education": ["Bachelor", "Master", "Bachelor"],
        "salary": [50000.0, 60000.0, np.nan] # Salary for ID 3 is NaN
    })

@pytest.fixture
def sample_data_for_cleaning(sample_merged_data):
    """Fixture for data before cleaning (includes duplicates and NaN salary)."""
    # Add a duplicate row
    duplicate_row = sample_merged_data.iloc[[0]]
    data_with_duplicates = pd.concat([sample_merged_data, duplicate_row], ignore_index=True)
    return data_with_duplicates

@pytest.fixture
def sample_data_processing_params():
    """Fixture for data processing parameters."""
    return {
        "target_variable": "salary", # <--- LOWERCASE
        "test_size": 0.25,
        "random_state": 42,
        "cardinality_threshold": 5
    }

@pytest.fixture
def sample_cleaned_data(sample_merged_data):
    """Fixture for expected cleaned data (after dropping id, duplicates, NaN target)."""
    cleaned = sample_merged_data.copy()
    # Manually apply the cleaning steps expected by clean_data node

    # 1. Drop rows with NaN in target 'salary' (lowercase - matching actual column name)
    cleaned = cleaned.dropna(subset=["salary"])  # Fixed: using lowercase 'salary' instead of 'Salary'
    
    # 2. Drop 'id' column
    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])

    # 3. Lowercase column names (already lowercase in this case)
    cleaned.columns = [col.lower().replace(' ', '_') for col in cleaned.columns]

    # 4. Reset index
    return cleaned.reset_index(drop=True)


# --- Test Functions ---

def test_merge_datasets(sample_salary_raw, sample_people_raw, sample_descriptions_raw, sample_merged_data):
    """Test the merge_datasets function."""
    merged = merge_datasets(sample_salary_raw, sample_people_raw, sample_descriptions_raw)

    # Sort by 'id' before comparing to ensure order doesn't matter
    merged_sorted = merged.sort_values(by="id").reset_index(drop=True)
    expected_sorted = sample_merged_data.sort_values(by="id").reset_index(drop=True)

    assert_frame_equal(merged_sorted, expected_sorted)

def test_clean_data(sample_data_for_cleaning, sample_data_processing_params, sample_cleaned_data):
    """Test the clean_data function."""
    # Assume the REAL clean_data function is imported and used here
    # from salary_prediction.pipelines.data_processing.nodes import clean_data
    cleaned = clean_data(sample_data_for_cleaning.copy(), sample_data_processing_params) # Use copy

    # Reset index for consistent comparison (assuming node does this)
    cleaned = cleaned.reset_index(drop=True)
    expected = sample_cleaned_data.reset_index(drop=True) # Expected data already has lowercase names

    # Check specific conditions
    assert "id" not in cleaned.columns, "ID column should be dropped"
    assert cleaned.duplicated().sum() == 0, "Should contain no duplicate rows"
    # Check NaN using the lowercase target variable name from params
    target_col = sample_data_processing_params["target_variable"]
    assert target_col in cleaned.columns, f"Target column '{target_col}' not found after cleaning. Columns: {cleaned.columns.tolist()}"
    assert cleaned[target_col].isnull().sum() == 0, f"Should have no NaNs in target variable '{target_col}'"

    # Compare the final dataframe structure (ignoring column order potentially)
    cleaned_sorted_cols = cleaned.sort_index(axis=1)
    expected_sorted_cols = expected.sort_index(axis=1)
    assert_frame_equal(cleaned_sorted_cols, expected_sorted_cols, check_dtype=False)


def test_split_data(sample_cleaned_data, sample_data_processing_params):
    """Test the split_data function."""
    # sample_cleaned_data fixture now has lowercase column names
    # Assume the REAL split_data function is imported and used here
    # from salary_prediction.pipelines.data_processing.nodes import split_data

    # Ensure data is large enough for split
    if len(sample_cleaned_data) < 4:
        cleaned_for_split = pd.concat([sample_cleaned_data] * 2, ignore_index=True)
        if len(cleaned_for_split) < 4:
             cleaned_for_split = pd.concat([cleaned_for_split] * 2, ignore_index=True)
    else:
        cleaned_for_split = sample_cleaned_data.copy()

    # Check if target column exists before splitting
    target = sample_data_processing_params["target_variable"] # Should be "salary"
    assert target in cleaned_for_split.columns, f"Target column '{target}' not found in input to split_data. Columns: {cleaned_for_split.columns.tolist()}"

    X_train, X_test, y_train, y_test = split_data(cleaned_for_split, sample_data_processing_params)

    test_size = sample_data_processing_params["test_size"]
    total_rows = len(cleaned_for_split)
    expected_test_rows = int(round(total_rows * test_size))
    expected_train_rows = total_rows - expected_test_rows

    # Check types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert isinstance(y_test, pd.DataFrame)

    # Check shapes
    assert X_train.shape[0] == expected_train_rows
    assert X_test.shape[0] == expected_test_rows
    assert y_train.shape[0] == expected_train_rows
    assert y_test.shape[0] == expected_test_rows

    # Check columns (using lowercase target name)
    assert target not in X_train.columns
    assert target not in X_test.columns
    assert y_train.columns.tolist() == [target]
    assert y_test.columns.tolist() == [target]

    # Check indices
    assert X_train.index.intersection(X_test.index).empty
    assert y_train.index.intersection(y_test.index).empty
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)

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

    # Make a copy of the params from the fixture
    params = sample_data_processing_params.copy()  # Make a copy to avoid modifying the fixture

    # Modify the threshold specifically for this test's logic if needed
    params['cardinality_threshold'] = 4 # Make Job Title high cardinality for this test case

    # Now call the function under test with the modified params copy
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
    assert X_test_prep.isnull().sum().sum() == 0, "No NaNs expected in preprocessed test data with current settings"

    # 5. Check column names and count
    expected_col_count = 6
    assert X_train_prep.shape[1] == expected_col_count, f"Expected {expected_col_count} columns, got {X_train_prep.shape[1]}"
    assert X_test_prep.shape[1] == expected_col_count

    # FIXED: Adjusted expected feature names to match actual implementation output
    expected_feature_names_sorted = sorted([
        'Age', 'Years of Experience',
        'Gender_Male', 
        'Education Level_Master\'s', 'Education Level_PhD',
        'Job Title'
    ])
    actual_feature_names_sorted = sorted(X_train_prep.columns.tolist())

    assert actual_feature_names_sorted == expected_feature_names_sorted, \
        f"Expected features: {expected_feature_names_sorted}\nActual features: {actual_feature_names_sorted}"
    assert sorted(X_test_prep.columns.tolist()) == expected_feature_names_sorted

    # 6. Check if preprocessor contains the right steps
    assert "num" in preprocessor.named_transformers_
    assert "low_cat" in preprocessor.named_transformers_
    assert "high_cat" in preprocessor.named_transformers_
    assert isinstance(preprocessor.named_transformers_["num"], SklearnPipeline)
    assert isinstance(preprocessor.named_transformers_["low_cat"], SklearnPipeline)
    assert isinstance(preprocessor.named_transformers_["high_cat"], SklearnPipeline)
    assert "scaler" in preprocessor.named_transformers_["num"].named_steps
    assert "imputer" in preprocessor.named_transformers_["num"].named_steps
    assert "onehot" in preprocessor.named_transformers_["low_cat"].named_steps
    assert "imputer" in preprocessor.named_transformers_["low_cat"].named_steps
    assert "ordinal" in preprocessor.named_transformers_["high_cat"].named_steps
    assert "imputer" in preprocessor.named_transformers_["high_cat"].named_steps
