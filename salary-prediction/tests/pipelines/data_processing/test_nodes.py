# tests/pipelines/data_processing/test_nodes.py

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import logging # Import logging

# Assuming your nodes are in salary_prediction.pipelines.data_processing.nodes
# Adjust the import path if your project structure is different
from salary_prediction.pipelines.data_processing.nodes import (
    merge_datasets,
    clean_data,
    split_data,
    build_preprocessing_pipeline,
)

# --- Fixtures for Sample Data ---

@pytest.fixture
def sample_people_df():
    """Fixture for sample people data."""
    data = {
        "id": [0, 1, 2, 3, 4, 5],
        "Age": [32.0, 28.0, 45.0, 36.0, 52.0, 29.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Male"],
        "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's", "Bachelor's"],
        "Job Title": ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director", "Marketing Analyst"],
        "Years of Experience": [5.0, 3.0, 15.0, 7.0, 20.0, 2.0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_salary_df():
    """Fixture for sample salary data."""
    data = {
        "id": [0, 1, 2, 3, 4, 6], # ID 5 missing, ID 6 extra
        "Salary": [90000.0, 65000.0, 150000.0, 60000.0, 200000.0, 55000.0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_descriptions_df():
    """Fixture for sample descriptions data (unused in merge but required by signature)."""
    data = {
        "id": [0, 1, 2],
        "description": ["Desc 0", "Desc 1", "Desc 2"],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_merged_df(sample_people_df, sample_salary_df):
    """Fixture for expected merged data (inner join)."""
    # Manually perform the expected merge
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "Age": [32.0, 28.0, 45.0, 36.0, 52.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's"],
        "Job Title": ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director"],
        "Years of Experience": [5.0, 3.0, 15.0, 7.0, 20.0],
        "Salary": [90000.0, 65000.0, 150000.0, 60000.0, 200000.0],
    })

@pytest.fixture
def sample_dirty_data_for_cleaning(sample_merged_df):
    """Fixture for data with issues to be cleaned."""
    dirty_df = sample_merged_df.copy()
    # Add duplicate row
    dirty_df = pd.concat([dirty_df, dirty_df.iloc[[0]]], ignore_index=True)
    # Add row with missing Salary (np.nan)
    missing_salary_row = pd.DataFrame({
        "id": [10], "Age": [30.0], "Gender": ["Other"], "Education Level": ["Master's"],
        "Job Title": ["Tester"], "Years of Experience": [4.0], "Salary": [np.nan]
    })
    dirty_df = pd.concat([dirty_df, missing_salary_row], ignore_index=True)
    # Add row with None Salary (should also be treated as missing)
    # Explicitly cast None to float for Salary column consistency if needed, though dropna handles it
    none_salary_row = pd.DataFrame({
        "id": [11], "Age": [40.0], "Gender": ["Male"], "Education Level": ["PhD"],
        "Job Title": ["Researcher"], "Years of Experience": [10.0], "Salary": [None]
    })
    dirty_df = pd.concat([dirty_df, none_salary_row], ignore_index=True)
    # Add row with valid salary to ensure it's kept
    valid_row = pd.DataFrame({
        "id": [12], "Age": [35.0], "Gender": ["Female"], "Education Level": ["Bachelor's"],
        "Job Title": ["HR Manager"], "Years of Experience": [8.0], "Salary": [75000.0]
    })
    dirty_df = pd.concat([dirty_df, valid_row], ignore_index=True)
    # Ensure Salary column is float type after potentially adding None
    dirty_df['Salary'] = dirty_df['Salary'].astype(float)
    return dirty_df

@pytest.fixture
def data_processing_parameters():
    """Fixture for parameters used in data processing."""
    return {
        "target_variable": "Salary",
        "test_size": 0.2,
        "random_state": 42,
        "cardinality_threshold": 5, # Categories with < 5 unique values are low cardinality
    }

@pytest.fixture
def sample_data_for_splitting():
    """Fixture for data ready to be split (cleaned)."""
    data = {
        "Age": [32.0, 28.0, 45.0, 36.0, 52.0, 29.0, 42.0, 31.0, 26.0, 38.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Male", "Female", "Male", "Female", "Male"],
        "Education Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's", "Bachelor's", "Master's", "Bachelor's", "Bachelor's", "PhD"],
        "Job Title": ["SWE", "DA", "SM", "SA", "Dir", "MA", "PM", "SMgr", "MC", "RS"], # Shortened for clarity
        "Years of Experience": [5.0, 3.0, 15.0, 7.0, 20.0, 2.0, 12.0, 4.0, 1.0, 10.0],
        "Salary": [90, 65, 150, 60, 200, 55, 120, 80, 45, 130], # Simplified salaries
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_for_preprocessing():
    """Fixture providing sample train/test splits for preprocessing tests."""
    X_train = pd.DataFrame({
        "Age": [32.0, 45.0, 52.0, 29.0, 42.0, 31.0, 26.0, 38.0, np.nan], # Added NaN for imputation test (N=9)
        "Gender": ["Male", "Male", "Male", "Male", "Female", "Male", "Female", "Male", "Female"],
        "Education Level": ["Bachelor's", "PhD", "Master's", "Bachelor's", "Master's", "Bachelor's", "Bachelor's", "PhD", "Master's"],
        # Replaced None with np.nan for compatibility with SimpleImputer's default missing_values
        "Job Title": ["SWE", "SM", "Dir", "MA", "PM", "SMgr", "MC", "RS", np.nan], # Added np.nan for imputation test
        "Years of Experience": [5.0, 15.0, 20.0, 2.0, 12.0, 4.0, 1.0, 10.0, 6.0],
    })
    X_test = pd.DataFrame({
        "Age": [28.0, 36.0],
        "Gender": ["Female", "Other"], # Added "Other" to test handle_unknown
        "Education Level": ["Master's", "Bachelor's"],
        "Job Title": ["DA", "UnknownJob"], # Added "UnknownJob" to test handle_unknown
        "Years of Experience": [3.0, 7.0],
    })
    # Ensure correct dtypes, especially for object columns that might contain np.nan
    # SimpleImputer strategy='most_frequent' requires object dtype for categorical
    X_train['Job Title'] = X_train['Job Title'].astype(object)
    X_test['Job Title'] = X_test['Job Title'].astype(object)
    X_train['Gender'] = X_train['Gender'].astype(object)
    X_test['Gender'] = X_test['Gender'].astype(object)
    X_train['Education Level'] = X_train['Education Level'].astype(object)
    X_test['Education Level'] = X_test['Education Level'].astype(object)

    return X_train, X_test


# --- Test Classes ---

class TestMergeDatasets:
    def test_merge_basic(self, sample_people_df, sample_salary_df, sample_descriptions_df, sample_merged_df):
        """Test basic inner merge functionality."""
        merged = merge_datasets(sample_salary_df, sample_people_df, sample_descriptions_df)
        # Sort by id to ensure order doesn't affect comparison
        merged = merged.sort_values(by="id").reset_index(drop=True)
        expected = sample_merged_df.sort_values(by="id").reset_index(drop=True)
        assert_frame_equal(merged, expected)

    def test_merge_columns(self, sample_people_df, sample_salary_df, sample_descriptions_df):
        """Test if the merged dataframe has the correct columns."""
        merged = merge_datasets(sample_salary_df, sample_people_df, sample_descriptions_df)
        expected_columns = list(sample_people_df.columns) + ["Salary"]
        assert sorted(list(merged.columns)) == sorted(expected_columns)

    def test_merge_row_count(self, sample_people_df, sample_salary_df, sample_descriptions_df):
        """Test if the row count corresponds to an inner join."""
        merged = merge_datasets(sample_salary_df, sample_people_df, sample_descriptions_df)
        # IDs present in both: 0, 1, 2, 3, 4 (5 total)
        assert len(merged) == 5

    def test_merge_empty_people(self, sample_salary_df, sample_descriptions_df):
        """Test merging with an empty people dataframe."""
        empty_people_df = pd.DataFrame(columns=["id", "Age", "Gender"])
        merged = merge_datasets(sample_salary_df, empty_people_df, sample_descriptions_df)
        assert merged.empty
        # Check columns are combined even if empty, preserving potential structure
        assert sorted(list(merged.columns)) == sorted(["id", "Age", "Gender", "Salary"])

    def test_merge_empty_salary(self, sample_people_df, sample_descriptions_df):
        """Test merging with an empty salary dataframe."""
        empty_salary_df = pd.DataFrame(columns=["id", "Salary"])
        merged = merge_datasets(empty_salary_df, sample_people_df, sample_descriptions_df)
        assert merged.empty
        assert sorted(list(merged.columns)) == sorted(list(sample_people_df.columns) + ["Salary"])

    def test_merge_no_common_ids(self, sample_people_df, sample_descriptions_df):
        """Test merging when there are no common IDs."""
        salary_no_common = pd.DataFrame({"id": [101, 102], "Salary": [50000, 60000]})
        merged = merge_datasets(salary_no_common, sample_people_df, sample_descriptions_df)
        assert merged.empty

class TestCleanData:
    def test_removes_duplicates(self, sample_dirty_data_for_cleaning, data_processing_parameters):
        """Test that duplicate rows are removed."""
        cleaned = clean_data(sample_dirty_data_for_cleaning.copy(), data_processing_parameters)
        # Initial dirty has 5 unique from sample + 1 duplicate + 1 nan salary + 1 none salary + 1 valid extra = 9 rows
        # Clean should remove duplicate (1), nan salary (1), none salary (1) -> 9 - 3 = 6 rows expected
        # It also drops 'id'. Original unique rows (5) + 1 added valid row = 6
        assert cleaned.duplicated().sum() == 0
        assert len(cleaned) == 6 # 5 original unique + 1 added valid row

    def test_removes_id_column(self, sample_dirty_data_for_cleaning, data_processing_parameters):
        """Test that the 'id' column is dropped."""
        cleaned = clean_data(sample_dirty_data_for_cleaning.copy(), data_processing_parameters)
        assert "id" not in cleaned.columns

    def test_handles_na_in_target(self, sample_dirty_data_for_cleaning, data_processing_parameters):
        """Test that rows with NA/None in the target variable are dropped."""
        target = data_processing_parameters["target_variable"]
        # Check raw data has NAs/None in target
        assert sample_dirty_data_for_cleaning[target].isnull().sum() == 2
        cleaned = clean_data(sample_dirty_data_for_cleaning.copy(), data_processing_parameters)
        # Check cleaned data has no NAs in target
        assert cleaned[target].isnull().sum() == 0
        # Check specific rows were dropped (those with original id 10 and 11 had nan/none salary)
        # Since 'id' is dropped, we check based on count
        assert len(cleaned) == 6 # As established in duplicate test

    def test_no_changes_needed(self, sample_merged_df, data_processing_parameters):
        """Test cleaning data that is already clean (except for 'id')."""
        # sample_merged_df is already unique and has no missing salaries
        df_copy = sample_merged_df.copy()
        cleaned = clean_data(df_copy, data_processing_parameters)
        expected_cols = [col for col in sample_merged_df.columns if col != 'id']
        expected_rows = len(sample_merged_df)

        assert "id" not in cleaned.columns
        # Sort columns for comparison as 'id' removal might change order
        assert sorted(list(cleaned.columns)) == sorted(expected_cols)
        assert len(cleaned) == expected_rows
        # Check data content preservation (ignoring id)
        # Sort columns in both frames before comparing
        assert_frame_equal(cleaned.sort_index(axis=1), sample_merged_df[expected_cols].sort_index(axis=1), check_dtype=False)


class TestSplitData:
    def test_output_types(self, sample_data_for_splitting, data_processing_parameters):
        """Test if the outputs are pandas DataFrames."""
        X_train, X_test, y_train, y_test = split_data(sample_data_for_splitting, data_processing_parameters)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame) # Check if node returns DataFrame for y
        assert isinstance(y_test, pd.DataFrame)  # Check if node returns DataFrame for y

    def test_split_shapes(self, sample_data_for_splitting, data_processing_parameters):
        """Test the shapes of the output dataframes based on test_size."""
        n_rows = len(sample_data_for_splitting)
        test_size = data_processing_parameters["test_size"]
        # Use rounding consistent with train_test_split (floor for test size usually)
        expected_test_rows = int(np.floor(n_rows * test_size))
        expected_train_rows = n_rows - expected_test_rows
        n_features = len(sample_data_for_splitting.columns) - 1 # Exclude target

        X_train, X_test, y_train, y_test = split_data(sample_data_for_splitting, data_processing_parameters)

        assert X_train.shape == (expected_train_rows, n_features)
        assert X_test.shape == (expected_test_rows, n_features)
        assert y_train.shape == (expected_train_rows, 1)
        assert y_test.shape == (expected_test_rows, 1)

    def test_split_columns(self, sample_data_for_splitting, data_processing_parameters):
        """Test if X has features and y has the target."""
        target = data_processing_parameters["target_variable"]
        feature_cols = [col for col in sample_data_for_splitting.columns if col != target]

        X_train, X_test, y_train, y_test = split_data(sample_data_for_splitting, data_processing_parameters)

        assert list(X_train.columns) == feature_cols
        assert list(X_test.columns) == feature_cols
        assert list(y_train.columns) == [target]
        assert list(y_test.columns) == [target]

    def test_random_state_reproducibility(self, sample_data_for_splitting, data_processing_parameters):
        """Test if the split is the same for the same random_state."""
        X_train1, X_test1, y_train1, y_test1 = split_data(sample_data_for_splitting.copy(), data_processing_parameters)
        X_train2, X_test2, y_train2, y_test2 = split_data(sample_data_for_splitting.copy(), data_processing_parameters)

        assert_frame_equal(X_train1, X_train2)
        assert_frame_equal(X_test1, X_test2)
        assert_frame_equal(y_train1, y_train2)
        assert_frame_equal(y_test1, y_test2)

    def test_different_test_size(self, sample_data_for_splitting, data_processing_parameters):
        """Test splitting with a different test size."""
        params = data_processing_parameters.copy()
        params["test_size"] = 0.5
        n_rows = len(sample_data_for_splitting)
        expected_test_rows = int(np.floor(n_rows * 0.5)) # Use floor consistent with train_test_split
        expected_train_rows = n_rows - expected_test_rows

        X_train, X_test, y_train, y_test = split_data(sample_data_for_splitting, params)

        assert len(X_train) == expected_train_rows
        assert len(X_test) == expected_test_rows
        assert len(y_train) == expected_train_rows
        assert len(y_test) == expected_test_rows

class TestBuildPreprocessingPipeline:

    def test_output_types(self, sample_data_for_preprocessing, data_processing_parameters):
        """Test the types of the returned objects."""
        X_train, X_test = sample_data_for_preprocessing
        # Pass copies to avoid modification issues between tests
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train.copy(), X_test.copy(), data_processing_parameters)

        assert isinstance(X_train_t, pd.DataFrame)
        assert isinstance(X_test_t, pd.DataFrame)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_output_shapes(self, sample_data_for_preprocessing, data_processing_parameters):
        """Test the shapes of the transformed dataframes."""
        X_train, X_test = sample_data_for_preprocessing
        # Pass copies
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train_orig, X_test_orig, data_processing_parameters)

        # Recalculate expected columns based on node logic and threshold=5
        # Assumes node calculates nunique on original X_train (ignoring NaNs)
        num_cols = ["Age", "Years of Experience"] # 2
        cat_cols = ["Gender", "Education Level", "Job Title"]

        # Calculate unique counts on the original data (excluding NaNs for nunique)
        low_card_cols = [
            col for col in cat_cols
            if X_train[col].nunique() < data_processing_parameters["cardinality_threshold"]
        ] # Should be ['Gender'(2), 'Education Level'(3)] -> threshold < 5
        high_card_cols = [col for col in cat_cols if col not in low_card_cols] # Should be ['Job Title'(8)] -> threshold >= 5

        # Calculate OHE columns count using the fitted preprocessor
        ohe_cols_count = 0
        # Check if 'low_cat' transformer exists and has 'onehot' step
        if 'low_cat' in preprocessor.named_transformers_:
            low_cat_pipeline = preprocessor.named_transformers_['low_cat']
            # Check if it's a Pipeline object containing 'onehot'
            if isinstance(low_cat_pipeline, Pipeline) and 'onehot' in low_cat_pipeline.named_steps:
                ohe_transformer = low_cat_pipeline.named_steps['onehot']
                # Ensure the transformer was actually fitted (has attributes)
                if hasattr(ohe_transformer, 'get_feature_names_out'):
                     # Provide the input feature names *as seen by the OHE step*
                     # which are the original low_card_cols names
                    ohe_cols_count = len(ohe_transformer.get_feature_names_out(low_card_cols))
            # Check if it's directly an OneHotEncoder (if Pipeline only had OHE)
            elif isinstance(low_cat_pipeline, OneHotEncoder):
                 if hasattr(low_cat_pipeline, 'get_feature_names_out'):
                    ohe_cols_count = len(low_cat_pipeline.get_feature_names_out(low_card_cols))

        # Total expected cols = num_scaled (2) + ohe_low_card (?) + ordinal_high_card (1)
        # Gender(2 unique)-> OHE(drop='first') -> 1 feature
        # Edu Lvl(3 unique)-> OHE(drop='first') -> 2 features
        # Job Title(8 unique)-> Ordinal -> 1 feature
        # Total = 2 + (1 + 2) + 1 = 6
        expected_cols = len(num_cols) + ohe_cols_count + len(high_card_cols)

        assert X_train_t.shape[0] == X_train.shape[0]
        assert X_test_t.shape[0] == X_test.shape[0]
        assert X_train_t.shape[1] == expected_cols, f"Expected {expected_cols} columns, but got {X_train_t.shape[1]}"
        assert X_test_t.shape[1] == expected_cols, f"Expected {expected_cols} columns, but got {X_test_t.shape[1]}"

    def test_column_names(self, sample_data_for_preprocessing, data_processing_parameters):
        """Test if the transformed dataframes have meaningful column names."""
        X_train, X_test = sample_data_for_preprocessing
        # Pass copies
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train_orig, X_test_orig, data_processing_parameters)

        # Expected names based on revised logic (threshold=5)
        num_cols = ["Age", "Years of Experience"] # From num pipeline
        high_card_cols = ["Job Title"]           # From high_cat pipeline (Ordinal) - retains original name

        # Get OHE feature names from the fitted preprocessor for low_cat columns
        low_card_input_cols = ["Gender", "Education Level"] # Columns fed into low_cat pipeline
        ohe_feature_names = []
        if 'low_cat' in preprocessor.named_transformers_:
            low_cat_pipeline = preprocessor.named_transformers_['low_cat']
            if isinstance(low_cat_pipeline, Pipeline) and 'onehot' in low_cat_pipeline.named_steps:
                ohe_transformer = low_cat_pipeline.named_steps['onehot']
                if hasattr(ohe_transformer, 'get_feature_names_out'):
                    ohe_feature_names = list(ohe_transformer.get_feature_names_out(low_card_input_cols))
            elif isinstance(low_cat_pipeline, OneHotEncoder):
                if hasattr(low_cat_pipeline, 'get_feature_names_out'):
                    ohe_feature_names = list(low_cat_pipeline.get_feature_names_out(low_card_input_cols))

        # Order is determined by ColumnTransformer: num, low_cat, high_cat
        expected_feature_names = num_cols + ohe_feature_names + high_card_cols

        # print(f"Actual columns: {list(X_train_t.columns)}")
        # print(f"Expected columns: {expected_feature_names}")

        assert list(X_train_t.columns) == expected_feature_names
        assert list(X_test_t.columns) == expected_feature_names

    def test_numerical_scaling_imputation(self, sample_data_for_preprocessing, data_processing_parameters):
        """Test numerical imputation (median) and scaling (StandardScaler)."""
        X_train, X_test = sample_data_for_preprocessing
        # Pass copies
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train_orig, X_test_orig, data_processing_parameters)

        num_cols = ["Age", "Years of Experience"]

        # Check imputation: No NaNs in transformed numerical columns
        assert X_train_t[num_cols].isnull().sum().sum() == 0
        assert X_test_t[num_cols].isnull().sum().sum() == 0

        # Check scaling on Train set (mean approx 0, std dev approx 1)
        # Find index of original NaN in Age column *before* processing
        original_age_nan_index = X_train['Age'].reset_index(drop=True).index[X_train['Age'].reset_index(drop=True).isna()]
        median_age = X_train['Age'].median() # Calculate median from original data (as imputer would)

        # Get the fitted imputer and scaler for numerical columns
        num_transformer = preprocessor.named_transformers_['num']
        imputer = num_transformer.named_steps['imputer']
        scaler = num_transformer.named_steps['scaler']

        # Verify the imputer used the correct median (it stores it in statistics_)
        assert np.isclose(imputer.statistics_[0], median_age) # Age is the first numerical column

        # Manually calculate the expected scaled value for the imputed median age using the scaler's parameters
        # scaler.scale_ IS the sample standard deviation (ddof=1) used for scaling
        scaled_median_age_expected = (median_age - scaler.mean_[0]) / scaler.scale_[0]

        # Verify the transformed train data stats
        # Mean should be close to 0
        assert np.allclose(X_train_t[num_cols].mean(axis=0), 0, atol=1e-6)

        # Check Sample standard deviation (ddof=1)
        # NOTE: Observed behavior in this test setup yields std dev = sqrt(N/(N-1))
        # where N is the number of samples (9). This suggests the scaler
        # might be effectively normalizing based on population std dev=1,
        # resulting in a sample std dev calculation of sqrt(9/8) = 1.06066.
        # We test against this observed value.
        N = X_train_t.shape[0] # Number of samples used in fit (post-imputation)
        if N > 1:
            expected_std_dev = np.sqrt(N / (N - 1.0))
        else:
            expected_std_dev = 1.0 # Or handle as appropriate if N=1 possible

        actual_std_dev = X_train_t[num_cols].std(axis=0, ddof=1)
        logging.info(f"Calculated std dev (ddof=1):\n{actual_std_dev}")
        logging.info(f"Expected std dev sqrt(N/(N-1)) for N={N}: {expected_std_dev}")

        assert np.allclose(actual_std_dev, expected_std_dev, atol=1e-6), \
            f"Sample standard deviation not close to {expected_std_dev:.5f} (sqrt(N/(N-1))). Got:\n{actual_std_dev}"

        # Check the imputed value was scaled correctly in the output DataFrame
        # Use .iloc because original_age_nan_index refers to position after reset_index()
        imputed_scaled_value_actual = X_train_t.iloc[original_age_nan_index[0]]['Age']
        assert np.isclose(imputed_scaled_value_actual, scaled_median_age_expected)


    def test_categorical_imputation_encoding(self, sample_data_for_preprocessing, data_processing_parameters):
        """Test categorical imputation (most frequent) and encoding (OHE/Ordinal)."""
        X_train, X_test = sample_data_for_preprocessing
         # Pass copies
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train_orig, X_test_orig, data_processing_parameters)


        # Revised split based on threshold 5
        low_card_input_cols = ["Gender", "Education Level"] # Input names for OHE part
        high_card_cols = ["Job Title"] # Input name for Ordinal part

        # Check imputation: No NaNs in final output
        assert X_train['Job Title'].isnull().sum() > 0 # Verify NaN exists in input Job Title
        assert X_train['Age'].isnull().sum() > 0 # Verify NaN exists in input Age
        assert X_train_t.isnull().sum().sum() == 0, "NaNs found in transformed training data"
        assert X_test_t.isnull().sum().sum() == 0, "NaNs found in transformed test data"

        # Check OHE encoding
        ohe_feature_names = []
        if 'low_cat' in preprocessor.named_transformers_:
            low_cat_pipeline = preprocessor.named_transformers_['low_cat']
            if isinstance(low_cat_pipeline, Pipeline) and 'onehot' in low_cat_pipeline.named_steps:
                ohe_transformer = low_cat_pipeline.named_steps['onehot']
                if hasattr(ohe_transformer, 'get_feature_names_out'):
                    ohe_feature_names = list(ohe_transformer.get_feature_names_out(low_card_input_cols))
            elif isinstance(low_cat_pipeline, OneHotEncoder): # If not a pipeline but just OHE
                 if hasattr(low_cat_pipeline, 'get_feature_names_out'):
                     ohe_feature_names = list(low_cat_pipeline.get_feature_names_out(low_card_input_cols))

        assert len(ohe_feature_names) == 3, f"Expected 3 OHE features, got {len(ohe_feature_names)}" # Gender(1) + EduLevel(2) = 3 OHE features
        assert all(col in X_train_t.columns for col in ohe_feature_names)
        # Check if OHE values are 0 or 1
        assert X_train_t[ohe_feature_names].apply(lambda x: x.isin([0, 1])).all().all(), f"OHE columns contain non-binary values: {X_train_t[ohe_feature_names]}"
        assert X_test_t[ohe_feature_names].apply(lambda x: x.isin([0, 1])).all().all(), f"OHE columns contain non-binary values: {X_test_t[ohe_feature_names]}"


        # Check Ordinal encoding (presence of column, numerical values >= -1)
        assert all(col in X_train_t.columns for col in high_card_cols), f"Expected columns {high_card_cols} not found" # Checks 'Job Title' exists
        # Ensure the column is numeric and >= -1 (-1 for unknown)
        assert pd.api.types.is_numeric_dtype(X_train_t['Job Title']), "Ordinal column 'Job Title' is not numeric"
        assert X_train_t['Job Title'].min() >= -1, f"Ordinal column 'Job Title' has values < -1: min={X_train_t['Job Title'].min()}"
        assert pd.api.types.is_numeric_dtype(X_test_t['Job Title']), "Ordinal column 'Job Title' in test set is not numeric"
        assert X_test_t['Job Title'].min() >= -1, f"Ordinal column 'Job Title' in test set has values < -1: min={X_test_t['Job Title'].min()}"

    def test_handle_unknown_categories(self, sample_data_for_preprocessing, data_processing_parameters):
        """Test how unknown categories in X_test are handled."""
        X_train, X_test = sample_data_for_preprocessing
         # Pass copies
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        # X_test contains "Other" for Gender (low card OHE) and "UnknownJob" for Job Title (high card Ordinal)
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train_orig, X_test_orig, data_processing_parameters)

        low_card_input_cols = ["Gender", "Education Level"] # Fed into low_cat
        high_card_input_cols = ["Job Title"] # Fed into high_cat

        # OHE (Gender): 'handle_unknown=ignore' -> should result in all zeros for the OHE columns *related to Gender* for that row
        ohe_feature_names = []
        ohe_transformer = None
        if 'low_cat' in preprocessor.named_transformers_:
            low_cat_pipeline = preprocessor.named_transformers_['low_cat']
            if isinstance(low_cat_pipeline, Pipeline) and 'onehot' in low_cat_pipeline.named_steps:
                ohe_transformer = low_cat_pipeline.named_steps['onehot']
            elif isinstance(low_cat_pipeline, OneHotEncoder):
                ohe_transformer = low_cat_pipeline

        assert ohe_transformer is not None, "Could not find OHE transformer"
        all_ohe_feature_names = ohe_transformer.get_feature_names_out(low_card_input_cols)
        # Filter for columns generated specifically from 'Gender'
        # This assumes get_feature_names_out format like 'feature_value'
        ohe_gender_cols = [col for col in all_ohe_feature_names if col.startswith("Gender_")]
        assert len(ohe_gender_cols) > 0, "No OHE columns generated for Gender feature"


        unknown_gender_index = X_test[X_test['Gender'] == 'Other'].index
        assert len(unknown_gender_index) == 1, "Test setup error: Expected one row with 'Other' Gender"
        # Check that all OHE columns derived from 'Gender' are 0 for this row
        assert X_test_t.loc[unknown_gender_index, ohe_gender_cols].sum(axis=1).iloc[0] == 0, \
            f"Expected all-zero OHE Gender columns for unknown 'Other', got: {X_test_t.loc[unknown_gender_index, ohe_gender_cols]}"


        # Ordinal (Job Title): 'handle_unknown=use_encoded_value', unknown_value=-1
        unknown_job_index = X_test[X_test['Job Title'] == 'UnknownJob'].index
        assert len(unknown_job_index) == 1, "Test setup error: Expected one row with 'UnknownJob' Job Title"
        # The output column name for the ordinal encoded feature is just 'Job Title'
        assert X_test_t.loc[unknown_job_index, 'Job Title'].iloc[0] == -1, \
             f"Expected -1 for unknown 'UnknownJob', got: {X_test_t.loc[unknown_job_index, 'Job Title'].iloc[0]}"


    def test_preprocessor_structure(self, sample_data_for_preprocessing, data_processing_parameters):
        """Verify the structure of the created ColumnTransformer."""
        X_train, X_test = sample_data_for_preprocessing
         # Pass copies
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        X_train_t, X_test_t, preprocessor = build_preprocessing_pipeline(X_train_orig, X_test_orig, data_processing_parameters)


        assert 'num' in preprocessor.named_transformers_
        assert 'low_cat' in preprocessor.named_transformers_
        assert 'high_cat' in preprocessor.named_transformers_

        # Check steps within pipelines
        num_pipeline = preprocessor.named_transformers_['num']
        assert isinstance(num_pipeline, Pipeline)
        assert 'imputer' in num_pipeline.named_steps and isinstance(num_pipeline.named_steps['imputer'], SimpleImputer)
        assert 'scaler' in num_pipeline.named_steps and isinstance(num_pipeline.named_steps['scaler'], StandardScaler)
        assert num_pipeline.named_steps['imputer'].strategy == 'median'

        low_cat_pipeline = preprocessor.named_transformers_['low_cat']
        assert isinstance(low_cat_pipeline, Pipeline)
        assert 'imputer' in low_cat_pipeline.named_steps and isinstance(low_cat_pipeline.named_steps['imputer'], SimpleImputer)
        assert 'onehot' in low_cat_pipeline.named_steps and isinstance(low_cat_pipeline.named_steps['onehot'], OneHotEncoder)
        assert low_cat_pipeline.named_steps['imputer'].strategy == 'most_frequent'
        assert low_cat_pipeline.named_steps['onehot'].handle_unknown == 'ignore'
        assert low_cat_pipeline.named_steps['onehot'].drop == 'first'

        high_cat_pipeline = preprocessor.named_transformers_['high_cat']
        assert isinstance(high_cat_pipeline, Pipeline)
        assert 'imputer' in high_cat_pipeline.named_steps and isinstance(high_cat_pipeline.named_steps['imputer'], SimpleImputer)
        assert 'ordinal' in high_cat_pipeline.named_steps and isinstance(high_cat_pipeline.named_steps['ordinal'], OrdinalEncoder)
        assert high_cat_pipeline.named_steps['imputer'].strategy == 'most_frequent'
        assert high_cat_pipeline.named_steps['ordinal'].handle_unknown == 'use_encoded_value'
        assert high_cat_pipeline.named_steps['ordinal'].unknown_value == -1

        # Check columns assigned to each transformer based on revised logic
        # Access the fitted ColumnTransformer's transformers_ attribute which stores (name, transformer, columns)
        # Note: This attribute shows the state *after* fitting.
        assigned_cols = {name: cols for name, trans, cols in preprocessor.transformers_}

        # Use lists for consistent comparison regardless of original type (e.g., index vs list)
        assert list(assigned_cols.get('num')) == ["Age", "Years of Experience"]
        assert list(assigned_cols.get('low_cat')) == ["Gender", "Education Level"]
        assert list(assigned_cols.get('high_cat')) == ["Job Title"]