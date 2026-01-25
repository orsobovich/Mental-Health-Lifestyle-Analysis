import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import (
    get_column_types,
    handle_missing_values_hybrid,
    remove_outliers,
    remove_duplicates
)

# NOTE: In this test file, synthetic data is created OUTSIDE the test functions # using @pytest.fixture.

# ----------------------------------------------------------------
# Fixtures (Data Setup)
# ----------------------------------------------------------------

@pytest.fixture
def dirty_df():
    """
    DataFrame with various issues:
    - Numeric NaNs (should be filled with mean)
    - Categorical NaNs (row should be dropped)
    - Whitespace strings in category (should be treated as NaN and dropped)
    """
    return pd.DataFrame({
        'Age': [20, 30, np.nan, 40, 50],       # Mean = 35 (excluding NaN)
        'Salary': [2000, 3000, 4000, 5000, 6000],
        'City': ['NY', 'LA', 'NY', None, ' '], # Index 3 is None, Index 4 is whitespace
        'Gender': ['M', 'F', 'M', 'F', 'M']
    })


@pytest.fixture
def outlier_df():
    """
    DataFrame specifically for outlier detection.
    Values: [10, 10, 10, 10, 1000]
    1000 is a clear outlier compared to the rest.
    """
    return pd.DataFrame({
        'Value': [10, 10, 10, 10, 1000],
        'Group': ['A', 'A', 'A', 'A', 'A']
    })
    
@pytest.fixture
def duplicate_df():
    """
    DataFrame specifically for testing duplicate removal.
    Contains:
    - 2 identical rows (should become 1)
    - 1 unique row
    Total rows: 3 -> Expected after cleaning: 2
    """
    return pd.DataFrame({
        'ID': [1, 1, 2],
        'Name': ['Alice', 'Alice', 'Bob'],
        'Score': [100, 100, 90]
    })


@pytest.fixture
def empty_df():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()




# ----------------------------------------------------------------
# 1. Tests for: get_column_types
# ----------------------------------------------------------------


def test_get_column_types(dirty_df):
    """Verifies correct separation of numeric and categorical columns."""
    numeric, non_numeric = get_column_types(dirty_df)
   
    # Check numeric columns
    assert 'Age' in numeric
    assert 'Salary' in numeric
    assert 'City' not in numeric
   
    # Check categorical columns
    assert 'City' in non_numeric
    assert 'Gender' in non_numeric
    assert 'Age' not in non_numeric


def test_get_column_types_empty(empty_df):
    """Edge case: Ensure it handles empty DF without crashing and returns empty lists."""
    numeric, non_numeric = get_column_types(empty_df)
    assert len(numeric) == 0, "Numeric list should be empty for empty DF"
    assert len(non_numeric) == 0, "Non-numeric list should be empty for empty DF"


   
    assert len(numeric) == 0
    assert len(non_numeric) == 0
def test_get_column_types_all_numeric():
    """Edge case: DataFrame with only numeric columns."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3.5, 4.5]})
    numeric, non_numeric = get_column_types(df)
   
    assert set(numeric) == {'A', 'B'}
    assert len(non_numeric) == 0
   
def test_get_column_types_all_categorical():
    """Edge case: DataFrame with only categorical columns."""
    df = pd.DataFrame({'A': ['x', 'y'], 'B': ['foo', 'bar']})
    numeric, non_numeric = get_column_types(df)
   
    assert len(numeric) == 0
    assert set(non_numeric) == {'A', 'B'}


# ----------------------------------------------------------------
# 2. Tests for: handle_missing_values_hybrid
# ----------------------------------------------------------------


def test_hybrid_fill_numeric(dirty_df):
    """Test that numeric NaNs are filled with the mean."""
    # Before: Age has NaN at index 2. Expected Mean = (20+30+40+50)/4 = 35
    cleaned_df = handle_missing_values_hybrid(dirty_df.copy())
   
    # Note: Index 3 and 4 will be dropped due to 'City' issues,
    # but Index 2 (where Age was NaN) has valid City ('NY'), so it should remain.
    assert cleaned_df.loc[2, 'Age'] == 35.0
    assert cleaned_df['Age'].isnull().sum() == 0


def test_hybrid_drop_categorical(dirty_df):
    """Test that rows with categorical NaNs or Whitespace are dropped."""
    # Index 3 is None in 'City' -> Should be dropped
    # Index 4 is " " in 'City' -> Should be converted to NaN and dropped
    cleaned_df = handle_missing_values_hybrid(dirty_df.copy())
   
    assert 3 not in cleaned_df.index
    assert 4 not in cleaned_df.index
    assert len(cleaned_df) == 3  # Original 5 - 2 dropped


def test_hybrid_empty_df(empty_df):
    """Edge case: Ensure it handles empty DF without crashing."""
    try:
        res = handle_missing_values_hybrid(empty_df)
        assert res.empty
    except Exception as e:
        pytest.fail(f"Function crashed on empty DataFrame: {e}")


def test_hybrid_no_missing():
    """Test that DataFrame without missing values remains unchanged."""
    df = pd.DataFrame({
        'Num': [1, 2, 3],
        'Cat': ['a', 'b', 'c']
    })
    cleaned_df = handle_missing_values_hybrid(df.copy())
   
    pd.testing.assert_frame_equal(df, cleaned_df)


def test_remove_outliers_detection(outlier_df):
    """Test that extreme values are removed based on Z-score."""
    # The value 1000 is massive compared to 10s, so Z-score will be > 3 (assuming small N correction doesn't skew it too much)
    # Let's verify manually:
    # Mean = 208, Std ~ 442.
    # (1000 - 208) / 442 = ~1.79.
    # Wait, with small N=5, std is huge. 1000 might NOT be > 3 std devs depending on calculation.
    # Let's use a clearer case for Z-score > 3.
   
    data = pd.DataFrame({'Val': [10]*50 + [1000]}) # 50 tens and one 1000
    # This ensures 1000 is definitely statistically far.
   
    cleaned_df = remove_outliers(data, threshold=3.0)
   
    # The outlier (1000) should be removed
    assert len(cleaned_df) == 50
    assert cleaned_df['Val'].max() == 10


def test_remove_outliers_no_outliers():
    """Test that data remains unchanged if no outliers exist."""
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    cleaned_df = remove_outliers(df)
   
    assert len(cleaned_df) == 5
    pd.testing.assert_frame_equal(df, cleaned_df)


def test_remove_outliers_threshold_param():
    """Test that changing the threshold affects strictness."""
    df = pd.DataFrame({'A': [10, 10, 10, 20]})
    # Z-scores roughly: 10s are -0.86, 20 is +1.5 (approx).
   
    # Threshold 1.0 should likely remove the 20
    cleaned_strict = remove_outliers(df.copy(), threshold=1.0)
    assert len(cleaned_strict) < 4 # Expect removal
   
    # Threshold 10.0 should keep everything
    cleaned_loose = remove_outliers(df.copy(), threshold=10.0)
    assert len(cleaned_loose) == 4 # Expect keep all


def test_remove_outliers_no_numeric_cols():
    """Edge case: DataFrame with only strings."""
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    # Should run without error and return original
    cleaned = remove_outliers(df)
    pd.testing.assert_frame_equal(df, cleaned)
def test_remove_outliers_empty(empty_df):
    """Edge case: Ensure it handles empty DF without crashing."""
    try:
        res = remove_outliers(empty_df)
        assert res.empty
    except Exception as e:
        pytest.fail(f"Function crashed on empty DataFrame: {e}")


# ----------------------------------------------------------------
# 3. Tests for: remove_duplicates
# ----------------------------------------------------------------


@pytest.fixture
def duplicate_df():
    """
    DataFrame specifically for testing duplicate removal.
    Contains:
    - 2 identical rows (should become 1)
    - 1 unique row
    Total rows: 3 -> Expected after cleaning: 2
    """
    return pd.DataFrame({
        'ID': [1, 1, 2],
        'Name': ['Alice', 'Alice', 'Bob'],
        'Score': [100, 100, 90]
    })


# ----------------------------------------------------------------
# Tests for: remove_duplicates
# ----------------------------------------------------------------


def test_remove_duplicates_found(duplicate_df):
    """Test that exact duplicates are removed correctly."""
    # Before: 3 rows (2 are identical)
    cleaned_df = remove_duplicates(duplicate_df.copy())
    
    # After: Should have 2 rows
    assert len(cleaned_df) == 2
    
    # Verify indices or values are unique
    assert cleaned_df.duplicated().sum() == 0


def test_remove_duplicates_none():
    """Test that a DataFrame with no duplicates remains unchanged."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    
    cleaned_df = remove_duplicates(df.copy())
    
    # Check that nothing changed
    pd.testing.assert_frame_equal(df, cleaned_df)


def test_remove_duplicates_partial_match():
    """
    Test that rows differing in only one column are NOT removed.
    (drop_duplicates checks ALL columns by default)
    """
    df = pd.DataFrame({
        'ID': [1, 1],          # Same ID
        'Value': [10, 20]      # Different Value
    })
    
    cleaned_df = remove_duplicates(df.copy())
    
    # Both rows should remain because they are not identical
    assert len(cleaned_df) == 2


def test_remove_duplicates_empty(empty_df):
    """Edge case: Ensure it handles empty DF without crashing."""
    try:
        res = remove_duplicates(empty_df)
        assert res.empty
    except Exception as e:
        pytest.fail(f"Function crashed on empty DataFrame: {e}")


def test_remove_duplicates_all_identical():
    """Edge case: DataFrame where ALL rows are exactly the same."""
    df = pd.DataFrame({'A': [5, 5, 5, 5, 5]})
    
    cleaned_df = remove_duplicates(df.copy())
    
    # Should result in exactly 1 row
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]['A'] == 5