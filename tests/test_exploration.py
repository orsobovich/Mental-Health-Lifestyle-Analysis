import pytest
import pandas as pd
import numpy as np
from src.exploration import (
    data_info,
    descriptive_stats,
    categorical_frequencies,
    numeric_ranges
)
import logging
import matplotlib.pyplot as plt

# NOTE: In this test file, synthetic data is created OUTSIDE the test functions # using @pytest.fixture.
# This allows the same dataset to be reused across # multiple tests (e.g., for both ANOVA and Planned Contrasts).

# ----------------------------------------------------------------
# Fixtures (Data Setup)
# ----------------------------------------------------------------
@pytest.fixture
def synthetic_df():
    """
    Creates a simple dataset with mixed types for testing.
    Numeric: 'Age' (10, 20, 30, 40, 50), 'Score' (1.5, 2.5, ...)
    Categorical: 'Color' (Red, Blue, Red...), 'City' (NY, LA...)
    """
    data = {
        'Age': [10, 20, 30, 40, 50],
        'Score': [1.5, 2.5, 3.5, 4.5, 5.5],
        'Color': ['Red', 'Blue', 'Red', 'Green', 'Red'],
        'City': ['NY', 'LA', 'NY', 'SF', 'Chicago']
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_df():
    """Creates an empty DataFrame to test error handling."""
    return pd.DataFrame()


# ----------------------------------------------------------------
# 1. Tests for: data_info
# ----------------------------------------------------------------


def test_data_info_structure(synthetic_df):
    """Verifies that data_info returns the correct tuple structure and shape info."""
    overview, info_df = data_info(synthetic_df)
   
    # Check overview string content
    assert "number of participants is \"5\"" in overview
    assert "number of variables is \"4\"" in overview
   
    # Check info DataFrame
    assert isinstance(info_df, pd.DataFrame)
    assert 'dtype' in info_df.columns
    assert 'unique_values' in info_df.columns
    assert info_df.loc['Color', 'unique_values'] == 3  # Red, Blue, Green


def test_data_info_empty(empty_df):
    """Verifies that data_info raises ValueError for empty DataFrame."""
    with pytest.raises(ValueError, match="DataFrame is empty or None"):
        data_info(empty_df)


# ----------------------------------------------------------------
# 2. Tests for: descriptive_stats
# ----------------------------------------------------------------


def test_descriptive_stats_separation(synthetic_df):
    """Verifies that numeric and categorical columns are separated correctly."""
    num_stats, cat_stats = descriptive_stats(synthetic_df)
   
    # Check numeric stats
    assert 'Age' in num_stats.index
    assert 'Score' in num_stats.index
    assert 'Color' not in num_stats.index
    assert num_stats.loc['Age', 'mean'] == 30.0
   
    # Check categorical stats
    assert 'Color' in cat_stats.index
    assert 'City' in cat_stats.index
    assert 'Age' not in cat_stats.index
    assert cat_stats.loc['Color', 'top'] == 'Red'


def test_descriptive_stats_empty(empty_df):
    """Verifies that descriptive_stats raises ValueError for empty DataFrame."""
    with pytest.raises(ValueError, match="DataFrame is empty or None"):
        descriptive_stats(empty_df)
def test_descriptive_stats_no_numeric():
    """Verifies behavior when no numeric columns exist."""
    df_str = pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']})
    num_stats, cat_stats = descriptive_stats(df_str)
   
    assert num_stats.empty
    assert 'A' in cat_stats.index
    assert 'B' in cat_stats.index
# ----------------------------------------------------------------
# 3. Tests for: categorical_frequencies
# ----------------------------------------------------------------


def test_categorical_frequencies_logic(synthetic_df):
    """Verifies frequency counting."""
    freq_dict = categorical_frequencies(synthetic_df, top_n=10, add_other=False)
   
    assert 'Color' in freq_dict
    assert 'City' in freq_dict
   
    # Check counts for 'Color' (Red should be 3)
    color_counts = freq_dict['Color']
    assert color_counts.loc['Red', 'count'] == 3
    assert color_counts.loc['Blue', 'count'] == 1


def test_categorical_frequencies_top_n_other():
    """Verifies 'Other' logic when categories exceed top_n."""
    # Create data with 5 unique letters
    df = pd.DataFrame({'Letter': ['A']*10 + ['B']*5 + ['C']*2 + ['D']*1 + ['E']*1})
   
    # Request top 2, expect A, B, and 'Other' (containing C, D, E sum = 4)
    freq_dict = categorical_frequencies(df, top_n=2, add_other=True)
    res = freq_dict['Letter']
   
    assert len(res) == 3 # A, B, Other
    assert 'Other' in res.index
    assert res.loc['Other', 'count'] == 4  # 2+1+1


def test_categorical_frequencies_empty(empty_df):
    """Verifies that categorical_frequencies raises ValueError for empty DataFrame."""
    with pytest.raises(ValueError, match="DataFrame is empty or None"):
        categorical_frequencies(empty_df)
   
def test_categorical_frequencies_invalid_top_n(synthetic_df):
    """Verifies that categorical_frequencies raises ValueError for invalid top_n."""
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        categorical_frequencies(synthetic_df, top_n=0)
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        categorical_frequencies(synthetic_df, top_n=-5)
# ----------------------------------------------------------------
# 4. Tests for: numeric_ranges
# ----------------------------------------------------------------


def test_numeric_ranges_values(synthetic_df):
    """Verifies min, max, mean calculations."""
    ranges = numeric_ranges(synthetic_df)
   
    assert 'Age' in ranges.index
    assert ranges.loc['Age', 'min'] == 10
    assert ranges.loc['Age', 'max'] == 50
    assert ranges.loc['Age', 'mean'] == 30
    assert ranges.loc['Age', 'median'] == 30


def test_numeric_ranges_no_numeric_cols():
    """Verifies behavior when no numeric columns exist."""
    df_str = pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']})
    ranges = numeric_ranges(df_str)
    assert ranges.empty


def test_numeric_ranges_empty(empty_df):
    """Verifies that numeric_ranges raises ValueError for empty DataFrame."""
    with pytest.raises(ValueError, match="DataFrame is empty or None"):
        numeric_ranges(empty_df)
