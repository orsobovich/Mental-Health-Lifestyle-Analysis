import pytest
import pandas as pd
import numpy as np

from src.ANOVA import (
    compute_group_means,
    run_one_way_anova,
    create_contrast_weights,
    run_planned_contrast
)

# ----------------------------------------------------------------
# Fixtures (Data Setup)
# ----------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """
    Creates a clean dataset for testing.
    Group A: High values (Mean = 10)
    Group B: Low values (Mean = 2)
    Group C: Medium values (Mean = 6)
    """
    data = {
        'Group': ['A']*10 + ['B']*10 + ['C']*10,
        'Value': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  # Mean 10
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2,          # Mean 2
                   6, 6, 6, 6, 6, 6, 6, 6, 6, 6]          # Mean 6
    }
    return pd.DataFrame(data)

@pytest.fixture
def messy_data():
    """
    Creates a dataset with spaces in column names to test robustness.
    """
    return pd.DataFrame({
        'Diet Type': ['Vegan', 'Keto', 'Vegan', 'Keto'],
        'Happiness Score': [10, 5, 9, 6]
    })

# ----------------------------------------------------------------
# 1. Tests for: compute_group_means
# ----------------------------------------------------------------

def test_compute_group_means_correctness(synthetic_data):
    """
    Verifies that means are calculated mathematically correctly.
    """
    means = compute_group_means(synthetic_data, 'Group', 'Value')
    
    assert means['A'] == 10.0
    assert means['B'] == 2.0
    assert means['C'] == 6.0

def test_compute_group_means_missing_column(synthetic_data):
    """
    Verifies that the function raises KeyError if a column is missing.
    """
    with pytest.raises(KeyError):
        compute_group_means(synthetic_data, 'NonExistent', 'Value')

# ----------------------------------------------------------------
# 2. Tests for: run_one_way_anova
# ----------------------------------------------------------------

def test_run_one_way_anova_success(synthetic_data):
    """
    Verifies that ANOVA returns a valid table and detects significance.
    """
    anova_table = run_one_way_anova(synthetic_data, 'Group', 'Value')
    
    # Check if result is a DataFrame and has expected ANOVA columns
    assert isinstance(anova_table, pd.DataFrame)
    assert 'F' in anova_table.columns
    assert 'PR(>F)' in anova_table.columns # This is the p-value column
    
    # Since groups are very different (10 vs 2 vs 6), p-value should be tiny
    p_value = anova_table['PR(>F)'].iloc[0]
    assert p_value < 0.05

def test_run_one_way_anova_spaces(messy_data):
    """
    Verifies that ANOVA works even with spaces in column names (Diet Type).
    """
    try:
        run_one_way_anova(messy_data, 'Diet Type', 'Happiness Score')
    except Exception as e:
        pytest.fail(f"ANOVA failed on columns with spaces: {e}")

# ----------------------------------------------------------------
# 3. Tests for: create_contrast_weights
# ----------------------------------------------------------------

def test_create_weights_logic():
    """
    Verifies that weights sum to zero and are assigned correctly.
    """
    pos = ['A', 'B']
    neg = ['C']
    
    weights = create_contrast_weights(pos, neg)
    
    # Logic: 1/2 for positives, -1/1 for negative
    assert weights['A'] == 0.5
    assert weights['C'] == -1.0
    assert sum(weights.values()) == pytest.approx(0, abs=1e-9)

def test_create_weights_empty_error():
    """
    Verifies error handling for empty lists.
    """
    with pytest.raises(ValueError, match="Groups lists cannot be empty"):
        create_contrast_weights([], ['A'])

# ----------------------------------------------------------------
# 4. Tests for: run_planned_contrast
# ----------------------------------------------------------------

def test_run_planned_contrast_significant(synthetic_data):
    """
    Verifies that the contrast detects a real difference.
    Hypothesis: Group A (10) > Group B (2).
    """
    weights = {'A': 1, 'B': -1}
    
    result = run_planned_contrast(
        synthetic_data, 
        group_col='Group', 
        value_col='Value', 
        contrast_weights=weights
    )
    
    assert result is not None
    assert result['t_statistic'] > 0  # Positive because A > B
    assert result['p_value'] < 0.05   # Significant difference

def test_run_planned_contrast_inverse(synthetic_data):
    """
    Verifies that reversing the hypothesis reverses the t-statistic sign.
    Hypothesis: Group B (2) > Group A (10) -> Should yield negative t.
    """
    # We expect B to be lower, but we assign it positive weight
    weights = {'B': 1, 'A': -1}
    
    result = run_planned_contrast(
        synthetic_data, 
        group_col='Group', 
        value_col='Value', 
        contrast_weights=weights
    )
    
    assert result['t_statistic'] < 0  # Negative t-value

def test_run_planned_contrast_missing_group_error(synthetic_data):
    """
    Verifies that the function catches if we try to contrast a group 
    that doesn't exist in the data.
    """
    weights = {'A': 1, 'GhostGroup': -1}
    
    with pytest.raises(KeyError, match="Groups missing from dataset"):
        run_planned_contrast(
            synthetic_data, 
            group_col='Group', 
            value_col='Value', 
            contrast_weights=weights
        )

def test_run_planned_contrast_robustness_spaces(messy_data):
    """
    Verifies that the contrast works with column names containing spaces.
    """
    weights = {'Vegan': 1, 'Keto': -1}
    
    result = run_planned_contrast(
        messy_data, 
        group_col='Diet Type', 
        value_col='Happiness Score', 
        contrast_weights=weights
    )
    
    assert result is not None
    assert 'p_value' in result