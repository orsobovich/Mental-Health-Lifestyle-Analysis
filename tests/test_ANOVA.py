import pytest
import pandas as pd
import numpy as np
from src.ANOVA import create_contrast_weights, run_planned_contrast
import statsmodels.api as sm
from statsmodels.formula.api import ols

#ELIS













#OR
# ----------------------------------------------------------------
# Tests for: create_contrast_weights
# ----------------------------------------------------------------

def test_create_weights_balance():
    """
    Verifies that weights are generated correctly and their sum is zero.
    """
    pos = ["Vegan", "Vegetarian"]
    neg = ["Keto", "Junk"]
    
    weights = create_contrast_weights(pos, neg)
    
    # Basic check: Ensure all groups are present
    assert len(weights) == 4
    
    # Check that positive groups have positive weights and vice versa
    assert weights["Vegan"] > 0
    assert weights["Keto"] < 0
    
    # Mathematical check: Sum of weights must be zero
    # Using pytest.approx due to potential floating-point precision issues
    assert sum(weights.values()) == pytest.approx(0, abs=1e-9)

def test_create_weights_uneven_groups():
    """
    Tests the case of uneven group sizes (e.g., 1 vs 2).
    """
    pos = ["GroupA"]
    neg = ["GroupB", "GroupC"]
    
    weights = create_contrast_weights(pos, neg)
    
    assert weights["GroupA"] == 1.0  # 1 / 1
    assert weights["GroupB"] == -0.5 # -1 / 2
    assert sum(weights.values()) == pytest.approx(0)

def test_create_weights_empty_input():
    """
    Verifies that the function raises a ValueError when an empty list is provided.
    """
    with pytest.raises(ValueError):
        create_contrast_weights([], ["GroupB"])

# ----------------------------------------------------------------
# Tests for: run_planned_contrast
# ----------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """
    Creates a synthetic dataset for testing purposes.
    Group A (Vegan) has high values, Group B (Junk) has low values.
    """
    data = {
        'Diet Type': ['Vegan']*10 + ['Junk']*10 + ['Keto']*10,
        'Happiness Score': [9, 9, 8, 9, 9, 8, 9, 9, 8, 9,  # High scores for Vegan
                            3, 2, 4, 3, 2, 3, 2, 4, 3, 2,  # Low scores for Junk
                            5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # Medium for Keto
    }
    return pd.DataFrame(data)

def test_run_planned_contrast_success(synthetic_data):
    """
    Tests a successful contrast execution: Vegan vs. Junk.
    We expect a statistically significant result (p < 0.05).
    """
    # 1. Define weights manually or via the helper function
    weights = {'Vegan': 1, 'Junk': -1}
    
    # 2. Run the function
    result = run_planned_contrast(
        synthetic_data, 
        group_col='Diet Type', 
        value_col='Happiness Score', 
        contrast_weights=weights
    )
    
    # 3. Assertions
    assert result is not None
    assert "t_statistic" in result
    assert "p_value" in result
    assert result["t_statistic"] > 0 # Positive t-score expected because Vegan (1) > Junk (-1)
    assert result["p_value"] < 0.05  # Should be clearly significant in this synthetic dataset

def test_run_planned_contrast_missing_group(synthetic_data):
    """
    Verifies that the function identifies a non-existent group in the data 
    and returns None (as handled in the exception block).
    """
    # 'Paleo' does not exist in the synthetic data
    bad_weights = {'Vegan': 1, 'Paleo': -1}
    
    result = run_planned_contrast(
        synthetic_data, 
        group_col='Diet Type', 
        value_col='Happiness Score', 
        contrast_weights=bad_weights
    )
    
    assert result is None

def test_run_planned_contrast_spaces_in_names(synthetic_data):
    """
    Tests that the function handles spaces in column names correctly 
    (validating the usage of Q() in the formula).
    """
    weights = {'Vegan': 0.5, 'Keto': 0.5, 'Junk': -1}
    
    result = run_planned_contrast(
        synthetic_data, 
        group_col='Diet Type',       # Column name with space
        value_col='Happiness Score', # Column name with space
        contrast_weights=weights
    )
    
    assert result is not None