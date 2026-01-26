import pytest
import pandas as pd
import numpy as np
from src.utils import find_sig
from src.correlation import calculate_correlation, is_valid_level, level_to_numeric

# NOTE: In this test file, synthetic data is created INSIDE each test function.


# Verify that a Series containing ordinal levels (Low, Moderate, High) is recognized as valid
def test_on_valid_levels_should_return_true():
    test_1=pd.Series(['Low', 'Moderate', 'High', 'Moderate', 'Low'], name="satisfaction")
    assert is_valid_level(test_1) == True


# Verify that a Series containing numeric data returns False, as these are not ordinal levels
def test_on_invalid_levels_should_return_false():
    test_2=pd.Series([25, 30, 35, 40, 45], name="age")
    assert is_valid_level(test_2) == False


# Verify that passing a list (instead of a pandas Series) returns False due to type mismatch
def test_on_invalid_levels_get_Exception_should_return_false():
    test_3=not_a_series = ["Low", "High"] #unique does not except lists
    assert is_valid_level(test_3)== False


# Verify that the presence of missing values (None/NaN) does not invalidate the check for ordinal levels
def test_on_valid_with_nan_should_return_true():
    test_4=pd.Series(['Low', None, 'Moderate', 'High', None, 'Moderate', 'Low', None], name="satisfaction")
    assert is_valid_level(test_4) == True




# Verify that the function correctly maps ordinal strings ('Low', 'Moderate', 'High') to their numeric values (1, 2, 3)
def test_on_valid_levels_should_return_correct_numbers():
    test_5=pd.Series(["Low","Moderate","High"], name="satisfaction")
    assert level_to_numeric(test_5).to_list() == [1,2,3]


# Verify that the function raises an AttributeError when it gets an integer, as it lacks pandas Series methods
def test_level_to_numeric_should_crash_on_invalid_input():
    invalid_input = 12345
#The function won't raise an exception when it gets AttributeError
    with pytest.raises(AttributeError):
        level_to_numeric(invalid_input)




# Verify that a p-value less than (0.05) returns True
def test_on_significant_pvalue_should_return_true():
    assert find_sig(0.04) == True


# Verify that a p-value greater than the threshold returns False
def test_on_insignificant_pvalue_should_return_false():
    assert find_sig(1) == False




# Test that calculating correlation between two numeric Series (Pearson) runs without errors
def test_correlation_pearson_on_both_numeric_should_not_raise_exception():
    var1 = pd.Series([1, 2, 3, 4, 5])
    var2 = pd.Series([2, 4, 6, 8, 10])
    try:
        calculate_correlation(var1, var2)
    except Exception as e:
        pytest.fail(f"Pearson correlation failed unexpectedly: {e}")        




# Test that calculating correlation between Numeric and Ordinal (Spearman) runs without errors
def test_correlation_spearman_on_one_numeric_one_ordinal_should_not_raise_exception():
    var_numeric = pd.Series([1, 2, 3])
    var_ordinal = pd.Series(['Low', 'Moderate', 'High'])
   
    try:
        calculate_correlation(var_numeric, var_ordinal)
        calculate_correlation(var_ordinal, var_numeric)
    except Exception as e:
        pytest.fail(f"Spearman correlation failed unexpectedly: {e}")
   
   
 # Test that calculating correlation between two ordinal Series (Spearman) runs without errors
def test_correlation_spearman_on_both_ordinal_should_not_raise_exception():
    var_ordinal_1 = pd.Series(['Low', 'Moderate', 'High'])
    var_ordinal_2 = pd.Series(['Low', 'Moderate', 'High'])
   
    try:
        calculate_correlation(var_ordinal_1, var_ordinal_2)
        calculate_correlation(var_ordinal_2, var_ordinal_1)
    except Exception as e:
        pytest.fail(f"Spearman correlation failed unexpectedly: {e}")
   


# Test that calculating correlation between two categorical (Cramer's V) variables runs without errors
def test_correlation_cramers_should_not_raise_exception():      
    var_name = pd.Series(["ore", "elis", "orian"])
    var_animal = pd.Series(["dog", "cat", "bird"])
   
    try:
        calculate_correlation(var_name, var_animal)
    except Exception as e:
        pytest.fail(f"Cramer's correlation failed unexpectedly: {e}")
       


# Verify that attempting to correlate incompatible types (Numeric and Nominal) raises a TypeError
def test_correlation_on_invalid_input_should_raise_exception():
   
    var_numeric = pd.Series([1, 2, 3])
    var_invalid = pd.Series(['Apple', 'Banana', 'Orange'])
    # The function won't raise an exception when it gets TypeError
    with pytest.raises(TypeError):
        calculate_correlation(var_numeric, var_invalid)
       
 
# Verify that attempting to correlate Series of different lengths raises a ValueError.      
def test_correlation_on_different_lengths_should_raise_value_error():
    var1 = pd.Series([1, 2, 3])  
    var2 = pd.Series([1, 2])      
    # The function won't raise an exception when it gets ValueError
    with pytest.raises(ValueError):
        calculate_correlation(var1, var2)