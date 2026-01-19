import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)

def calculate_correlation(var_1, var_2):
    """
    Calculates the correlation between two variables based on their data types.
    """
    try:
        # Case 1: Both variables are numeric -> Pearson Correlation
        if pd.api.types.is_numeric_dtype(var_1) and pd.api.types.is_numeric_dtype(var_2):
            corr, p_value = pearsonr(var_1, var_2)
            logging.info(f"Pearson correlation: {corr}")
           
        # Case 2: One is numeric AND the other is ordinal -> Spearman Correlation
        elif (pd.api.types.is_numeric_dtype(var_1) and is_valid_level(var_2)) or \
             (pd.api.types.is_numeric_dtype(var_2) and is_valid_level(var_1))or \
             (is_valid_level(var_1) and is_valid_level(var_2)):
           
            # Convert whichever variable is ordinal to numeric
            if is_valid_level(var_1):
                var_1 = level_to_numeric(var_1)
           
            if is_valid_level(var_2):
                var_2 = level_to_numeric(var_2)
               
            # Now both are numeric, run Spearman
            corr, p_value = spearmanr(var_1, var_2)
            logging.info(f"Spearman correlation: {corr}")


        # Case 3: Both are categorical (Not numeric) -> Chi-Squared & Cramer's V
        elif not pd.api.types.is_numeric_dtype(var_1) and not pd.api.types.is_numeric_dtype(var_2):
            # Create contingency table
            crosstab = pd.crosstab(var_1, var_2)
           
            # Chi-squared test
            chi2, p_value, _, _ = chi2_contingency(crosstab)
           
            # Calculate Cramer's V
            n = crosstab.sum().sum()    # Total sample size
            min_dim = min(crosstab.shape) - 1  # Minimum dimension minus 1
            corr = np.sqrt(chi2 / (n * min_dim)) # Calculate Cramer's V
           
            logging.info(f"Chi-Squared (Cramer's V): {corr}")


        # raise an exception so it gets caught by the general exception handler
        else:
            logging.error("Unsupported variable types for correlation")
            raise TypeError("Invalid input types")


    except ValueError as e:
        logging.error(f"Value error in correlation: {e}")
        raise e


    except Exception as e: # catch all the exception except ValueError
        logging.exception("Unexpected error in correlation")
        raise e
       
    return var_1, var_2, corr, p_value


def is_valid_level(series):
    """
    Checks if a series contains only valid ordinal levels: 'Low', 'Moderate', 'High'.
    Returns True if all unique values (ignoring NaNs) are within this set.
    """
    try:
        ordinal_levels = {"Low", "Moderate", "High"}
        # Create a set of unique values from the series, dropping NaNs first
        unique_values = set(series.dropna().unique())
       
        # Check if the unique values are a subset of the allowed levels
        is_valid = unique_values.issubset(ordinal_levels)
       
        if not is_valid: # help us to understad that the funqtion isn't valid_level without an error, will return the string that dosen't belong to valid_level
            logger.debug(f"Column contains values outside {ordinal_levels}: {unique_values - ordinal_levels}")
           
        return is_valid


    except Exception as e:
        logger.error(f"Error in is_valid_level: {e}")
        return False


def level_to_numeric(series: pd.Series):
    """
    Maps categorical levels ('Low', 'Moderate', 'High') to numeric ranks (1, 2, 3).
    Useful for Spearman's rank correlation analysis.
    """
    try:
        mapping = {"Low": 1, "Moderate": 2, "High": 3}
        # Perform the mapping
        mapped_series = series.map(mapping)
        
        logger.info("Converted ordinal column to numeric ranks.")
        return mapped_series

    except Exception as e:
        logger.error(f"Error in level_to_numeric: {e}")
        raise e