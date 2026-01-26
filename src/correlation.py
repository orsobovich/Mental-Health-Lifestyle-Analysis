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
        # Case 1: Both variables are numeric -> Calculate Pearson Correlation
        if pd.api.types.is_numeric_dtype(var_1) and pd.api.types.is_numeric_dtype(var_2):
            corr, p_value = pearsonr(var_1, var_2)


            # Log the results
            logging.info(f"Pearson correlation: {corr}")
           
        # Case 2: One is numeric AND the other is ordinal -> Calculate Spearman Correlation
        elif (pd.api.types.is_numeric_dtype(var_1) and is_valid_level(var_2)) or \
             (pd.api.types.is_numeric_dtype(var_2) and is_valid_level(var_1))or \
             (is_valid_level(var_1) and is_valid_level(var_2)):
           
            # Convert whichever variable is ordinal to numeric
            if is_valid_level(var_1):
                var_1 = level_to_numeric(var_1)
           
            if is_valid_level(var_2):
                var_2 = level_to_numeric(var_2)
               
            # Now that both are numeric -> run Spearman Correlation
            corr, p_value = spearmanr(var_1, var_2)


            # Log the results
            logging.info(f"Spearman correlation: {corr}")


        # Case 3: Both are categorical (Not numeric) -> Calculate Chi-Squared & Cramer's V
        elif not pd.api.types.is_numeric_dtype(var_1) and not pd.api.types.is_numeric_dtype(var_2):
            # Create contingency table
            crosstab = pd.crosstab(var_1, var_2)
           
            # Run a Chi-squared test
            chi2, p_value, _, _ = chi2_contingency(crosstab)
           
            # Calculate Cramer's V
            n = crosstab.sum().sum()    # Total sample size
            min_dim = min(crosstab.shape) - 1  # Minimum dimension minus 1
            corr = np.sqrt(chi2 / (n * min_dim)) # Calculate Cramer's V
           
            # Log the results
            logging.info(f"Chi-Squared (Cramer's V): {corr}")


         # Case 4: Fallback for unsupported or mixed types (e.g., Date vs Boolean)
        else:
            logging.error("Unsupported variable types for correlation")
            # Raise TypeError to alert aller that these specific types cannot be correlated
            raise TypeError("Invalid input types")


    except ValueError as e:
        # Catch specific math/value errors (e.g., arrays with different lengths, empty inputs)
        logging.error(f"Value error in correlation: {e}")
        raise e


    except Exception as e:
        # Catch any other unexpected errors
        logging.exception("Unexpected error in correlation")
        raise e
       
    # Return the variables and statistical results
    # Note: var_1 and var_2 are returned because they might have been converted to numeric (in Case 2)
    return var_1, var_2, corr, p_value


def is_valid_level(series):
    """
    Checks if a series contains only valid ordinal levels: 'Low', 'Moderate', 'High'.
    Returns True if all unique values (ignoring NaNs) are within this set.
    """
    try:
        # Define the allowed schema for ordinal data
        ordinal_levels = {"Low", "Moderate", "High"}


        # Create a set of unique values from the series, dropping NaNs first
        unique_values = set(series.dropna().unique())
       
        # Check if the observed unique values are a subset of the allowed levels
        is_valid = unique_values.issubset(ordinal_levels)
       
        if not is_valid:
            # Debugging: If validation fails, log the specific values that caused the mismatch.
            # Calculate the difference to identify the specific invalid values.
            logger.debug(f"Column contains values outside {ordinal_levels}: {unique_values - ordinal_levels}")


        # Return the boolean result (True if valid, False otherwise)  
        return is_valid


    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Error in is_valid_level: {e}")
        # If an error occurs, assume the data is invalid and return False
        return False


def level_to_numeric(series: pd.Series):
    """
    Maps categorical levels ('Low', 'Moderate', 'High') to numeric ranks (1, 2, 3).
    Useful for Spearman's rank correlation analysis.
    """
    try:
        # Define the mapping schema: Ordinal text -> Integer rank
        mapping = {"Low": 1, "Moderate": 2, "High": 3}


        # Apply the mapping to transform the series values to numbers
        # Note: Values not found in the dictionary would become NaN (though validation prevents this)
        mapped_series = series.map(mapping)
       
        # Log the successful conversion for tracking
        logger.info("Converted ordinal column to numeric ranks.")


        # Return the transformed numeric series
        return mapped_series


    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Error in level_to_numeric: {e}")
        raise e