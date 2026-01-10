from data_cleaning import is_valid_level, level_to_numeric
import pandas as pd
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr, chi2_contingency

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
             (pd.api.types.is_numeric_dtype(var_2) and is_valid_level(var_1)):
            
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
        raise

    except Exception as e: # catch all the exception except ValueError
        logging.exception("Unexpected error in correlation")
        raise
        
    return var_1, var_2, corr, p_value