import pandas as pd
import numpy as np
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)


def get_column_types(df: pd.DataFrame):
    """
    Helper function to split columns into numeric and categorical.
    Returns two index objects: numeric_cols, non_numeric_cols.
    """
    # Select columns with numeric data types
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Select columns that are not numeric (categorical/object)
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
   
    # Return both lists of columns
    return numeric_cols, non_numeric_cols


def handle_missing_values_hybrid(df: pd.DataFrame):
    """Fills numeric NaNs with mean, drops categorical NaNs."""
    try:
        # Replace empty strings or whitespace with actual NaN values
        df = df.replace(r'^\s*$', np.nan, regex=True)
       
        # Use the helper function to identify column types
        numeric_cols, non_numeric_cols = get_column_types(df)

        # Fill missing values in numeric columns with the column mean
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        logger.info(f"Filled NaNs in numeric columns {list(numeric_cols)} with their means.")

        # Handle non-numeric columns: store row count before dropping
        rows_before = len(df)
        # Drop rows where categorical columns have missing values
        df = df.dropna(subset=non_numeric_cols)
        # Calculate how many rows were dropped
        dropped = rows_before - len(df)
       
        # Log the operation result if rows were dropped
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows (categorical missing).")
        return df
   
    except Exception as e:
        # Log error and re-raise
        logger.error(f"Error in hybrid cleaning: {e}")
        raise e
    

def remove_outliers(df: pd.DataFrame, threshold: float = 3.0):
    """
    Removes rows containing outliers in any numeric column based on the Z-score method.
    Standard threshold is 3.0 (values beyond 3 standard deviations from the mean are removed).
    """
    try:
        # 1. Identify numeric columns
        numeric_cols, _ = get_column_types(df)
        
        # 2. Calculate Z-scores for all numeric columns at once
        z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        
        # 3. Reporting Logic: Check which columns contain outliers before filtering
        logger.info(f"--- Outlier Detection Report (Threshold > {threshold}) ---")
        
        for col in numeric_cols:
            # Count how many values in this specific column exceed the threshold
            col_outliers = (np.abs(z_scores[col]) > threshold).sum()
            
            if col_outliers > 0:
                logger.info(f"Column '{col}': found {col_outliers} outliers.")

        # 4. Create a boolean mask to filter rows
        # 'all(axis=1)' ensures we keep rows where ALL numeric values are within the threshold
        mask = (np.abs(z_scores) <= threshold).all(axis=1)
        
        # 5. Apply the filter
        initial_count = len(df)
        df_clean = df[mask]
        removed_count = initial_count - len(df_clean)
        
        # Log the final result
        if removed_count > 0:
            logger.info(f"Total rows removed due to outliers: {removed_count}")
        else:
            logger.info("No outliers detected in the dataset.")
            
        return df_clean
   
    except Exception as e:
        # Log the specific error and stop execution
        logger.error(f"Error removing outliers: {e}")
        raise e