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
    
    
def calculate_z_scores(df: pd.DataFrame, threshold: float = 3.0):
    """Helper to calculate Z-scores mask."""
    # Use the helper function to get only numeric columns
    numeric_cols, _ = get_column_types(df)
   
    # Calculate Z-scores for numeric columns
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
   
    # Create boolean mask where all z-scores are within threshold
    mask = (np.abs(z_scores) <= threshold).all(axis=1)
    return mask


def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Removes rows with outliers in any numeric column using Z-score.
       Standard threshold is 3 (values beyond 3 standard deviations are removed).
       
       Args:
       df(pd.DataFrame): The input dataset.
       threshold (float): Z-score threshold (default is 3.0).
       
        Returns:
        pd.DataFrame: The dataset with outliers removed.
    """

    try:
        # Store initial row count
        initial_count = len(df)
        # Get mask for valid rows (uses the helper internally)
        mask = calculate_z_scores(df, threshold)
        # Filter DataFrame to keep only non-outliers
        df_clean = df[mask]
       
        # Calculate number of removed rows
        removed = initial_count - len(df_clean)
        # Log if rows were removed
        if removed > 0:
            logger.info(f"Removed {removed} outlier rows.")
        return df_clean
   
    except Exception as e:
        # Log error and re-raise
        logger.error(f"Error removing outliers: {e}")
        raise e