import pandas as pd
import numpy as np
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the dataset.
    Logs the number of duplicates found and removed.
    """
    try:
        # Capture the initial row count
        initial_count = len(df)


        # Drop rows where all columns are identical (keeps the first occurrence by default)
        df = df.drop_duplicates()


        # Calculate exactly how many rows were removed
        dropped_count = initial_count - len(df)
       
        # Log how many rows were removed for data integrity tracking
        if dropped_count > 0:
            logger.info(f"Removed {dropped_count} duplicate rows.")
        else:
            logger.info("No duplicate rows found.")
       
        # Return the cleaned DataFrame
        return df
       
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Error removing duplicates: {e}")
        raise e


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

        # Check if there are any NaNs in numeric columns before filling
        # .any().any() checks if there is at least one True in the whole subset
        if df[numeric_cols].isna().any().any():
            # Fill missing values in numeric columns with the column mean
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            # Log only if values were actually filled
            logger.info(f"Filled NaNs in numeric columns {list(numeric_cols)} with their means.")
        else:
            # Log if no numeric NaNs were found
            logger.info("No missing values found in numeric columns.")


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
        # 1. Identify numeric columns for Z-score calculation
        numeric_cols, _ = get_column_types(df)
       
        # 2. Calculate Z-scores for all numeric columns at once (Vectorized operation)
        # Formula: Z = (Value - Mean) / Standard Deviation
        z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
       
        # 3. Reporting Logic: Iterate through columns to identify which ones contain outliers
        logger.info(f"--- Outlier Detection Report (Threshold > {threshold}) ---")
       
        # Loop over each numeric column name (e.g., 'Age', 'Salary') individually
        for col in numeric_cols:


            # Extract the data for the CURRENT column from the 'z_scores' DataFrame
            # Count how many rows in this specific column exceed the threshold
            col_outliers = (np.abs(z_scores[col]) > threshold).sum()
           
            # Log the findings ONLY if outliers exist in this specific column
            if col_outliers > 0:
                logger.info(f"Column '{col}': found {col_outliers} outliers.")


        # 4. Create a boolean mask to filter rows
        # 'all(axis=1)' ensures we keep a row ONLY if ALL its numeric values are within the threshold
        mask = (np.abs(z_scores) <= threshold).all(axis=1)
       
        # 5. Apply the filter to subset the DataFrame
        initial_count = len(df)
        df_clean = df[mask]


        # Calculate the removed rows
        removed_count = initial_count - len(df_clean)
       
        # Log the final result
        if removed_count > 0:
            logger.info(f"Total rows removed due to outliers: {removed_count}")
        else:
            logger.info("No outliers detected in the dataset.")
       
        # Return the cleaned dataset without outliers
        return df_clean
   
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Error removing outliers: {e}")
        raise e