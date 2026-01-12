from pathlib import Path
from typing import Dict, Tuple
from src.data_cleaning import get_column_types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)



def data_info(df: pd.DataFrame):
    """
    Creates a summary table with basic information for each column in the DataFrame.


    For every column, the table includes:
    - dtype: the data type of the column (e.g., int, float, object)
    - missing_count: how many missing (NaN) values exist in the column
    - missing_percent: percentage of missing values relative to the total number of rows
    - unique_values: number of unique (non-missing) values in the column


    The resulting table is sorted by missing_percent in descending order,
    so columns with the most missing data appear first.
    """
    try:
        # Validate input: the function expects a non-empty pandas DataFrame
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")


        # Build a DataFrame where each row represents one column from the original dataset
        info = pd.DataFrame({
            # Convert data types to string for readability (e.g., 'int64' instead of dtype object)
            "dtype": df.dtypes.astype(str),


            # Count how many NaN values exist in each column
            "missing_count": df.isna().sum(),


            # Calculate the percentage of missing values per column
            # df.isna().mean() gives the proportion of NaNs, multiplied by 100 for percentage
            "missing_percent": (df.isna().mean() * 100).round(2),


            # Count the number of unique non-missing values in each column
            "unique_values": df.nunique(dropna=True),
        })


        # Sort columns so that variables with the highest percentage of missing values appear first
        info = info.sort_values("missing_percent", ascending=False)


        # Log successful creation of the summary table
        logger.info("Data info table created successfully")


        # Return the summary DataFrame for further analysis or testing
        return info


    except Exception as e:
        # Log the error 
        logger.error("Error in data_info: %s", e)
        raise e



def descriptive_stats(df: pd.DataFrame):
    """
    Computes descriptive statistics separately for numeric and categorical variables.
    Returns two DataFrames: numeric_stats and categorical_stats.
    """
    try:
        # Validate input DataFrame
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")


        # Split columns by type: numeric vs categorical
        numeric_cols, non_numeric_cols = get_column_types(df)


        # -------- Numeric variables --------
        if len(numeric_cols) > 0:
            # Basic statistics: count, mean, std, min, quartiles, max
            numeric_stats = df[numeric_cols].describe().T

        else:
            # No numeric columns found
            numeric_stats = pd.DataFrame()


        # -------- Categorical variables --------
        if len(non_numeric_cols) > 0:
            # Frequency-based summary: count, unique, top, freq
            categorical_stats = df[non_numeric_cols].describe(include="all").T
        else:
            # No categorical columns found
            categorical_stats = pd.DataFrame()


        # Log successful execution
        logger.info(
            "Descriptive stats computed (numeric=%d, categorical=%d)",
            len(numeric_cols), len(non_numeric_cols)
        )


        # Return both summary tables
        return numeric_stats, categorical_stats


    except Exception as e:
        # Log and re-raise error for pytest or caller
        logger.error("Error in descriptive_stats: %s", e)
        raise e



def categorical_frequencies(df: pd.DataFrame, top_n: int = 10, add_other: bool = True):
    """
    Creates frequency tables for categorical variables.
    Returns {column_name -> DataFrame with counts}.
    Optionally adds an 'Other' row for categories outside Top N.
    """
    try:
        # --- 1. Input Validation ---
        # Ensure the DataFrame is valid and contains data
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
       
        # Ensure top_n is a valid positive integer
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer")


        # --- 2. Identify Categorical Columns ---
        # Get list of non-numeric (categorical) columns using helper function
        _, non_numeric_cols = get_column_types(df)
       
        # Dictionary to store the results
        result: Dict[str, pd.DataFrame] = {}


        # --- 3. Process Each Column ---
        for col in non_numeric_cols:
            # Count frequency of each value (include NaN values with dropna=False)
            # Convert to object type to handle mixed types safely
            vc_full = df[col].astype("object").value_counts(dropna=False)
           
            # Select only the top N most frequent categories
            vc_top = vc_full.head(top_n)


            # --- 4. Handle "Other" Category ---
            # If requested AND there are more categories than top_n, group the rest as "Other"
            if add_other and len(vc_full) > top_n:
                # Sum the counts of all remaining categories (from index top_n onwards)
                other_count = int(vc_full.iloc[top_n:].sum())
               
                # Append the "Other" row to the selected top categories
                vc_top = pd.concat([vc_top, pd.Series({"Other": other_count})])


            # --- 5. Format Output ---
            # Convert Series to DataFrame and store it in the result dictionary
            result[col] = vc_top.to_frame(name="count")

        # Log success
        logger.info("Categorical frequencies created successfully (top_n=%d, add_other=%s)", top_n, add_other)
        return result


    except Exception as e:
        # Log any errors encountered
        logger.error("Error in categorical_frequencies: %s", e)
        raise e



    except Exception as e:
        # Log error and re-raise for pytest or caller
        logger.error("Error in categorical_frequencies: %s", e)
        raise e



def numeric_ranges(df: pd.DataFrame):
    """
    Computes basic statistics for numeric variables:
    min, max, mean, standard deviation, and median.
    """
    try:
        # Validate input DataFrame
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")


        # Get only numeric columns
        numeric_cols, _ = get_column_types(df)


        # If no numeric columns exist, return empty result
        if len(numeric_cols) == 0:
            logger.info("numeric_ranges: no numeric columns found")
            return pd.DataFrame()


        # Compute basic statistics and transpose for readability
        stats = df[numeric_cols].agg(
            ["min", "max", "mean", "std", "median"]
        ).T


        # Log successful computation
        logger.info("Numeric ranges computed successfully")


        # Return statistics table
        return stats


    except Exception as e:
        # Log error and re-raise
        logger.error("Error in numeric_ranges: %s", e)
        raise e