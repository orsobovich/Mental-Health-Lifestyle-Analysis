from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.data_cleaning import get_column_types
from src.utils import setup_logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)


def data_info(df: pd.DataFrame):
    """
    Creates a summary table with basic information for each column in the DataFrame,
    and returns a general overview string of the dataset shape.


    Returns:
        tuple: (overview_string, info_dataframe)
    """
    try:
        # Validate input: the function expects a non-empty pandas DataFrame
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        # Build a DataFrame where each row represents one column from the original dataset
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "unique_values": df.nunique(dropna=True),
        })

       
        # Create the overview string
        overview = f'The number of participants is "{df.shape[0]}" and the number of variables is "{df.shape[1]}"'


        # Log successful creation
        logger.info("Data info table and overview created successfully")


        # Return both the string and the table
        return overview, info

    except Exception as e:
        logger.error("Error in data_info: %s", e)
        raise e


def descriptive_stats(df: pd.DataFrame):
    """
    Compute descriptive statistics for numeric and categorical variables separately.
    Returns two DataFrames:
    1. numeric_stats
    2. categorical_stats
    """
    try:
        # Validate input
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        # Identify numeric and non-numeric columns
        numeric_cols, non_numeric_cols = get_column_types(df)

        # ----- Numeric variables -----
        if len(numeric_cols) > 0:
            # Compute descriptive statistics (mean, std, min, quartiles, max)
            # 'count' is explicitly removed before transposing
            numeric_stats = df[numeric_cols].describe().drop('count').T
        else:
            # Handle case with no numeric columns
            numeric_stats = pd.DataFrame()

        # ----- Categorical variables -----
        if len(non_numeric_cols) > 0:
            # Compute frequency-based statistics (count, unique, top, freq)
            # To remove 'count' here as well, apply .drop('count') after describe()
            categorical_stats = df[non_numeric_cols].describe(include="all").T
        else:
            # Handle case with no categorical columns
            categorical_stats = pd.DataFrame()

        # Log successful execution with column counts
        logger.info(
            "Descriptive stats computed (numeric=%d, categorical=%d)",
            len(numeric_cols), len(non_numeric_cols)
        )

        # Return summary tables
        return numeric_stats, categorical_stats

    except Exception as e:
        # Log error and re-raise for external handling or testing
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
   