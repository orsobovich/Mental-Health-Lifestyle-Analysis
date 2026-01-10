from pathlib import Path
import pandas as pd
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)
# Configure logger as per instructions 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(filename):
    """
    Loads a dataset from a given filename or absolute path.
    If a relative filename is provided, the file is searched for
    in the project root directory.
    """
    try:
        logging.info("Starting dataset loading")

        base_dir = Path(__file__).resolve().parents[1]
        data_path = Path(filename)

        if not data_path.is_absolute():
            data_path = base_dir / filename

        logging.info(f"Resolved dataset path: {data_path}")

        df = pd.read_csv(data_path)

        logging.info("Dataset loaded successfully")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise FileNotFoundError(f"File not found: {data_path}")

    except Exception as e:
        logging.error(f"Unexpected error while loading dataset: {e}")
        raise RuntimeError(f"Error loading dataset: {e}")


def overview(df, n=5):
    """
    Return a basic overview of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    n : int
        Number of rows to display from the top.

    Returns
    -------
    dict
        Dictionary containing dataset overview information.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    overview_dict = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "head": df.head(n)
    }

    return overview_dict


def handle_missing_values_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hybrid cleaning function:
    1. Numeric columns: Fills missing values with the column mean.
    2. Non-numeric columns: Drops rows with missing values.
    
    Args:
        df (pd.DataFrame): The input dataset.
        
    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    try:
        initial_count = len(df)
        
        # 1. Pre-processing: Convert empty strings/whitespace to NaN
        # This ensures that " " is treated as missing data, not as a valid string value.
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # 2. Identify numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns

        # 3. Handle numeric columns: Fill with mean
        # Calculate the mean for each numeric column
        means = df[numeric_cols].mean()
        # Fill NaN values in numeric columns with their respective means
        df[numeric_cols] = df[numeric_cols].fillna(means)
        logger.info(f"Filled NaNs in numeric columns {list(numeric_cols)} with their means.")

        # 4. Handle non-numeric columns: Drop rows
        # We drop rows only if the missing value is in a categorical/text column
        rows_before_drop = len(df)
        df = df.dropna(subset=non_numeric_cols)
        
        dropped_count = rows_before_drop - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows due to missing values in non-numeric columns.")

        return df

    except Exception as e:
        logger.error(f"Error in handle_missing_values_hybrid: {e}")
        raise e
    


    """
    Detects potential outliers using Z-scores in numeric columns.
    Returns a Series counting the number of outliers per column.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        threshold (float): The Z-score threshold (default is 3).
        
    Returns:
        pd.Series: A count of outliers for each numeric column.
    """
    try:
        # 1. Select numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        outliers_count = {}

        # 2. Iterate and calculate Z-scores
        for col in num_cols:
            s = df[col].dropna()
            
            if s.std() == 0:
                outliers_count[col] = 0
                continue

            # Calculate Z-score
            z_scores = (s - s.mean()) / s.std()
            
            # Count outliers
            count = (np.abs(z_scores) > threshold).sum()
            outliers_count[col] = count
            
            if count > 0:
                logger.info(f"Column '{col}' has {count} outliers (Z-score > {threshold}).")

        return pd.Series(outliers_count).sort_values(ascending=False)

    except Exception as e:
        logger.error(f"Error in detect_outliers: {e}")
        raise e
    
    
def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Removes rows containing outliers in any numeric column using Z-score.
    Standard threshold is 3 (values beyond 3 standard deviations are removed).
    
    Args:
        df(pd.DataFrame): The input dataset.
        threshold (float): Z-score threshold (default is 3.0).
        
    Returns:
        pd.DataFrame: The dataset with outliers removed.
    """
    try:
        # 1. Save original row count for logging purposes
        initial_count = len(df)

        # 2. Identify numeric columns only (Z-score applies only to numbers)
        numeric_cols = df.select_dtypes(include=['number']).columns

        # 3. Calculate Z-scores for all numeric columns at once
        # Formula: (Value - Mean) / Standard Deviation
        z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

        # 4. Keep rows where ALL numeric values are within the threshold
        # We check if absolute Z-score <= threshold
        mask = (np.abs(z_scores) <= threshold).all(axis=1)
        
        # 5. Filter the dataframe
        df_clean = df[mask]

        # Log the number of dropped rows
        dropped_count = initial_count - len(df_clean)
        if dropped_count > 0:
            logger.info(f"Removed {dropped_count} rows with outliers (Z-score > {threshold}).")
        
        return df_clean

    except Exception as e:
        logger.error(f"Error in remove_outliers: {e}")
        raise e
    
#change .   
    
def is_valid_level(series: pd.Series) -> bool:
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
        
        if not is_valid:
            logger.debug(f"Column contains values outside {ordinal_levels}: {unique_values - ordinal_levels}")
            
        return is_valid

    except Exception as e:
        logger.error(f"Error in is_valid_level: {e}")
        return False


def level_to_numeric(series: pd.Series) -> pd.Series:
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