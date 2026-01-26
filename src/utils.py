import logging
from pathlib import Path
import pandas as pd


def setup_logging():
    """
    Sets up the logging configuration for the entire project.
    """
    # Configure the logging system with level INFO and a specific format including time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_dataset(filename):
    """
    Loads a dataset from a given filename or absolute path.
    Handles relative paths by resolving them against the project root.
    """
    try:
        # Log that the function has started
        logging.info("Starting dataset loading")


        # --- Path Resolution ---
        # Resolve project root dynamically.
        # '.parents[1]' assumes this script is located in a subdirectory (e.g., 'src/')
        base_dir = Path(__file__).resolve().parents[1]
        data_path = Path(filename)


        # Check if the provided path is relative (e.g., 'data/file.csv').
        # If so, join it with the base directory to create a full absolute path.
        if not data_path.is_absolute():
            data_path = base_dir / filename


        # Log the final resolved path to verify correct file location before loading
        logging.info(f"Resolved dataset path: {data_path}")


        # --- Custom NaN Handling ---
        # Define specific strings to be treated as NaN (missing values).
        # CRITICAL: We explicitly exclude the string 'None' from this list.
        # This ensures that 'None' (e.g., in 'Mental Health Condition') is read as a valid category, not a null value.
        missing_values = ["", " ", "NaN", "nan", "NA", "null"]
       
        # Load CSV using the custom missing values list
        # 'keep_default_na=False' ensures pandas does not add its default list (which includes 'None') to our custom list.
        df = pd.read_csv(data_path, na_values=missing_values, keep_default_na=False)


        # Log confirmation that the operation was completed without errors
        logging.info("Dataset loaded successfully")


        # Return the loaded DataFrame
        return df


    except FileNotFoundError:
        # Handle specific error: File does not exist at the resolved path
        logging.error(f"File not found: {data_path}")
        raise FileNotFoundError(f"File not found: {data_path}")


    except Exception as e:
        # Handle any unexpected errors (e.g., permission issues, corrupted file format)
        logging.error(f"Unexpected error while loading dataset: {e}")
        raise RuntimeError(f"Error loading dataset: {e}")


def find_sig(p_value, alpha=0.05):
    """
    Determines statistical significance by comparing the p-value to the alpha threshold.
    Returns True if significant (p < alpha), False otherwise.
    """
    # Compare the calculated p-value against the significance level (default 0.05)
    if p_value < alpha:
        # Significant result: We have enough evidence to reject the Null Hypothesis (H0)
        logging.info(f"p-value is significant: {p_value}")
        return True
    else:
        # Non-significant result: We fail to reject the Null Hypothesis
        logging.info(f"p-value isn't significant: {p_value}")
        return False