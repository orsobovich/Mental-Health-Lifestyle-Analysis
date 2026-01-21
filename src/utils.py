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
        logging.info("Starting dataset loading")

        # Resolve project root dynamically (assumes script is in src/)
        base_dir = Path(__file__).resolve().parents[1]
        data_path = Path(filename)

        # Construct absolute path if a relative one is provided
        if not data_path.is_absolute():
            data_path = base_dir / filename

        logging.info(f"Resolved dataset path: {data_path}")

        # Define specific strings to be treated as NaN
        # In our dataset we want to avoid converting 'None' in the Mental Health Condition into a null value
        missing_values = ["", " ", "NaN", "nan", "NA", "null"]
        
        # Load CSV using the custom missing values list
        df = pd.read_csv(data_path, na_values=missing_values, keep_default_na=False)

        logging.info("Dataset loaded successfully")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise FileNotFoundError(f"File not found: {data_path}")

    except Exception as e:
        logging.error(f"Unexpected error while loading dataset: {e}")
        raise RuntimeError(f"Error loading dataset: {e}")
    
    
def find_sig(p_value, alpha=0.05):
    if p_value < alpha:
        logging.info(f"p-value is significant: {p_value}")
        return True
    else:
        logging.info(f"p-value isn't significant: {p_value}")
        return False