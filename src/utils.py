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
