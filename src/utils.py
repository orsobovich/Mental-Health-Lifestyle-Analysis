import logging

def setup_logging():
    """
    Sets up the logging configuration for the entire project.
    """
    # Configure the logging system with level INFO and a specific format including time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#load