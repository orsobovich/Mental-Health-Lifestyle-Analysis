import sys
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)

def find_sig(p_value, alpha=0.05):
    if p_value < alpha:
        logging.info(f"p-value is significant: {p_value}")
        return True
    else:
        logging.info(f"p-value isn't significant: {p_value}")
        return False