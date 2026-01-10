import logging
import sys
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout,force=True)


def find_sig(p_value, alpha=0.05):
    if p_value < alpha:
        logging.info(f"p-value is significant: {p_value}")
        return True
    else:
        logging.info(f"p-value isn't significant: {p_value}")
        return False