from src.visualization import (
    heat_map_correlation_pearson, 
    plot_correlation               
)
import logging
from src.utils import setup_logging, load_dataset
from src.correlation import calculate_correlation
from scipy.stats import spearmanr, pearsonr
from src.significance_check import find_sig


# Initialize the logging configuration
setup_logging()


# Get the logger instance for the main module
logger = logging.getLogger(__name__)


#load data
df = load_dataset('Mental_Health_Lifestyle_Dataset.csv')



#exploration


#data_cleaning.py


#correlation       
logging.info("correlation between Stress Level and Sleep Hours:")        
calculate_correlation(df['Stress Level'], df['Sleep Hours'])

logging.info("correlation between Age and Sleep Hours:")
calculate_correlation(df["Age"], df["Sleep Hours"]) 

logging.info("correlation between Social Interaction Score and Stress Level:")
calculate_correlation(df["Social Interaction Score"], df["Stress Level"])

logging.info("correlation between Age and Social Interaction Score:")
calculate_correlation(df["Age"], df["Social Interaction Score"])

heat_map_correlation_pearson(df)
corr, p_val = spearmanr(df['Sleep Hours'], df['Stress Level'])
logger.info(f"Sleep vs Stress: corr={corr:.2f}, p={p_val:.4f}")
plot_correlation(df['Sleep Hours'], df['Stress Level'], p_val)



#anova


#visualization.py

