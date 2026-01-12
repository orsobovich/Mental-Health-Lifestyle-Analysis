from src.visualization import (
    heat_map_correlation_pearson,
    plot_correlation,
    display_descriptive_table,
    plot_distributions,
    display_anova_table,
    display_contrast_weights
)
from src.ANOVA import (
    create_contrast_weights, 
    run_planned_contrast, 
    run_one_way_anova
)
from src.data_cleaning import handle_missing_values_hybrid, get_column_types, remove_outliers
import logging
from src.utils import setup_logging, load_dataset
from src.correlation import calculate_correlation, is_valid_level, level_to_numeric
from scipy.stats import spearmanr, pearsonr
from src.significance_check import find_sig
import pandas as pd
import numpy as np

# Configure the logger
# This sets up the logging format (Time - Level - Message) for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to execute the full data analysis pipeline.
    Steps:
    1. Data Loading & Cleaning
    2. Exploratory Data Analysis (Correlation)
    3. Hypothesis Testing (ANOVA & Planned Contrasts)
    """
    logger.info("Starting Analysis Pipeline...")

    # ==============================================================================
    # Step 1: Data Loading and Cleaning
    # ==============================================================================
    logger.info("--- Step 1: Data Loading & Preprocessing ---")
    
    filename = 'Mental_Health_Lifestyle_Dataset.csv'
    
    # Load the raw dataset
    df = load_dataset(filename)
    
    # Handle missing values using a hybrid approach:
    # - Numeric columns are filled with the mean (to preserve distribution)
    # - Rows with missing categorical values are dropped (to avoid bias)
    df_clean = handle_missing_values_hybrid(df)
    
    # 1.2 Remove Outliers (New Step!)
    # Removes extreme values (Z-score > 3) to ensure statistical validity
    df_clean = remove_outliers(df_clean, threshold=3.0)

    # ==============================================================================
    # Step 2: Visualization of Correlations (Exploratory + Hypothesis 2)
    # ==============================================================================
    logger.info("--- Step 2: Visualizing Correlations ---")
    
    # A. General Heatmap
    # Displays Pearson correlations between all numeric variables to identify patterns
    heat_map_correlation_pearson(df_clean)
    
    # B. Specific Hypothesis: Sleep Hours vs. Stress Level
    # We want to check if more sleep correlates with less stress.
    
    sleep_col = df_clean['Sleep Hours']
    stress_col = df_clean['Stress Level']
    
    # Calculate Spearman correlation because 'Stress Level' is ordinal (ranked data)
    corr1, p_val1 = spearmanr(sleep_col, stress_col)
    logger.info(f"Hypothesis 2 Result (Sleep vs Stress): Correlation r={corr1:.3f}, p-value={p_val1:.4f}")
    
    # Plot the correlation with a regression line ONLY if the result is significant (p < 0.05)
    plot_correlation(sleep_col, stress_col, p_val1)
    
    # C. Additional Correlation Checks
    # Check correlation between Happiness Score and Social Interaction Score
    
    happiness_col = df_clean['Happiness Score']
    social_col = df_clean['Social Interaction Score']
    
    # Calculate Spearman correlation because 'Stress Level' is ordinal (ranked data)
    corr2, p_val2 = spearmanr(happiness_col, social_col)
    logger.info(f"Hypothesis 2 Result (Happiness Score vs Social Interaction Score): Correlation r={corr2:.3f}, p-value={p_val2:.4f}")
    
    # Plot the correlation with a regression line ONLY if the result is significant (p < 0.05)
    plot_correlation(happiness_col, social_col, p_val2)

    # ==============================================================================
    # Step 3: Hypothesis Testing (ANOVA & Planned Contrasts)
    # Hypothesis 1: Diet Type vs. Happiness Score
    # ==============================================================================
    logger.info("--- Step 3: ANOVA & Contrasts (Diet vs Happiness) ---")
    
    group_col = 'Diet Type'
    value_col = 'Happiness Score'

    # A. Visual Inspection
    # Plot the distribution (Boxplot + Stripplot) to see the spread of data before analysis
    plot_distributions(df_clean, group_col, value_col)

    # B. Descriptive Statistics
    # Display a table with Mean, Standard Deviation, and Count for each diet group
    display_descriptive_table(df_clean, group_col, value_col)

    # C. One-Way ANOVA
    # Run the overall ANOVA test to check if *any* group is different
    anova_results = run_one_way_anova(df_clean, group_col, value_col)
    
    # If ANOVA ran successfully, display the results table
    if anova_results is not None:
        display_anova_table(anova_results)

    # D. Planned Contrast Analysis
    # We specifically hypothesize that Plant-Based diets lead to higher happiness than others.
    
    # Define the groups
    plant_based = ['Vegan', 'Vegetarian']
    others = ['Junk', 'Balanced', 'Keto'] # Ensure these match exact names in your CSV!
    
    try:
        # Create weights: Positive for plant-based, Negative for others
        # The function ensures weights sum to zero
        weights = create_contrast_weights(plant_based, others)
        
        # Display the calculated weights as a table for transparency
        display_contrast_weights(weights)
        
        # Run the statistical contrast test (t-test on the linear combination)
        # Results (t-statistic and p-value) will be logged to the console
        contrast_results = run_planned_contrast(df_clean, group_col, value_col, weights)
        
        if contrast_results:
             logger.info(f"Contrast Analysis Result: t={contrast_results['t_statistic']:.2f}, p={contrast_results['p_value']:.4f}")
        
    except Exception as e:
        logger.error(f"Could not run contrast visualization: {e}")

    logger.info("Analysis Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()





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

