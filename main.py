import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# --- Custom Modules Imports ---
from src.utils import setup_logging, load_dataset
from src.significance_check import find_sig
from src.data_cleaning import (
    handle_missing_values_hybrid, 
    get_column_types, 
    remove_outliers
)
from src.exploration import (
    data_info, 
    descriptive_stats
)
from src.correlation import (
    calculate_correlation, 
    is_valid_level, 
    level_to_numeric
)
from src.ANOVA import (
    create_contrast_weights, 
    run_planned_contrast, 
    run_one_way_anova
)
from src.visualization import (
    plot_correlation,
    display_descriptive_table,
    plot_distributions,
    display_anova_table,
    display_contrast_weights,
    plot_cleaning_report,
    heat_map,
    plot_numeric_distributions_grid,
    plot_categorical_pies
)

# Initialize logging configuration once
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """
    Main execution pipeline for the Mental Health & Lifestyle Analysis.
    
    Workflow Steps:
    1. Data Loading & Preprocessing (Cleaning, Outliers)
    2. Exploratory Data Analysis (EDA - Correlations, Distributions)
    3. Hypothesis 1: Diet Type vs. Happiness Score (ANOVA & Contrast)
    4. Hypothesis 2: Mental Health Condition vs. Social Interaction (ANOVA & Contrast)
    """
    logger.info("Starting Analysis Pipeline...")

    # ==============================================================================
    # Step 1: Data Loading & Preprocessing
    # ==============================================================================
    logger.info("--- Step 1: Data Loading & Preprocessing ---")
    
    filename = 'Mental_Health_Lifestyle_Dataset.csv'
    
    # 1. Load Data
    df = load_dataset(filename)
    
    # Keep a raw copy for "Before vs After" visualization
    df_raw_for_plot = df.copy()
    
    # 2. Hybrid Cleaning (Impute numeric, Drop categorical missing)
    df_clean = handle_missing_values_hybrid(df)
    
    # 3. Outlier Removal (Z-score > 3.0)
    df_clean = remove_outliers(df_clean, threshold=3.0)
    
    # 4. Visualize Cleaning Impact
    plot_cleaning_report(df_raw_for_plot, df_clean)

    # ==============================================================================
    # Step 2: Visualization of Correlations (EDA + Correlation Hypotheses)
    # ==============================================================================
    logger.info("--- Step 2: Visualizing Correlations & EDA ---")
    
    # 1. General Statistics Overview
    # data_info returns (Summary String, Info DataFrame)
    summary_text, info_df = data_info(df_clean)
    
    # The summary is already a string, so we just log it directly
    logger.info(f"Dataset Summary:\n{summary_text}")
    
    # The info_df is a DataFrame, so we use .to_string() to format it nicely
    logger.info(f"Detailed Data Info:\n{info_df.to_string()}")
    
    # descriptive_stats returns (Numeric DataFrame, Categorical DataFrame)
    numeric_stats, categorical_stats = descriptive_stats(df_clean)
    
    logger.info(f"Numeric Statistics:\n{numeric_stats.to_string()}")
    logger.info(f"Categorical Statistics:\n{categorical_stats.to_string()}")
    
    # 2. General Plots
    logger.info("Generating Categorical Pie Charts...")
    plot_categorical_pies(df_clean)
    
    logger.info("Generating Numeric Distributions Grid...")
    plot_numeric_distributions_grid(df_clean)
    
    # 3. Correlation Heatmap
    logger.info("Generating Correlation Heatmap for Numeric Variables")
    heat_map(df_clean)
    
    # --------------------------------------------------------------------------
    # A. Specific Check: Sleep Hours vs. Stress Level 
    # Hypothesis: More sleep correlates with less stress.
    # --------------------------------------------------------------------------
    sleep_col = df_clean['Sleep Hours']
    stress_col = df_clean['Stress Level']
    
    # Spearman correlation (Stress Level is ordinal)
    corr1, p_val1 = spearmanr(sleep_col, stress_col)
    logger.info(f"Hypothesis Check (Sleep vs Stress): r={corr1:.3f}, p={p_val1:.4f}")
    
    # Plot only if significant
    plot_correlation(sleep_col, stress_col, p_val1)
    
    # --------------------------------------------------------------------------
    # B. Specific Check: Happiness Score vs. Social Interaction Score
    # --------------------------------------------------------------------------
    happiness_col = df_clean['Happiness Score']
    social_col = df_clean['Social Interaction Score']
    
    corr2, p_val2 = spearmanr(happiness_col, social_col)
    logger.info(f"Hypothesis Check (Happiness vs Social Interaction): r={corr2:.3f}, p={p_val2:.4f}")
    
    plot_correlation(happiness_col, social_col, p_val2)

    # ==============================================================================
    # Step 3: Hypothesis 1 - Diet Type vs. Happiness Score
    # ==============================================================================
    logger.info("--- Step 3: Hypothesis 1 (Diet vs Happiness) ---")
   
    group_col = 'Diet Type'
    value_col = 'Happiness Score'

    # A. Visual Inspection
    plot_distributions(df_clean, group_col, value_col)

    # B. Descriptive Stats Table
    display_descriptive_table(df_clean, group_col, value_col)

    # C. One-Way ANOVA
    anova_results = run_one_way_anova(df_clean, group_col, value_col)
    
    if anova_results is not None:
        display_anova_table(anova_results, group_col, value_col)

    # D. Planned Contrast Analysis
    # Hypothesis: Plant-Based (Vegan, Vegetarian) > Others (Junk, Balanced, Keto)
    plant_based = ['Vegan', 'Vegetarian']
    others = ['Junk Food', 'Balanced', 'Keto']
   
    try:
        weights = create_contrast_weights(plant_based, others)
        display_contrast_weights(weights, group_col)
        
        contrast_results = run_planned_contrast(df_clean, group_col, value_col, weights)
        
        if contrast_results:
             logger.info(f"Contrast 1 Result (Plant-Based vs Others): t={contrast_results['t_statistic']:.2f}, p={contrast_results['p_value']:.4f}")
       
    except Exception as e:
        logger.error(f"Could not run contrast visualization for Hypothesis 1: {e}")

    # ==============================================================================
    # Step 4: Hypothesis 2 - Mental Health Condition vs. Social Interactions
    # ==============================================================================
    logger.info("--- Step 4: Hypothesis 2 (Mental Health vs Social Interactions) ---")
   
    group_col = 'Mental Health Condition'
    value_col = 'Social Interaction Score'

    # A. Visual Inspection
    plot_distributions(df_clean, group_col, value_col)

    # B. Descriptive Stats Table
    display_descriptive_table(df_clean, group_col, value_col)

    # C. One-Way ANOVA
    anova_results_2 = run_one_way_anova(df_clean, group_col, value_col)
    
    if anova_results_2 is not None:
        display_anova_table(anova_results_2, group_col, value_col)

    # D. Planned Contrast Analysis
    # Hypothesis: 'None' (Healthy) > 'Condition' (Depression, Anxiety, etc.)
    try:
        # Define groups dynamically
        healthy_group = ['None']
        all_groups = df_clean[group_col].unique()
        condition_groups = [g for g in all_groups if g != 'None']
       
        logger.info(f"Contrast Groups -> Healthy: {healthy_group} vs. Conditions: {condition_groups}")

        # Create & Display Weights
        weights_2 = create_contrast_weights(healthy_group, condition_groups)
        display_contrast_weights(weights_2, group_col)
       
        # Run Contrast
        contrast_results_2 = run_planned_contrast(df_clean, group_col, value_col, weights_2)
       
        if contrast_results_2:
             logger.info(f"Contrast 2 Result (Healthy vs Conditions): t={contrast_results_2['t_statistic']:.2f}, p={contrast_results_2['p_value']:.4f}")
       
    except Exception as e:
        logger.error(f"Could not run contrast analysis for Hypothesis 2: {e}")

    logger.info("Analysis Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()