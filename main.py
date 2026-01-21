import logging
import pandas as pd
import numpy as np

# --- Custom Modules Imports ---
from src.utils import setup_logging, load_dataset
from src.data_cleaning import (
    handle_missing_values_hybrid, 
    remove_outliers,
    remove_duplicates
)
from src.exploration import (
    data_info, 
    descriptive_stats
)
from src.correlation import (
    calculate_correlation, 
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
    2. Exploratory Data Analysis (General EDA)
    3. Hypothesis Testing: Correlations (Sleep vs. Stress)
    4. Hypothesis Testing: Diet Type vs. Happiness (ANOVA & Contrast)
    5. Hypothesis Testing: Mental Health vs. Social Interaction (ANOVA & Contrast)
    """
    logger.info("===============================================================")
    logger.info("Starting Analysis Pipeline...")
    logger.info("===============================================================")

    # ==============================================================================
    # Step 1: Data Loading & Preprocessing
    # ==============================================================================
    logger.info("--- Step 1: Data Loading & Preprocessing ---")
    
    filename = 'Mental_Health_Lifestyle_Dataset.csv'
    
    # 1. Load Data
    df = load_dataset(filename)
    
    # 2. Remove Duplicates (Critical first step)
    df = remove_duplicates(df)
    
    # Keep a raw copy for "Before vs After" visualization
    df_raw_for_plot = df.copy()
    
    # 3. Hybrid Cleaning (Impute numeric, Drop categorical missing)
    df_clean = handle_missing_values_hybrid(df)
    
    # 4. Outlier Removal (Z-score > 3.0)
    df_clean = remove_outliers(df_clean, threshold=3.0)
    
    # 5. Visualize Cleaning Impact
    plot_cleaning_report(df_raw_for_plot, df_clean)

    # ==============================================================================
    # Step 2: Exploratory Data Analysis (General EDA)
    # ==============================================================================
    logger.info("===============================================================")
    logger.info("--- Step 2: Exploratory Data Analysis (General Overview) ---")
    logger.info("===============================================================")
    
    # (2.1) Statistics Overview
    logger.info("(2.1) Generating General Statistics...")
    summary_text, info_df = data_info(df_clean)
    logger.info(f"Dataset Summary:\n{summary_text}")
    logger.info(f"Detailed Data Info:\n{info_df.to_string()}")
    
    numeric_stats, categorical_stats = descriptive_stats(df_clean)
    logger.info(f"Numeric Statistics:\n{numeric_stats.to_string()}")
    logger.info(f"Categorical Statistics:\n{categorical_stats.to_string()}")
    
    # (2.2) Visualizations
    logger.info("(2.2) Generating Categorical Pie Charts...")
    plot_categorical_pies(df_clean)
    
    logger.info("(2.2) Generating Numeric Distributions Grid...")
    plot_numeric_distributions_grid(df_clean)
    
    logger.info("(2.2) Generating Global Correlation Heatmap...")
    heat_map(df_clean)
    
    # ==============================================================================
    # Step 3: Hypothesis Testing - Correlations
    # ==============================================================================
    logger.info("===============================================================")
    logger.info("--- Step 3: Hypothesis Testing - Correlations ---")
    logger.info("===============================================================")
    
    # A. Sleep Hours vs. Stress Level (Hypothesis: More sleep -> Less stress)
    logger.info("(3.1) Testing Hypothesis: Sleep Hours vs. Stress Level")
    sleep_col = df_clean['Sleep Hours']
    stress_col = df_clean['Stress Level']
    
    var_1, var_2, corr, p_value = calculate_correlation(sleep_col, stress_col)
    
    
    # Plot only if significant
    plot_correlation(var_1, var_2, p_value)
    
    # B. Happiness vs. Social Interaction (Secondary Check)
    logger.info("(3.2) Testing Check: Happiness vs. Social Interaction")
    happiness_col = df_clean['Happiness Score']
    social_col = df_clean['Social Interaction Score']
    
    var_3,var_4,corr_2,p_value_2= calculate_correlation(happiness_col, social_col)
    
    # Plot only if significant
    plot_correlation(var_3, var_4, p_value_2)

    # ==============================================================================
    # Step 4: Hypothesis Testing - Diet Type vs. Happiness
    # ==============================================================================   
    logger.info("===============================================================")
    logger.info("--- Step 4: Hypothesis Testing (Diet Type vs. Happiness) ---")
    logger.info("===============================================================")
    
    group_col = 'Diet Type'
    value_col = 'Happiness Score'

    # (4.1) Visual Inspection
    logger.info("(4.1) Visualizing Distributions (Boxplots)...")
    plot_distributions(df_clean, group_col, value_col)

    # (4.2) Descriptive Stats
    display_descriptive_table(df_clean, group_col, value_col)

    # (4.3) One-Way ANOVA
    logger.info("(4.3) Running One-Way ANOVA...")
    anova_results = run_one_way_anova(df_clean, group_col, value_col)
    
    if anova_results is not None:
        display_anova_table(anova_results, group_col, value_col)

    # (4.4) Planned Contrast Analysis
    logger.info("(4.4) Running Planned Contrast (Plant-Based vs. Others)...")
    plant_based = ['Vegan', 'Vegetarian']
    others = ['Junk Food', 'Balanced', 'Keto']
   
    try:
        weights = create_contrast_weights(plant_based, others)
        display_contrast_weights(weights, group_col)
        
        contrast_results = run_planned_contrast(df_clean, group_col, value_col, weights)
        
        if contrast_results:
             logger.info(f"      Contrast Result: t={contrast_results['t_statistic']:.2f}, p={contrast_results['p_value']:.4f}")
       
    except Exception as e:
        logger.error(f"Error in Hypothesis 1 Contrast: {e}")

    # ==============================================================================
    # Step 5: Hypothesis Testing - Mental Health vs. Social Interactions
    # ==============================================================================
    logger.info("===============================================================")
    logger.info("--- Step 5: Hypothesis Testing (Mental Health vs. Social) ---")
    logger.info("===============================================================")
   
    group_col = 'Mental Health Condition'
    value_col = 'Social Interaction Score'

    # (5.1) Visual Inspection
    logger.info("(5.1) Visualizing Distributions...")
    plot_distributions(df_clean, group_col, value_col)

    # (5.2) Descriptive Stats
    display_descriptive_table(df_clean, group_col, value_col)

    # (5.3) One-Way ANOVA
    logger.info("(5.3) Running One-Way ANOVA...")
    anova_results_2 = run_one_way_anova(df_clean, group_col, value_col)
    
    if anova_results_2 is not None:
        display_anova_table(anova_results_2, group_col, value_col)

    # (5.4) Planned Contrast Analysis
    logger.info("(5.4) Running Planned Contrast (Healthy vs. Conditions)...")
    try:
        # Define groups dynamically
        healthy_group = ['None']
        all_groups = df_clean[group_col].unique()
        condition_groups = [g for g in all_groups if g != 'None']
       
        # Create & Display Weights
        weights_2 = create_contrast_weights(healthy_group, condition_groups)
        display_contrast_weights(weights_2, group_col)
       
        # Run Contrast
        contrast_results_2 = run_planned_contrast(df_clean, group_col, value_col, weights_2)
       
        if contrast_results_2:
             logger.info(f"      Contrast Result: t={contrast_results_2['t_statistic']:.2f}, p={contrast_results_2['p_value']:.4f}")
       
    except Exception as e:
        logger.error(f"Error in Hypothesis 2 Contrast: {e}")

    logger.info("===============================================================")
    logger.info("Analysis Pipeline Completed Successfully.")
    logger.info("===============================================================")


if __name__ == "__main__":
    main()