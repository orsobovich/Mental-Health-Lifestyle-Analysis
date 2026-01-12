import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)


def compute_group_means(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Compute the mean of `value_col` for each category in `group_col`.
    Handles specific errors and raises them to stop execution.
    """
    try:
        # Log that the function has started
        logger.info(f"Computing mean of '{value_col}' per '{group_col}'")


        # Compute the mean of value_col for each group in group_col
        means = data.groupby(group_col)[value_col].mean()
       
        # Log the first few results for monitoring
        logger.info(f"Computed means (head):\n{means.head()}")
        return means


    except KeyError as key_error:
        # Catch the error if the specified column does not exist in the DataFrame
        logger.error(f"Column not found: {key_error}")
        raise key_error


    except TypeError as type_error:
        # Catch errors if the input data types are incorrect
        logger.error(f"Type error encountered: {type_error}")
        raise type_error


    except ValueError as value_error:
        # Catch errors if the computation cannot be performed due to invalid values
        logger.error(f"Value error encountered: {value_error}")
        raise value_error


    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during group mean computation: {e}")
        raise e
 
    
def run_one_way_anova(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Perform one-way ANOVA and return ANOVA table.
    """
    try:
        logger.info(f"Running one-way ANOVA on '{value_col}' grouped by '{group_col}'")


        # Step 1: Compute means (If this fails, the detailed errors above will be raised)
        compute_group_means(data, group_col, value_col)


        # Step 2: Prepare formula for OLS
        # Wrap dependent variable column with Q() to handle spaces/special characters
        # Wrap independent categorical variable with C(Q()) to mark it as categorical
        formula = f'Q("{value_col}") ~ C(Q("{group_col}"))'
       
        # Fit the model using Ordinary Least Squares regression
        model = ols(formula, data=data).fit()


        # Perform ANOVA on the fitted model, Type II sums of squares
        anova_table = sm.stats.anova_lm(model, typ=2)


        # Warn if any groups are very small
        counts = data[group_col].value_counts()
        small_groups = counts[counts < 5]
        if not small_groups.empty:
            logger.warning(f"Small groups detected: {small_groups.to_dict()}")


        # Log ANOVA results
        logger.info(f"ANOVA completed:\n{anova_table}")
        return anova_table


    except Exception as unexpected_error:
        # Log and raise to stop main.py
        logger.error(f"Unexpected error during ANOVA: {unexpected_error}")
        raise unexpected_error


def create_contrast_weights(positive_groups: list, negative_groups: list):
    """
    Creates a weight dictionary for planned contrasts.
    """
    try:
        if not positive_groups or not negative_groups:
            logger.error("Both positive and negative groups must be provided.")
            raise ValueError("Groups lists cannot be empty")


        # Ensures the sum of weights is zero (required for valid contrasts) --> sum negative weights + sum positive weights = 0
       
        # positive weight / number of positive groups
        pos_weight = 1.0 / len(positive_groups)
       
        # negative weight / number of negative groups
        neg_weight = -1.0 / len(negative_groups)


        weights = {}
       
        # build the weights dictionary
        for group in positive_groups:
            weights[group] = pos_weight
           
        for group in negative_groups:
            weights[group] = neg_weight
           
        logger.info(f"Contrast weights created: {weights}")
        return weights


    except Exception as e:
	   # Log and raise to stop main.py
        logger.error(f"Error creating contrast weights: {e}")
        raise e


def run_planned_contrast(data: pd.DataFrame, group_col: str, value_col: str, contrast_weights: dict):
    """
    Runs a planned contrast using OLS regression (statsmodels).
    Automatic handling of means and standard errors via linear hypothesis test.
    """

    try:
        logger.info(f"Starting planned contrast for '{value_col}' by '{group_col}'")
        
        # Compute means (Log means before analysis - for consistency)
        compute_group_means(data, group_col, value_col)
        
        # Validation
        missing = set(contrast_weights) - set(data[group_col].unique())
        if missing:
            raise KeyError(f"Groups missing from dataset: {missing}")
        
        # Fit OLS Model
        formula = f"Q('{value_col}') ~ C(Q('{group_col}')) - 1"
        model = smf.ols(formula, data=data).fit()
        
        # Map Weights to model parameters
        contrast_vector = [
            next((w for g, w in contrast_weights.items() if f"[{g}]" in p or f"[T.{g}]" in p), 0)
            for p in model.params.index
        ]
        
        # Run T-Test
        t_result = model.t_test(contrast_vector)
        t_stat, p_val = t_result.tvalue.item(), t_result.pvalue.item()
        
        logger.info(f"Contrast Result: t={t_stat:.3f}, p={p_val:.4f}")
        
        return {
            "t_statistic": t_stat,
            "degrees_of_freedom": t_result.df_denom,
            "p_value": p_val
        }
        
    except Exception as e:
        # Log and raise to stop main.py
        logger.error(f"Error in planned contrast: {e}")
        raise e