import pandas as pd
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


        # Return the Series containing the group means
        return means


    except KeyError as key_error:
        # Catch the error if a specified column does not exist in the DataFrame
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
        # Log that the function has started
        logger.info(f"Running one-way ANOVA on '{value_col}' grouped by '{group_col}'")


        # Step 1: Data Validation & Inspection
        # We call this helper explicitly to validate input data (catch missing columns/types)
        # and to log the group means for sanity check before running the statistical model.
        compute_group_means(data, group_col, value_col)


        # Step 2: Prepare formula for OLS
        # Construct the model formula string: target ~ predictor.
        # Uses Q() to handle column names with spaces and C() to force categorical interpretation.
        formula = f'Q("{value_col}") ~ C(Q("{group_col}"))'
       
        # Fit the model using Ordinary Least Squares regression
        model = ols(formula, data=data).fit()


        # Perform ANOVA (Type II is used to handle potentially unbalanced group sizes)
        anova_table = sm.stats.anova_lm(model, typ=2)


        # Warn if any groups are very small
        counts = data[group_col].value_counts()
        small_groups = counts[counts < 5]
        if not small_groups.empty:
            logger.warning(f"Small groups detected: {small_groups.to_dict()}")


        # Log ANOVA results
        logger.info(f"ANOVA completed:\n{anova_table}")


        # Return ANOVA table
        return anova_table


    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during ANOVA: {e}")
        raise e


def create_contrast_weights(positive_groups: list, negative_groups: list):
    """
    Creates a weight dictionary for planned contrasts.
    Logic: Assigns weights so that the sum of all weights equals 0 (required for valid contrasts).
    """
    try:
        # Catch if one of the lists is empty
        if not positive_groups or not negative_groups:
            logger.error("Both positive and negative groups must be provided.")
            raise ValueError("Groups lists cannot be empty")
       
        # --- Calculate Weights ---
        # For a contrast to be valid, the sum of weights must be zero.
        # We assign a total mass of +1.0 to the positive side and -1.0 to the negative side.
       
        # Divide +1.0 by the number of positive groups so each group contributes equally
        # Example: If 2 groups, each gets +0.5
        pos_weight = 1.0 / len(positive_groups)
       
        # Divide -1.0 by the number of negative groups so each group contributes equally
        # Example: If 3 groups, each gets -0.33
        neg_weight = -1.0 / len(negative_groups)


        # Create a new dictionary
        weights = {}      
       
       # Map each value in the positive list to the calculated positive weight
        for group in positive_groups:
            weights[group] = pos_weight
       
        # Map each value in the negative list to the calculated negative weight
        for group in negative_groups:
            weights[group] = neg_weight
       
        # Log the final weight mapping for verification
        logger.info(f"Contrast weights created: {weights}")


        # Return the completed dictionary mapping values to their weights
        return weights


    except Exception as e:
       # Catch any unexpected errors
        logger.error(f"Error creating contrast weights: {e}")
        raise e


def run_planned_contrast(data: pd.DataFrame, group_col: str, value_col: str, contrast_weights: dict):
    """
    Runs a planned contrast using OLS regression (without intercept).
   
    Args:
        data (pd.DataFrame): Dataset.
        group_col (str): Categorical grouping column.
        value_col (str): Continuous target column.
        contrast_weights (dict): Mapping of {Group: Weight} (sum must be 0).
       
    Returns:
        dict: t-statistic, p-value, and degrees of freedom.
    """
    try:
        logger.info(f"Starting planned contrast for '{value_col}' by '{group_col}'")
       
        # 1. Log means for validation
        compute_group_means(data, group_col, value_col)
       
        # 2. Validate groups exist
        missing = set(contrast_weights) - set(data[group_col].unique())
        if missing:
            raise KeyError(f"Groups missing from dataset: {missing}")
       
        # 3. Fit OLS Model (No Intercept)
        # CRITICAL: We use "- 1" to remove the intercept.
        # This forces the model coefficients to represent the absolute group means directly,
        # rather than the difference from a reference group.
        formula = f"Q('{value_col}') ~ C(Q('{group_col}')) - 1"
        model = smf.ols(formula, data=data).fit()
       
        # 4. Map Weights to Model Parameters (The "Alignment" Step)
        # Statsmodels creates internal parameter names (e.g., "C(Diet Type)[T.Vegan]").
        # We cannot rely on the dictionary order. This logic iterates through the
        # model's actual parameters and fetches the correct weight for each group.
        contrast_vector = [
            next((w for g, w in contrast_weights.items() if f"[{g}]" in p or f"[T.{g}]" in p), 0)
            for p in model.params.index
        ]
       
        # 5. Run Linear Hypothesis Test (T-Test)
        # Performs a t-test on the linear combination of coefficients (means) based on the vector
        t_result = model.t_test(contrast_vector)
       
        # Return results
        return {
            "t_statistic": t_result.tvalue.item(),
            "degrees_of_freedom": t_result.df_denom,
            "p_value": t_result.pvalue.item()
        }
       
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Planned contrast failed: {e}")
        raise e