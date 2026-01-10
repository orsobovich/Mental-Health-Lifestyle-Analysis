
def compute_group_means(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Compute the mean of `value_col` for each category in `group_col`.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    group_col : str
        The name of the categorical column to group by.
        (Why str? Because we need the column name as text)
    value_col : str
        The name of the numeric column to calculate the mean for.

    Returns
    -------
    pd.Series or None
        A pandas Series with group names as index and their mean as values.
        Returns None if an error occurs.
    """
    
    # Log that the function has started
    logger.info(f"Computing mean of '{value_col}' per '{group_col}'")

    try:
        # Compute the mean of value_col for each group in group_col
        means = data.groupby(group_col)[value_col].mean()
        
        # Log the first few results for monitoring
        logger.info(f"Computed means (head):\n{means.head()}")
        return means

    except KeyError as key_error:
        # Catch the error if the specified column does not exist in the DataFrame
        logger.error(f"Column not found: {key_error}")
        return None

    except TypeError as type_error:
        # Catch errors if the input data types are incorrect (for example non-DataFrame values)
        logger.error(f"Type error encountered: {type_error}")
        return None

    except ValueError as value_error:
        # Catch errors if the computation cannot be performed due to invalid values (for example non-numeric values)
        logger.error(f"Value error encountered: {value_error}")
        return None

    except Exception as e:
        # Catch any other unexpected errors that might occur during computation
        logger.error(f"Unexpected error during group mean computation: {e}")
        return None
    
    
def run_one_way_anova(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Perform one-way ANOVA and return ANOVA table.

    Features:
    - Logging for info, warnings, and errors
    - Handles spaces/special characters in column names with Q()
    - Marks categorical variable with C() for ANOVA
    - Warns if some groups are very small (<5)
    - Safe execution with try-except
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    try:
        logger.info(f"Running one-way ANOVA on '{value_col}' grouped by '{group_col}'")

        # Wrap dependent variable column with Q() to handle spaces/special characters
        # Wrap independent categorical variable with C(Q()) to mark it as categorical
        # Build formula for OLS model
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
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during ANOVA: {unexpected_error}")
        return None


def create_contrast_weights(positive_groups: list, negative_groups: list) -> dict:
    """
    Creates a weight dictionary for planned contrasts.
    
    Logic:
    - Assigns positive weights to 'positive_groups'.
    - Assigns negative weights to 'negative_groups'.
    - Ensures the sum of weights is zero (required for valid contrasts).
    
    Args:
        positive_groups (list): List of group names expected to have HIGHER values.
        negative_groups (list): List of group names expected to have LOWER values.
        
    Returns:
        dict: A dictionary mapping group names to their calculated weights.
    """
    try:
        if not positive_groups or not negative_groups:
            logger.error("Both positive and negative groups must be provided.")
            raise ValueError("Groups lists cannot be empty")

        # sum negative weights + sum positive weights = 0
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
        logger.error(f"Error creating contrast weights: {e}")
        raise e


def run_planned_contrast(data: pd.DataFrame, group_col: str, value_col: str, contrast_weights: dict, visualize: bool = False):
    """
    Perform a planned contrast on the specified groups.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - group_col: categorical independent variable
    - value_col: continuous dependent variable
    - contrast_weights: dictionary specifying weights for each group
    - visualize: if True, plot weighted means
    
    Returns:
    - dict with t_statistic, degrees_of_freedom, p_value or None on error
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    try:
        # Log start of planned contrast
        logger.info(f"Running planned contrast for '{value_col}' across '{group_col}'")
        # Check that all groups in contrast_weights exist in the data
        missing_groups = [g for g in contrast_weights if g not in data[group_col].unique()]
        if missing_groups:
            raise KeyError(f"Groups missing from data: {missing_groups}")

        # Group data by the categorical variable
        grouped = data.groupby(group_col)[value_col]
        means = grouped.mean()      # Compute mean per group
        ns = grouped.count()        # Compute sample size per group

        # Compute weighted contrast value
        contrast_value = sum(contrast_weights[g] * means[g] for g in contrast_weights)

        # Compute variance of the contrast
        variance = sum((contrast_weights[g] ** 2) * grouped.var()[g] / ns[g] for g in contrast_weights)

        # Calculate t-statistic and degrees of freedom
        t_stat = contrast_value / (variance ** 0.5)
        df = len(data) - len(means)

        # Compute two-tailed p-value
        p_value = stats.t.sf(abs(t_stat), df) * 2

        # Log results
        logger.info(f"Planned contrast result: t={t_stat:.3f}, df={df}, p={p_value:.4f}")

        # Optional visualization of weighted means
        if visualize:
            weighted_means = {g: means[g]*contrast_weights[g] for g in contrast_weights}
            plt.figure(figsize=(8,5))
            plt.bar(weighted_means.keys(), weighted_means.values(), color='skyblue')
            plt.ylabel(f"Weighted {value_col}")
            plt.title("Planned Contrast Weighted Means")
            plt.show()

        # Return results as dictionary
        return {"t_statistic": t_stat, "degrees_of_freedom": df, "p_value": p_value}

    except KeyError as key_error:
        # Raised if a specified group does not exist
        logger.error(f"Key error: {key_error}")
        return None

    except ZeroDivisionError:
        # Raised if variance calculation is zero (cannot divide by zero)
        logger.error("Variance calculation resulted in zero. Contrast cannot be computed")
        return None

    except Exception as unexpected_error:
        # Catch all other unexpected errors
        logger.error(f"Unexpected error during planned contrast: {unexpected_error}")
        return None