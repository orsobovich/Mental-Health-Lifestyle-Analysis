import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from src.significance_check import find_sig 
import pandas as pd
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)


 # Create a correlation plot
def plot_correlation(cor_1, cor_2, p_value): 
    if find_sig(p_value):
        sns.regplot(x=cor_1, y=cor_2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title("Correlation Plot")
        plt.xlabel(cor_1.name) #Takes the name of the column
        plt.ylabel(cor_2.name) #Takes the name of the column
        plt.show()
        
        
def heat_map_correlation_pearson(df):
    numeric_df = df.select_dtypes(include=['number']) # including only the numeric category
    corr_matrix = numeric_df.corr() #create matrix of pearson correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

           
def plot_distributions(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Plot the distribution of a numeric column across categories using a boxplot.
    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    group_col : str
        The categorical column to group by (must be str).
    value_col : str
        The numeric column to plot.

    Returns
    -------
    bool
        True if the plot was successfully created, False otherwise.
    """

    # Log that the function started
    logger.info(f"Plotting distribution of '{value_col}' across '{group_col}'")

    try:
        # Prepare the data: select relevant columns and drop missing values
        plot_data = data[[group_col, value_col]].dropna()

        # Create the figure with a fixed size
        plt.figure(figsize=(10, 6))

        # Draw the boxplot with seaborn
        sns.boxplot(data=plot_data, x=group_col, y=value_col)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=30)

        # Add a title to the plot
        plt.title(f"Distribution of {value_col} by {group_col}")

        # Adjust layout to avoid clipping
        plt.tight_layout()

        # Display the plot
        plt.show()

        # Return True to indicate success
        return True

    except Exception as unexpected_error:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during plotting: {unexpected_error}")
        return False
    
