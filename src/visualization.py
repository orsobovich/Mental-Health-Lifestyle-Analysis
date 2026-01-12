import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, chi2_contingency
from src.significance_check import find_sig 
import pandas as pd
import logging

# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)
        
        
def heat_map_correlation_pearson(df):
    numeric_df = df.select_dtypes(include=['number']) # including only the numeric category
    corr_matrix = numeric_df.corr() #create matrix of pearson correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()



def plot_correlation(cor_1, cor_2, p_value):
    """
    Plots a regression line (scatter plot + line) ONLY if the correlation is significant.
    """
    try:
        # Check if the p-value indicates statistical significance (True if significant)
        if find_sig(p_value):
            # Create a figure
            plt.figure(figsize=(8, 6))
           
            # Create a regression plot using Seaborn
            # scatter_kws={'alpha':0.5}: Makes the dots semi-transparent
            # line_kws={'color':'red'}: Makes the regression line red
            sns.regplot(x=cor_1, y=cor_2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
           
            # Set the title dynamically based on column names
            plt.title(f"Correlation: {cor_1.name} vs {cor_2.name}")
           
            # Label the axes
            plt.xlabel(cor_1.name)
            plt.ylabel(cor_2.name)
           
            # Show the plot
            plt.show()
        else:
            # If not significant, log that we are skipping the plot
            logger.info("Correlation not significant, skipping plot.")
           
    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Error plotting correlation: {e}")
        # Stop the program immediately by raising the error
        raise e
   


def plot_dataframe_as_table(df: pd.DataFrame, title: str):
    """
    Renders a pandas DataFrame as a visual table image using Matplotlib.
    Used by other functions to display ANOVA results and Weights.
    """
    try:
        # Create a new figure and a set of axes with size 8x3 inches
        fig, ax = plt.subplots(figsize=(8, 3))
       
        # Turn off the X-axis
        ax.xaxis.set_visible(False)
       
        # Turn off the Y-axis
        ax.yaxis.set_visible(False)
       
        # Remove the black box frame/border around the plot area
        ax.set_frame_on(False)
       
        # Create the visual table using the data from the DataFrame
        table = plt.table(
            cellText=df.values,        # The actual data (numbers/text) inside the cells
            colLabels=df.columns,      # The headers (column names)
            rowLabels=df.index,        # The row names (index)
            cellLoc='center',          # Center the text inside each cell
            loc='center'               # Place the table in the center of the figure
        )
       
        # Disable automatic font size scaling to have manual control
        table.auto_set_font_size(False)
       
        # Set the font size to 10 for readability
        table.set_fontsize(10)
       
        # Scale the table cells to be larger
        table.scale(1.2, 1.2)
       
        # Add a title to the image, with some padding (distance) from the table
        plt.title(title, fontsize=14, pad=20)
       
        # Automatically adjust subplot parameters to give specified padding
        plt.tight_layout()
       
        # Display the plot window
        plt.show()
       
    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Failed to plot table '{title}': {e}")
        # Stop the program immediately by raising the error
        raise e



def display_descriptive_table(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Calculates and displays a summary table with Mean, Std, and Count for each group.
    This usually shown before the ANOVA analysis.
    """
    try:
        # Log the start of the descriptive statistics calculation
        logger.info(f"Generating descriptive table for '{value_col}' by '{group_col}'")
       
        # Group the data by the category column and calculate Mean, Std Dev, and Count
        summary = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
       
        # Rename the columns
        summary = summary.rename(columns={'mean': 'Mean', 'std': 'Std Dev', 'count': 'N'})
       
        # Round the numerical results to 2 decimal places for cleaner display
        summary = summary.round(2)
       
        # Send the processed summary DataFrame to the helper function to be drawn
        plot_dataframe_as_table(summary, f"Descriptive Statistics: {value_col}")
       
    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Failed to display descriptive table: {e}")
        # Stop the program immediately by raising the error
        raise e



def plot_distributions(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Plots a boxplot overlaid with a stripplot.
    Purpose: To visualize the data distribution, median, and spread BEFORE analysis.
    """
    try:
        # Log the action
        logger.info(f"Plotting distribution of '{value_col}' across '{group_col}'")
       
        # Create a new figure with size 10x6 inches
        plt.figure(figsize=(10, 6))
       
        # Draw a Boxplot: Shows the median, quartiles (25%-75%), and potential outliers
        sns.boxplot(data=df, x=group_col, y=value_col, palette="Set2")
       
        # Draw a Stripplot: Shows the actual data points as dots
        # This helps visualize the sample size and density of the data
        sns.stripplot(data=df, x=group_col, y=value_col, color='black', alpha=0.3, jitter=True)
       
        # Set the main title of the graph
        plt.title(f"Distribution of {value_col} by {group_col}", fontsize=14)
       
        # Label the X-axis (The groups)
        plt.xlabel(group_col, fontsize=12)
       
        # Label the Y-axis (The values)
        plt.ylabel(value_col, fontsize=12)
       
        # Rotate the group names on the X-axis by 45 degrees if they are long
        plt.xticks(rotation=45)
       
        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()
       
        # Show the plot
        plt.show()


    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Failed to plot distribution: {e}")
        # Stop the program immediately by raising the error
        raise e



def display_anova_table(anova_table: pd.DataFrame):
    """
    Takes the ANOVA results DataFrame, formats it, and displays it as an image.
    """
    try:
        # Round all numbers in the ANOVA table to 4 decimal places
        formatted_table = anova_table.round(4)
       
        # Call the helper function to draw the table image
        plot_dataframe_as_table(formatted_table, "ANOVA Results Table")
       
    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Failed to display ANOVA table: {e}")
        # Stop the program immediately by raising the error
        raise e



def display_contrast_weights(weights: dict):
    """
    Takes the weights dictionary, converts it to a DataFrame, and displays it.
    """
    try:
        # Convert the dictionary {'Group': Weight} into a list of items for the DataFrame
        df_weights = pd.DataFrame(list(weights.items()), columns=['Group', 'Weight'])
       
        # Set the 'Group' column as the index
        df_weights.set_index('Group', inplace=True)
       
        # Call the helper function to draw the table
        plot_dataframe_as_table(df_weights, "Planned Contrast Weights")
       
    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Failed to display weights table: {e}")
        # Stop the program immediately by raising the error
        raise e