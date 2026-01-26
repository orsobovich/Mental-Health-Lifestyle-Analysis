import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import find_sig
from src.exploration import numeric_ranges,categorical_frequencies, data_info, descriptive_stats
import numpy as np
import pandas as pd
import logging
import textwrap


# Initialize the logger for this specific module
# This logger automatically inherits the configuration (format, level) defined in utils/main
logger = logging.getLogger(__name__)


def heat_map(df):
    """
    Generates and displays a correlation heatmap for numeric columns.
    """


    # Filter the DataFrame to include only numeric columns (Correlation is meaningless for non-numeric data)
    numeric_df = df.select_dtypes(include=['number'])


    # Create matrix of pearson correlation
    corr_matrix = numeric_df.corr()


    # Initialize the figure with a specific size
    plt.figure(figsize=(10, 8))


    # Create the heatmap using Seaborn
    # annot=True: Write the data value in each cell
    # fmt=".2f": Format the annotations to 2 decimal places
    # cmap='coolwarm': Use a diverging color map (blue to red) for better contrast
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")


    plt.title('Correlation Heatmap') # Set the title
    plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
    plt.tight_layout()       # Adjust layout to prevent labels from being clipped
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
            sns.regplot(x=cor_1, y=cor_2, line_kws={'color':'red'}, scatter=False)
           
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
        # Catch any unexpected errors
        logger.error(f"Error plotting correlation: {e}")
        # Stop the program immediately by raising the error
        raise e
   

def plot_dataframe_as_table(df: pd.DataFrame, title: str):
    """
    Renders a pandas DataFrame as a visual table.
    Used by other functions to display ANOVA results and Weights.
    """
    try:
        # Create a new figure and a set of axes with size 8x3 inches
        fig, ax = plt.subplots(figsize=(10, 4))
       
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


        # Automatically wrap long titles into multiple lines (limit to 50 chars per line)
        wrapped_title = "\n".join(textwrap.wrap(title, width=50))
       
        # Set the main figure title using the wrapped text
        plt.suptitle(wrapped_title, fontsize=14, fontweight='bold', y=0.95)
       
        # Automatically adjust subplot parameters to give specified padding
        plt.tight_layout(rect=[0, 0, 1, 0.95])
       
        # Display the plot window
        plt.show()
       
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Failed to plot table '{title}': {e}")
        # Stop the program immediately by raising the error
        raise e


def display_descriptive_table(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Calculates and displays a summary table with Mean, Std, and Count for each group.
    This is shown before the ANOVA analysis.
    """
    try:
        # Log the start of the descriptive statistics calculation
        logger.info(f"Generating descriptive table for '{value_col}' by '{group_col}'")
       
        # Group the data by the category column and calculate Mean, Std Dev, and Count
        summary = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
       
        # Rename the columns
        summary = summary.rename(columns={'mean': 'Mean', 'std': 'Std', 'count': 'N'})
       
        # Round the numerical results to 2 decimal places for cleaner display
        summary = summary.round(2)
       
        # Send the processed summary DataFrame to the helper function to be drawn
        plot_dataframe_as_table(summary, f"Descriptive Statistics: {value_col}")
       
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Failed to display descriptive table: {e}")
        # Stop the program immediately by raising the error
        raise e


def plot_distributions(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Plots a boxplot overlaid with a stripplot to visualize distribution details.
    """
    try:
        # Log the start of the distribution
        logger.info(f"Plotting distribution of '{value_col}' across '{group_col}'")
       
        # Initialize figure with specific dimensions (width=10, height=6 inches)
        plt.figure(figsize=(10, 6))
       
        # Boxplot shows summary stats (median, quartiles)
        sns.boxplot(data=df, x=group_col, y=value_col, hue=group_col, palette="Set2", legend=False)
       
        # Overlay raw data points. jitter adds noise to spread points, alpha adds transparency
        sns.stripplot(data=df, x=group_col, y=value_col, color='black', alpha=0.3, jitter=True)
       
        # Set the title
        plt.title(f"Distribution of {value_col} by {group_col}", fontsize=14)


        # Label the axes
        plt.xlabel(group_col, fontsize=12)
        plt.ylabel(value_col, fontsize=12)
       
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)


        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()


        # Show the plot
        plt.show()


    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Failed to plot distribution: {e}")


def display_anova_table(anova_table: pd.DataFrame, group_name: str, value_name: str):
    """
    Takes the ANOVA results DataFrame, cleans the raw column names (removes statsmodels wrapping),
    formats the numbers, and displays it as an image.
    """
    try:
        # Generate the Title
        title = f"ANOVA Results: {group_name} vs. {value_name}"


        # Create a copy to avoid modifying the original dataframe
        formatted_table = anova_table.copy()


        # Renaming the technical statsmodels column names to professional presentation names
        formatted_table = formatted_table.rename(columns={
            'sum_sq': 'Sum of Squares',  
            'df': 'df',  
            'F': 'F-Statistic',          
            'PR(>F)': 'p-value'          
        })
       
        # Clean index names using Regex (removes statsmodels wrappers)
        formatted_table.index = formatted_table.index.astype(str).str.replace(r'C\(Q\("|"\)\)', '', regex=True)
        formatted_table.index = formatted_table.index.str.replace('Residual', 'Residuals')


        # Round numbers
        formatted_table = formatted_table.round(4)


        # Replace NaN with a space
        formatted_table = formatted_table.fillna('')


        # Draw table
        plot_dataframe_as_table(formatted_table, title)
       
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Failed to display ANOVA table: {e}")
        # Stop the program immediately by raising the error
        raise e
   

def display_contrast_weights(weights, group_name: str):
    """
    Formats the weights and displays a table.
    """
    try:
        # Generate a descriptive title
        title = f"Contrast Weights for {group_name}"


        # Normalize input: Ensure we always work with a DataFrame
        if isinstance(weights, (dict, pd.Series)):
            # Convert dictionary or Series to DataFrame for consistent plotting structure
            weights_df = pd.DataFrame(list(weights.items()), columns=['Group', 'Weight'])
        else:
            # Assume input is already a DataFrame
            weights_df = weights


        # Render the DataFrame as a visual table
        plot_dataframe_as_table(weights_df, title)


    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Failed to display weights table: {e}")


def plot_cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """
    Generates visual report: Missing values (Bar) & Data retention (Pie).
    """
    try:
        # Initialize a side-by-side subplot layout (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
       
        # ---------------------------
        # Left Plot: Missing Values
        # ---------------------------
       
        # Calculate missing values per column and filter only those > 0
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
       
        if not missing.empty:
            # Plot bar chart if missing values exist
            sns.barplot(x=missing.index, y=missing.values, ax=ax1, palette="viridis")
            ax1.set(title="Missing Values (Raw Data)", ylabel="Count (NaNs)", xlabel="Column")
            ax1.tick_params(axis='x', rotation=45) # Rotate labels to prevent overlap
            ax1.bar_label(ax1.containers[0], padding=3, fontweight='bold') # Add exact numbers on bars
        else:
            # If no missing values exist, display a message
            ax1.text(0.5, 0.5, "No Missing Values!", ha='center', va='center', fontsize=12)
            ax1.set_title("Missing Values Check")


        # ---------------------------
        # Right Plot: Data Retention
        # ---------------------------
       
        # Calculate rows kept vs. rows removed
        retained, dropped = len(df_clean), len(df_raw) - len(df_clean)
        labels = [f'Retained\n({retained})', f'Dropped\n({dropped})']
       
        # Pie chart with specific styling:
        # explode=(0, 0.1): Pulls out the 'Dropped' slice to emphasize data loss
        # autopct='%1.1f%%': Formats labels to show one decimal place percentage
        ax2.pie([retained, dropped], explode=(0, 0.1), labels=labels, colors=['#2ecc71', '#e74c3c'],
                autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'})
               
        ax2.set_title("Data Retention Rate", fontsize=14, fontweight='bold')


        # Final layout adjustments
        plt.suptitle("Hybrid Cleaning Impact Report", fontsize=16)
        plt.tight_layout()
        plt.show()
       
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Error generating cleaning report: {e}")


def clean_unused_axes(fig, axes, n_items):
    """
    Helper to remove empty subplots from the grid.
    """
    # Iterate over the remaining axes indices (those beyond the number of actual items)
    for j in range(n_items, len(axes)):
        # Remove the empty subplot so it doesn't appear as a blank box
        fig.delaxes(axes[j])


def plot_numeric_distributions_grid(df: pd.DataFrame):
    """
    Plots a grid of histograms with KDE using precomputed statistics.
    """
    try:
        # Get precomputed numeric statistics (mean, std, etc.)
        stats_df = numeric_ranges(df)


        # Extract numeric column names from stats table index (convert to list)
        numeric_cols = stats_df.index.tolist()


        # Exit early if no numeric data exists to avoid plotting errors
        if not numeric_cols:
            logger.warning("No numeric columns found to plot.")
            return


        # Define round grid size (fixed number of columns)
        cols = 3
        rows = int(np.ceil(len(numeric_cols) / cols))


        # Create subplot grid and converting a 2D axes grid into a 1D array for iteration
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()


        # Loop over numeric columns and assign each one to a subplot (plot space)
        for i, col in enumerate(numeric_cols):
            ax = axes[i]


            # Retrieve mean and standard deviation from stats table
            mean_val = stats_df.loc[col, "mean"]
            std_val = stats_df.loc[col, "std"]


            # Plot histogram with KDE using raw data (shows a smooth estimate of the distribution shape)
            sns.histplot(df[col], kde=True, ax=ax, stat="density", alpha=0.6)


            # Draw vertical line at the mean
            ax.axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")


            # Highlight the ±1 standard deviation range
            ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, label=f"Std Dev: {std_val:.2f}")


            # Set title, axis labels, and legend
            ax.set(title=f"Distribution of {col}", xlabel=col, ylabel="Density")
            ax.legend(fontsize="small")


        # Remove extra axes created by the grid that were not used for plotting
        clean_unused_axes(fig,axes,len(numeric_cols))
       
        # Adjust layout and display the figure
        plt.tight_layout()
        plt.show()


    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Error plotting distributions grid: {e}")
        # Stop the program immediately by raising the error
        raise e
   
   
def plot_categorical_pies(df: pd.DataFrame):
    """
    Plots a pie chart for each categorical variable showing the distribution of values.
    """
    try:
        # Get frequency data (output-Dictionary of DataFrames)
        freq_dict = categorical_frequencies(df)
       
       # Exit early if no categorical data is available
        if not freq_dict:
            logger.warning("No categorical variables to plot.")
            return
       
        # Count total variables to determine the necessary grid size
        num_vars = len(freq_dict)


        # Define round grid size (fixed number of columns)
        cols = 3
        rows = int(np.ceil(num_vars / cols))


        # Create subplots grid
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
       
        # Handle single plot case (where axes is not an array)
        if num_vars == 1:
            axes = [axes]
        else:
            axes = axes.flatten() # Flatten 2D array to 1D for easy iteration


        # Iterate over keys (column names) and values (frequency tables)
        for i, (col_name, df_freq) in enumerate(freq_dict.items()):
            ax = axes[i]
           
            # Extract data for plotting
            # df_freq is a DataFrame with index (categories) and 'count' column
            counts = df_freq['count']
            labels = df_freq.index
           
            # Plot the pie chart: 'autopct' shows %, 'startangle' rotates for better aesthetics
            ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
           
            # Set title for each subplot
            ax.set_title(f'Distribution of {col_name}', fontsize=12, fontweight='bold')


        # Remove extra axes created by the grid that were not used for plotting
        clean_unused_axes(fig,axes,num_vars)


        # Adjust layout to prevent overlaps between subplots
        plt.tight_layout()
        plt.show()
       
        # Log successful execution
        logger.info("Categorical pie charts plotted successfully.")


    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Error plotting categorical pies: {e}")
        # Stop the program immediately by raising the error
        raise e    