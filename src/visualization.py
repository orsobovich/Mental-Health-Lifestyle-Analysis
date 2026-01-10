import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr


 # Create a correlation plot
def plot_correlation(cor_1, cor_2, p_value): 
    if find_sig(p_value):
        sns.regplot(x=cor_1, y=cor_2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title("Correlation Plot")
        plt.xlabel(cor_1.name) #Takes the name of the column
        plt.ylabel(cor_2.name) #Takes the name of the column
        plt.show()