import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_data(data, feature, saving_dir,group=None, alpha=0.05):
    """
    Perform statistical analysis based on normality, homoscedasticity, 
    and number of groups.

    Parameters:
        data (pd.DataFrame): The dataset containing the variables.
        feature (str): The name of the feature to analyze (dependent variable).
        group (str): The name of the grouping variable (optional for >1 group).
        alpha (float): The significance level for tests. Default is 0.05.

    Returns:
        str: Summary of the appropriate statistical test and results.
    """
    # Helper functions
    def qq_plot(data,saving_dir):
        """Generate a QQ plot for the data."""
        sm.qqplot(data, line='45')
        plt.title('QQ Plot')
        plt.savefig(Path(saving_dir,f"qq_plot_{feature}_{group}.png"))
        #return plt.show()
    
    def homoscedasticity_plot(data, groups):
        """Generate residual variance plot for homoscedasticity."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=groups, y=data)
        plt.title('Homoscedasticity Check')
        plt.xlabel('Group')
        plt.ylabel('Feature')

        return plt.show()

    # Normality test
    def check_normality(data):
        stat, p_value = stats.shapiro(data)
        return p_value > .2, p_value

    # Homoscedasticity test
    def check_homoscedasticity(data, groups):
        groups = pd.Categorical(groups).codes
        stat, p_value = stats.levene(data, groups)
        return p_value > alpha, p_value

    # Begin analysis
    if group:
        groups = data[group].unique()
        group_data = [data[data[group] == g][feature] for g in groups]
        
        if all([gd.nunique() == 1 for gd in group_data]):
            return {"Test": "All values are equal", "Statistic": None, "P-value": None, "conclusion": "All values are equal."}
        if len(groups) == 1:
            return {"Test": "Single group", "Statistic": None, "P-value": None, "conclusion": "Single group."}
        # Normality check for each group
        normality_results = {g: check_normality(gd) for g, gd in zip(groups, group_data)}
        #print("Normality Results by Group:", normality_results)

        # Check homoscedasticity
        flat_data = np.concatenate(group_data)
        group_labels = np.concatenate([[g] * len(gd) for g, gd in zip(groups, group_data)])
        homoscedastic, homo_p = check_homoscedasticity(flat_data, group_labels)
        #print(f"Homoscedasticity: {'Passed' if homoscedastic else 'Failed'} (p={homo_p:.4f})")

        # Generate plots
        #for g, gd in zip(groups, group_data):
            #print(f"QQ Plot for Group {g}")
            #qq_plot(gd,saving_dir)

        # Choose statistical test
        #Check if all values are equal for all groups

        if all(res[0] for res in normality_results.values()) and homoscedastic:
            if len(groups) == 2:
                test_stat, test_p = stats.ttest_ind(*group_data)
                test_name = "Independent T-Test"
            else:
                test_stat, test_p = stats.f_oneway(*group_data)
                test_name = "One-Way ANOVA"
        else:
            if len(groups) == 2:
                test_stat, test_p = stats.mannwhitneyu(*group_data)
                test_name = "Mann-Whitney U Test"
            else:
                test_stat, test_p = stats.kruskal(*group_data)
                test_name = "Kruskal-Wallis Test"
    else:
        # Single group: test for normality
        normal, norm_p = check_normality(data[feature])
        #print(f"Normality: {'Passed' if normal else 'Failed'} (p={norm_p:.4f})")
        #print("QQ Plot:")
        #qq_plot(data[feature],saving_dir)

        # Choose test based on normality
        if normal:
            test_stat, test_p = stats.ttest_1samp(data[feature], 0)
            test_name = "One-Sample T-Test"
        else:
            test_stat, test_p = stats.wilcoxon(data[feature])
            test_name = "Wilcoxon Signed-Rank Test"

    # Output results
    result_summary = {"Test": test_name, "Statistic": test_stat, "P-value": test_p}
    
    return result_summary

def stat_describe(data, feature, group=None):
    """
    Perform statistical analysis on a dataset.

    Parameters:
        data (pd.DataFrame): The dataset containing the variables.
        feature (str): The name of the feature to analyze (dependent variable).
        group (str): The name of the grouping variable (optional for >1 group).

    Returns:
        pd.DataFrame: Summary statistics of the data.
    """
    if data.empty:
        return None
    
    if group:
        groups = data[group].unique()
        group_data = [data[data[group] == g][feature] for g in groups]

        # Generate summary statistics for each group
        summary_stats = pd.concat([gd.describe().rename(g) for g, gd in zip(groups, group_data)], axis=1)
        summary_stats.columns = groups
        #Round the values
        summary_stats = summary_stats.round(3)

    else:
        summary_stats = data[feature].describe().to_frame().T

    return summary_stats

def descriptive_plots(data, feature, path_to_save,group=None):
    """
    Generate descriptive plots for the data.

    Parameters:
        data (pd.DataFrame): The dataset containing the variables.
        feature (str): The name of the feature to analyze (dependent variable).
        group (str): The name of the grouping variable (optional for >1 group).

    Returns:
        None
    """
    if group:
        # Generate violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=group, y=feature, data=data, hue=group, palette='Set2', linewidth=1, width=0.8, saturation=0.75, inner='quartile')
        plt.title(f'Violin Plot of {feature} by Group')
        plt.xlabel('Group')
        plt.ylabel(feature)
        
        plt.savefig(Path(path_to_save,f'violin_plot_{feature}_{group}.png'))

    # Generate histogram
    plt.figure(figsize=(8, 6))
    try:
        sns.histplot(data[[group,feature]],x=feature,hue=group,kde=True, color='skyblue')
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(Path(path_to_save,f'hist_plot_{feature}_{group}.png'))
    except:
        pass
