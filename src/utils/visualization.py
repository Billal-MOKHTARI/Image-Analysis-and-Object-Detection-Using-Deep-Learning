import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def show_distribution(data: pd.DataFrame, col: str, grouped_by:str=None, xlabel:str=None, ylabel:str=None, save_path:str=None, title:str = None, **kwargs):
    """
    Plot the distribution of data using histograms.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        col (str): The name of the column to plot.
        grouped_by (str, optional): The column to group the data by. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        save_path (str, optional): Filepath to save the plot. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        **kwargs: Additional keyword arguments to customize the plot.

    Returns:
        None
    """
    # Extracting keyword arguments with default values
    style = kwargs.get("style", 'darkgrid')
    figsize = kwargs.get("figsize", (8, 6))
    bins = kwargs.get("bins", 40)
    kde = kwargs.get("kde", True)
    color = kwargs.get("color", 'darkcyan')
    dpi = kwargs.get("dpi", 300)
    
    # Set seaborn style
    sns.set(style=style)

    # Plot the distribution using a histogram
    if grouped_by is None:
        plt.figure(figsize=figsize)
        sns.histplot(data[col], kde=kde, color=color, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        # Iterate over groups
        for group_name, group_data in data.groupby(grouped_by):
            sns.histplot(group_data[col], kde=kde, bins=bins, label=group_name, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title=title)
    
    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
        
    plt.show()