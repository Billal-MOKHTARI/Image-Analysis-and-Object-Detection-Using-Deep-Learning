import numpy as np
import scipy.stats as st
import pandas as pd

def confidence_interval(data, alpha=0.1):
    """
    Calculate the confidence interval for a given dataset.

    Parameters:
        data (array-like): The dataset for which to calculate the confidence interval.
        alpha (float, optional): The significance level. Default is 0.1.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    conf = 1 - alpha
    return st.t.interval(conf, len(data) - 1,
                        loc=np.mean(data),
                        scale=st.sem(data))

def select_items(data, val_col, alpha=0.1, transform=lambda x: x):
    """
    Select items from a DataFrame based on a confidence interval of transformed values.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        val_col (str): The name of the column containing the values to transform and analyze.
        alpha (float, optional): The significance level for the confidence interval. Default is 0.1.
        transform (function, optional): A function to transform the values before calculating the confidence interval. Default is the identity function.

    Returns:
        DataFrame: A DataFrame containing only the rows within the confidence interval.
    """
    # Apply the transformation to the values in the specified column
    transformed_data = data[val_col].apply(transform) 
    
    # Calculate the confidence interval for the transformed data
    ci = confidence_interval(transformed_data, alpha=alpha)

    # Select rows where the transformed value falls within the confidence interval
    return data[(transformed_data > ci[0]) & (transformed_data < ci[1])]
    
DATA_PATH = "/workspaces/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/6. object_features_with_type.csv"
data = pd.read_csv(DATA_PATH, header=0, index_col=0)
print(select_items(data, "degree", alpha=0.01, transform=lambda x : x))
    