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

def select_items_ci(data, val_col, alpha=0.1, transform=lambda x: x):
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
    
def select_frequent_items_by_number(data:pd.DataFrame, measure:str, number:int, ascending=False, cols=None):
    sorted_data = data.sort_values(by=measure, ascending=ascending)
    result = sorted_data[:number]
    
    if cols is not None :
        result = result[cols]
        
    return result
    
def select_items_by_interval(data:pd.DataFrame, measure, lb, ub, cols=None):
    measure_values = data[measure]
    result = data[(measure_values > lb) & (measure_values < ub)]
    
    if cols is not None:
        result = result[cols]
        
    return result

def combine_dataframes(dataframes, operator='intersection'):
    if operator == 'intersection':
        # Initialize intersection_df as the first DataFrame in dataframes
        intersection_df = dataframes[0]

        # Iterate over the remaining DataFrames in dataframes and perform inner merge
        for df in dataframes[1:]:
            intersection_df = pd.merge(intersection_df, df, how='inner')

        # Resulting intersection_df will contain only the rows that are common across all DataFrames in dataframes
        return intersection_df

    elif operator == 'union':
        # Concatenate the DataFrames in dataframes
        result_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()
        return result_df

    else:
        return None  # Handle invalid operator

def select_frequent_items_measures(data, methods:dict, operator="intersection", cols=None):
    ids = methods.keys()
    args = methods.values()
    frequent_items = []
    for ids, arg in zip(ids, args):
        
        if arg["by"] == "number":
            frequent_items.append(select_frequent_items_by_number(data, 
                                                                arg["measure"], 
                                                                arg["number"], 
                                                                arg["ascending"],
                                                                cols=cols))
        elif arg["by"] == "interval":
            frequent_items.append(select_items_by_interval(data,
                                                            arg["measure"],
                                                            arg["lb"],
                                                            arg["ub"],
                                                            cols=cols))
    
    result = combine_dataframes(frequent_items, operator=operator)
    
    return result 
    
        
    

DATA_PATH = "/workspaces/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/6. object_features_with_type.csv"
data = pd.read_csv(DATA_PATH, header=0, index_col=0)
print(select_frequent_items_measures(data,
                                    methods={"1": {"by": "interval", "measure": "degree", "lb":80, "ub": 102},
                                            "2": {"by": "interval", "measure": "pageRank", "lb":0.5, "ub": 1},
                                            "3": {"by": "interval", "measure": "betweenness", "lb":1000, "ub": 15000}}, operator="union"))
