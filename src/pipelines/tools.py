import pandas as pd
import numpy as np

def add_prefix_suffix(dataframe, prefix=None, suffix=None):
    """
    Add prefix and/or suffix to DataFrame column names.

    Parameters:
    - dataframe: DataFrame, the input DataFrame
    - prefix: str, optional prefix to add to column names
    - suffix: str, optional suffix to add to column names

    Returns:
    - DataFrame: DataFrame with modified column names if prefix and/or suffix are provided,
                 otherwise returns the input DataFrame unchanged.
    """
    if prefix is not None and suffix is not None:
        # Add both prefix and suffix to column names
        modified_columns = [f"{prefix}{col}{suffix}" for col in dataframe.columns]
    elif prefix is not None:
        # Add only prefix to column names
        modified_columns = [f"{prefix}{col}" for col in dataframe.columns]
    elif suffix is not None:
        # Add only suffix to column names
        modified_columns = [f"{col}{suffix}" for col in dataframe.columns]
    else:
        # No prefix or suffix provided, return the original DataFrame
        return dataframe
    
    # Create a copy of the DataFrame with modified column names
    modified_dataframe = dataframe.copy()
    modified_dataframe.columns = modified_columns
    
    return modified_dataframe

def calculate_co_occurrence_matrix(mat):
    """
    Fill frequency occurrence matrix based on the input matrix.

    Parameters:
        mat (DataFrame): Input matrix.

    Returns:
        DataFrame: Frequency occurrence matrix.
    """
    l_rows, l_columns= mat.shape
    co_occ_mat = np.zeros((l_columns, l_columns))
    for _, row in mat.iterrows():
        indice= np.argwhere(list(row))
        indice = indice.reshape((indice.shape[0],))
        for i in range(len(indice)):
            for j in range(len(indice)):
                co_occ_mat[indice[i], indice[j]]+=1
    # Convert all values to integers
    co_occ_mat = np.array([[int(s) for s in row] for row in co_occ_mat])
    co_occ_mat = pd.DataFrame(co_occ_mat, index=mat.columns, columns=mat.columns)

    return pd.DataFrame(co_occ_mat, index=mat.columns, columns=mat.columns)

def matrix_to_list(adj_matrix_df, node_name = 'Node', neighbor_name = 'Neighbor', weight_name = 'Weight'):
    """
    Converts an adjacency matrix DataFrame to an adjacency list DataFrame.

    Parameters:
    - adj_matrix_df: a pandas DataFrame representing the adjacency matrix

    Returns:
    - adj_list_df: a pandas DataFrame representing the adjacency list
    """
    adj_list = {node_name: [], neighbor_name: [], weight_name: []}
    for i in range(len(adj_matrix_df)):
        node = adj_matrix_df.index[i]
        neighbors = list(adj_matrix_df.columns[adj_matrix_df.iloc[i] != 0])
        for neighbor in neighbors:
            weight = adj_matrix_df.loc[node, neighbor]
            adj_list[node_name].append(node)
            adj_list[neighbor_name].append(neighbor)
            adj_list[weight_name].append(weight)
    adj_list_df = pd.DataFrame(adj_list)
    return adj_list_df
