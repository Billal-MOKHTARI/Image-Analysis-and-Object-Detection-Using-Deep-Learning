import numpy as np
import pandas as pd

def remove_zero_columns(data):
    """
    Remove columns that are full of zeros from a NumPy array or a Pandas DataFrame.

    Args:
    - data: NumPy array or Pandas DataFrame

    Returns:
    - Filtered data without columns full of zeros
    """
    if isinstance(data, np.ndarray):
        # For NumPy array
        non_zero_columns = np.any(data != 0, axis=0)
        filtered_data = data[:, non_zero_columns]
        return filtered_data
    elif isinstance(data, pd.DataFrame):
        # For Pandas DataFrame
        non_zero_columns = data.columns[data.any()]
        filtered_data = data[non_zero_columns]
        return filtered_data
    else:
        raise ValueError("Input must be a NumPy array or a Pandas DataFrame")


def calculate_co_occurrence_matrix(data):
    """
    Calculate the co-occurrence matrix for a dataset.

    Parameters:
    - data: DataFrame, the input dataset

    Returns:
    - co_occurrence_matrix: DataFrame, the co-occurrence matrix
    """
    # Initialize an empty co-occurrence matrix
    co_occurrence_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=int).fillna(0)

    # Iterate through each row of the dataset
    for _, row in data.iterrows():
        # Iterate through each pair of columns
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                # Check if both values are non-zero
                if i != j and row[data.columns[i]] != 0 and row[data.columns[j]] != 0:
                    # Increment the co-occurrence count for the pair of columns
                    co_occurrence_matrix.at[data.columns[i], data.columns[j]] += 1

        # Increment the diagonal for each non-zero value in the row
        for i in range(len(data.columns)):
            if row[data.columns[i]] != 0:
                co_occurrence_matrix.at[data.columns[i], data.columns[i]] += 1
    # Convert all values to integers
    co_occurrence_matrix = co_occurrence_matrix.astype(int)

    return co_occurrence_matrix


def matrix_to_list(adj_matrix_df):
    """
    Converts an adjacency matrix DataFrame to an adjacency list DataFrame.

    Parameters:
    - adj_matrix_df: a pandas DataFrame representing the adjacency matrix

    Returns:
    - adj_list_df: a pandas DataFrame representing the adjacency list
    """
    adj_list = {'Node': [], 'Neighbor': [], 'Weight': []}

    for i in range(len(adj_matrix_df)):
        node = adj_matrix_df.index[i]
        neighbors = list(adj_matrix_df.columns[adj_matrix_df.iloc[i] != 0])

        for neighbor in neighbors:
            weight = adj_matrix_df.loc[node, neighbor]
            adj_list['Node'].append(node)
            adj_list['Neighbor'].append(neighbor)
            adj_list['Weight'].append(weight)

    adj_list_df = pd.DataFrame(adj_list)

    return adj_list_df
