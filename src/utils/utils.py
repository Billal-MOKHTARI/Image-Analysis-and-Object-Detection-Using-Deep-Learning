import numpy as np
import pandas as pd
import constants
import re

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

def closeness_centrality(matrix):
    pass

def unary_degree(adj_list, node):
    degrees = adj_list[node].value_counts()
    return pd.DataFrame({node: degrees.index, "unary degree": degrees.tolist()})

def degree(adjacency_matrix):
    # Assuming the dataframe contains the adjacency matrix
    # Summing along the rows to calculate the degree of each node
    degrees = adjacency_matrix.sum(axis=1)
    return pd.DataFrame({"Node":degrees.index, "degree":degrees.tolist()})

def delete_empty_columns(data, threshold, delete_by=['nan']):
    """
    Delete columns with a percentage of zeros greater than the threshold.

    Parameters:
    - data: DataFrame, the input dataset
    - threshold: float, the threshold for the percentage of zeros

    Returns:
    - filtered_data: DataFrame, the filtered dataset
    """
    columns_to_delete = []
    for col in data.columns:
        for val in delete_by:
            values = data[col].astype(str)
            percentage = values[values == val].count() / len(values)
            if percentage > threshold:
                columns_to_delete.append(col)
    data.drop(columns=columns_to_delete, inplace=True)

    return data

def is_matching_regex(data, pattern=constants.unit_regex_pattern):
    """
    Match a regular expression pattern to the values in a DataFrame.

    Parameters:
    - data: DataFrame, the input dataset
    - pattern: str, the regular expression pattern

    Returns:
    - matched_data: DataFrame, the subset of the dataset that matches the pattern
    """
    
    if re.fullmatch(pattern, data) is not None:
        return True
    else:
        return False

def bring_up_measure_units(dataframe):
    """
    Bring up the measure units from column values to column names.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing columns with measure units.

    Returns:
        pandas.DataFrame: The DataFrame with measure units brought up to column names.
    """
    def extract_measure_unit(cell):
        """
        Extract the measure unit from a cell value.

        Args:
            cell (str): The cell value.

        Returns:
            str: The measure unit extracted from the cell value.
        """
        if pd.isna(cell):
            return None
        else:
            # Extract measure unit by splitting the cell value
            parts = cell.split(' ')
            if len(parts) > 1:
                return parts[-1]  # Return the last part as measure unit
            else:
                return None
    
    def update_column_name(col_name, measure_unit):
        """
        Update column name with measure unit.

        Args:
            col_name (str): The original column name.
            measure_unit (str): The measure unit.

        Returns:
            str: The updated column name with measure unit.
        """
        if measure_unit:
            # Check if the column name already contains parenthesis
            if '(' in col_name and ')' in col_name:
                # If it does, replace the content inside the parenthesis
                return col_name[:col_name.find('(')] + f'({measure_unit})'
            else:
                # If not, add the measure unit in parenthesis
                return f'{col_name} ({measure_unit})'
        else:
            return col_name
    
    # Iterate through columns
    for col in dataframe.columns:
        if dataframe[col].dtype != 'object':
            continue
        # Extract measure unit from the first non-null cell value in the column
        measure_unit = extract_measure_unit(dataframe[col].dropna().iloc[0])
        # Update column name with measure unit
        dataframe.rename(columns={col: update_column_name(col, measure_unit)}, inplace=True)
    
    return dataframe

data = pd.read_csv('/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/metadata/image_metadata.csv', 
                   index_col=0,
                   header=0)
# print(data['GainControl'].astype(str).iloc[1:13])
filtered_data = delete_empty_columns(data, 1-17/177)
transformed_data = bring_up_measure_units(filtered_data)
print(transformed_data.columns)
