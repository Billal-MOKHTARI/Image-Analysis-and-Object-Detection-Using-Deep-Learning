import numpy as np
import pandas as pd
import constants
import re
from fractions import Fraction
from datetime import datetime

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
    
    return bool(re.fullmatch(pattern, data))

def bring_up_measure_units(dataframe, exclude=[]):
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
            # Check if the measure unit matches the regex pattern
            if is_matching_regex(measure_unit):
                # If it does, add the measure unit in parenthesis
                return f'{col_name} ({measure_unit})'
            else:
                # If not, return the original column name
                return col_name
        else:
            return col_name
    
    # Iterate through columns
    for col in dataframe.columns:
        if col not in exclude and dataframe[col].dtype == 'object':
            
            # Extract measure unit from the first non-null cell value in the column
            measure_unit = extract_measure_unit(dataframe[col].dropna().iloc[0])
            # Update column name with measure unit
            dataframe.rename(columns={col: update_column_name(col, measure_unit)}, inplace=True)
    
    return dataframe

def focal_length_equivalent(data, column='FocalLength35efl'):
    """
    Calculate the focal length equivalent for a dataset.

    Parameters:
    - data: DataFrame, the input dataset
    - column: str, the column name for the focal length equivalent
    """
    # Extract the focal length and crop factor from the column name
    focal_length = data[column]

    parts = focal_length.str.split(' ')
    focal_length = parts.str[-2:-1][0]

    data[column] = focal_length

    return data

def extract_matching_part(exp, regex):
    """
    Extract the part of a string that matches a regular expression.

    Parameters:
    - exp: str, the input string
    - regex: str, the regular expression pattern

    Returns:
    - matched_part: str, the part of the string that matches the regular expression
    """
    matched_part = re.search(regex, exp).group(0)
    return matched_part

def to_nan(data, values=['Unknown (0)']):
    """
    Convert specified values to NaN in a DataFrame.

    Parameters:
    - data: DataFrame, the input dataset
    - values: list, the values to convert to NaN

    Returns:
    - data: DataFrame, the dataset with specified values converted to NaN
    """
    data.replace(values, np.nan, inplace=True)
    return data

def first_valid_value(data):
    """
    Find the first valid value in a DataFrame.

    Parameters:
    - data: Series, the input dataset

    Returns:
    - first_valid_value: any, the first valid value in the dataset
    """
    # Get the index of the first non-NaN value
    index = data.first_valid_index()

    # If the series is not entirely composed of NaN values
    if index is not None:
        # Return the first non-NaN value
        return data.loc[index]
    else:
        return None

def convert_to_datetime(datetime_str, date_format = "%d/%m/%Y %H:%M:%S %z"):
    # Define the format of the input datetime string

    try:
        # Convert the string to a datetime object
        datetime_object = datetime.strptime(datetime_str, date_format)
        return datetime_object
    except ValueError:
        # Handle invalid datetime string
        print("Invalid datetime format. Please provide datetime in format DD/MM/YYYY hh:mm:ss +01:00")
        return None

def convert_datetime(data, regex_from=r'\d{4}:\d{2}:\d{2} (\d{2}:\d{2}:\d{2})?(\+\d{2}:\d{2})?'):

    columns = data.columns
    columns_to_convert = [col for col in columns if is_matching_regex(str(first_valid_value(data[col])), regex_from)]
    with_tz = [col for col in columns_to_convert if re.search(r'\+\d{2}:\d{2}', str(first_valid_value(data[col])))]
    without_tz = [col for col in columns_to_convert if col not in with_tz]
    date_only_matching = [col for col in columns if is_matching_regex(str(first_valid_value(data[col])), r'\d{4}:\d{2}:\d{2}')]


    for col in without_tz:
        # Convert the column to datetime format
        data[col] = pd.to_datetime(data[col], format='%Y:%m:%d %H:%M:%S', errors='coerce')
        # Change the datetime format to 'DD/MM/YYYY hh:mm:ss'
        data[col] = data[col].dt.strftime('%d/%m/%Y %H:%M:%S')

    for col in with_tz:
        format = "%d/%m/%Y %H:%M:%S %z"

        datetime_part = data[col].str.extract(r'(\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2})')[0]
        timezone_part = data[col].str.extract(r'(\+\d{2}:\d{2})')[0]
        
        datetime_part = pd.to_datetime(datetime_part, format='%Y:%m:%d %H:%M:%S', errors='coerce')
        datetime_part = datetime_part.dt.strftime('%d/%m/%Y %H:%M:%S')

        data[col] = datetime_part + ' ' +timezone_part
        data[col] = data[col].apply(lambda x: convert_to_datetime(x, date_format=format))

    for col in date_only_matching:
        data[col] = pd.to_datetime(data[col], format='%Y:%m:%d', errors='coerce')
        data[col] = data[col].dt.strftime('%d/%m/%Y')


    return data

def drop_columns(data, columns):
    """
    Drop columns from a DataFrame.

    Parameters:
    - data: DataFrame, the input dataset
    - columns: list, the columns to drop

    Returns:
    - data: DataFrame, the dataset with specified columns dropped
    """
    data.drop(columns=columns, inplace=True)
    return data

def convert_fraction_columns_to_float(df):
    # Function to convert string fractions to floats
    def string_fraction_to_float(fraction_str):
        try:
            fraction = Fraction(fraction_str)
            return float(fraction)
        except ValueError:
            return fraction_str  # Return original value if it cannot be converted
    
    # Iterate over each column in the DataFrame
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if column type is string
            try:
                # Try converting all values in the column to float
                df[col] = df[col].apply(string_fraction_to_float)
            except Exception as e:
                print(f"Error converting column '{col}': {e}")
    
    return df

def permissions_to_int(permissions):
    pass

def set_index_from_column(data, column_name, index_name=None):
    """
    Set the index of the DataFrame using the specified column.

    Parameters:
    - data: DataFrame
        The DataFrame for which the index should be set.
    - column_name: str
        The name of the column to be used as the index.
    - index_name: str, optional
        The name to be assigned to the index. If None, no name is assigned.

    Returns:
    - DataFrame
        The DataFrame with the specified column set as the index.
    """
    if column_name not in data.columns:
        print(column_name)
        print(f"Column '{column_name}' not found in DataFrame.")
        return data
    else:
        data.set_index(column_name, inplace=True)
        if index_name:
            data.index.name = index_name
        return data

# Test
data = pd.read_csv('/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/metadata/image_metadata.csv', 
                   index_col=0,
                   header=0)
filtered_data = data.copy()
filtered_data = set_index_from_column(filtered_data, 'FileName')
filtered_data = to_nan(filtered_data, ['Unknown (0)'])
filtered_data = delete_empty_columns(data, 1-17/177)
filtered_data = focal_length_equivalent(filtered_data)
filtered_data = bring_up_measure_units(filtered_data)
filtered_data = convert_datetime(filtered_data)
filtered_data = drop_columns(filtered_data, columns=['Directory', 'FocalLength35efl', 'GPSDateTime', 'GPSPosition'])
filtered_data = convert_fraction_columns_to_float(filtered_data)

print(data.columns[:50])