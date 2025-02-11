import numpy as np
import pandas as pd
import re
from datetime import datetime
import json
from urllib.request import urlopen
from urllib.parse import urlparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dataset import data_preprocessing

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def use_absolute_path(path, use_abs_path):
    if use_abs_path:
        return os.path.abspath(path)
    return path

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def remove_zero_columns(data):
    """
    Remove columns that are full of zeros from a NumPy array or a Pandas DataFrame.

    Args:
    - data: NumPy array or Pandas DataFrame

    Returns:
    - Filtered data without columns full of zeros
    """
    if isinstance(data, np.ndarray):
        non_zero_columns = np.any(data != 0, axis=0)
        filtered_data = data[:, non_zero_columns]
        return filtered_data
    elif isinstance(data, pd.DataFrame):
        non_zero_columns = data.columns[data.any()]
        filtered_data = data[non_zero_columns]
        return filtered_data
    else:
        raise ValueError("Input must be a NumPy array or a Pandas DataFrame")


def closeness_centrality(matrix):
    pass

def unary_degree(adj_list, node):
    degrees = adj_list[node].value_counts()
    return pd.DataFrame({node: degrees.index, "unary degree": degrees.tolist()})

def degree(adjacency_matrix):
    degrees = adjacency_matrix.sum(axis=1)
    return pd.DataFrame({"Node":degrees.index, "degree":degrees.tolist()})

def extract_diagonal(matrix_df, column_names):
    """
    Extracts the index of a squared matrix DataFrame and its diagonal values.

    Parameters:
    - matrix_df: DataFrame, the squared matrix DataFrame

    Returns:
    - diagonal_df: DataFrame, a DataFrame containing the index and diagonal values
    """
    # Get the index of the matrix
    index = matrix_df.index
    
    # Extract the diagonal values
    diagonal_values = matrix_df.values.diagonal()
    
    # Create a DataFrame with index and diagonal values
    dict_data = {column_names[0]: index, column_names[1]: diagonal_values}
    diagonal_df = pd.DataFrame(dict_data)
    
    return diagonal_df


# Metadata preprocessing helper functions
def is_matching_regex(data, pattern):
    """
    Match a regular expression pattern to the values in a DataFrame.

    Parameters:
    - data: DataFrame, the input dataset
    - pattern: str, the regular expression pattern

    Returns:
    - matched_data: DataFrame, the subset of the dataset that matches the pattern
    """
    return bool(re.fullmatch(pattern, data))

def focal_length_equivalent(data, column='FocalLength35efl'):
    """
    Calculate the focal length equivalent for a dataset.

    Parameters:
    - data: DataFrame, the input dataset
    - column: str, the column name for the focal length equivalent
    """
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

def first_valid_value(data):
    """
    Find the first valid value in a DataFrame.

    Parameters:
    - data: Series, the input dataset

    Returns:
    - first_valid_value: any, the first valid value in the dataset
    """
    index = data.first_valid_index()
    if index is not None:
        return data.loc[index]
    else:
        return None

def convert_to_datetime(datetime_str, date_format = "%d/%m/%Y %H:%M:%S %z"):
    try:
        datetime_object = datetime.strptime(datetime_str, date_format)
        return datetime_object
    except ValueError:
        print("Invalid datetime format. Please provide datetime in format DD/MM/YYYY hh:mm:ss +01:00")
        return None

def permissions_to_int(permissions):
    pass

def load_json_file_from_path(file_path):
    """
    Load a JSON file.

    Parameters:
    - file_path: str, the path to the JSON file

    Returns:
    - data: dict, the loaded JSON data
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_json_file_from_url(url):
    """
    Load a JSON file from a URL.

    Parameters:
    - url: str, the URL of the JSON file

    Returns:
    - data_json: dict, the loaded JSON data
    """
    response = urlopen(url)
    data_json = json.loads(response.read())
    return data_json

def load_json_file(file_path):
    try:
        if is_url(file_path):
            return load_json_file_from_url(file_path)
        else:
            return load_json_file_from_path(file_path)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def is_url(path):
    """
    Check if a given string is a valid URL.

    Parameters:
    - path: str, the string to check

    Returns:
    - bool: True if the string is a valid URL, False otherwise
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def preprocess_image_metadata(metadata, configs):
    dpp = data_preprocessing.DataPreprocessing(metadata)  # Assuming DataPreprocessing class is imported as DataPreprocessing

    # Load the metadata
    methods = configs.keys()

    for method_name in methods:
        method_params = configs[method_name]

        # Call the method dynamically
        method = getattr(dpp, method_name)
        metadata = method(**method_params)
    return metadata
    

# dpp.delete_empty_columns(1-17/177)
# filtered_data = drop_columns(columns=['Directory', 'FocalLength35efl', 'GPSDateTime', 'GPSPosition'])
# filtered_data = set_index_from_column('FileName')
# # filtered_data = focal_length_equivalent(filtered_data)
# filtered_data = bring_up_measure_units()
# filtered_data = convert_datetime(filtered_data)
# filtered_data = convert_fraction_columns_to_float(filtered_data)
# filtered_data = split_string_column(filtered_data, 'ImageSize', ['ImageWidth', 'ImageHeight'])
# filtered_data = gps_dms_to_dd(filtered_data)
# filtered_data = modify_column(filtered_data, 'YCbCrSubSampling', ['YCbCr'], ['\(\d+ \d+\)'], None)
# filtered_data = remove_prefix(filtered_data, 'MIMEType', 'image/')
# Open the JSON file

# url = 'https://raw.githubusercontent.com/Billal-MOKHTARI/Image-Analysis-and-Object-Detection-Using-Deep-Learning/main/configs/preprocess_args.json'
# print(load_json_file_from_url(url))
