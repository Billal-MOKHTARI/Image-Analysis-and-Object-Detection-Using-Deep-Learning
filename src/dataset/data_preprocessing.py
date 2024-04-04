import pandas as pd
import numpy as np
from datetime import datetime
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import utils

class DataPreprocessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def drop_columns(self, columns):
        """
        Drop columns from a DataFrame.

        Parameters:
        - data: DataFrame, the input dataset
        - columns: list, the columns to drop

        """
        self.data.drop(columns=columns, inplace=True)

    def set_index_from_column(self, column_name, index_name=None):
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
        if column_name not in self.data.columns:
            print(column_name)
            print(f"Column '{column_name}' not found in DataFrame.")
 
        else:
            self.data.set_index(column_name, inplace=True)
            if index_name:
                self.data.index.name = index_name

    def delete_empty_columns(self, threshold, delete_by):
        """
        Delete columns with a percentage of zeros greater than the threshold.

        Parameters:
        - data: DataFrame, the input dataset
        - threshold: float, the threshold for the percentage of zeros

        Returns:
        - filtered_data: DataFrame, the filtered dataset
        """
        for val in delete_by:
            self.data.replace(val, np.nan, inplace=True)
        
        columns_to_delete = []
        for col in self.data.columns:
            for val in delete_by:
                percentage = self.data[col].isnull().sum() / len(self.data[col])
                if percentage > threshold:
                    columns_to_delete.append(col)
        self.data.drop(columns=columns_to_delete, inplace=True)

    def bring_up_measure_units(self, exclude=[]):
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
                if utils.is_matching_regex(measure_unit):
                    # If it does, add the measure unit in parenthesis
                    return f'{col_name} ({measure_unit})'
                else:
                    # If not, return the original column name
                    return col_name
            else:
                return col_name
        
        # Iterate through columns
        for col in self.data.columns:
            if col not in exclude and self.data[col].dtype == 'object':
                
                # Extract measure unit from the first non-null cell value in the column
                measure_unit = extract_measure_unit(self.data[col].dropna().iloc[0])
                # Update column name with measure unit
                self.data.rename(columns={col: update_column_name(col, measure_unit)}, inplace=True)
        
        return self.data
    
    def convert_datetime(self, regex_from=r'\d{4}:\d{2}:\d{2} (\d{2}:\d{2}:\d{2})?(\+\d{2}:\d{2})?'):

        columns = self.data.columns
        columns_to_convert = [col for col in columns if utils.is_matching_regex(str(utils.first_valid_value(self.data[col])), regex_from)]
        with_tz = [col for col in columns_to_convert if re.search(r'\+\d{2}:\d{2}', str(utils.first_valid_value(self.data[col])))]
        without_tz = [col for col in columns_to_convert if col not in with_tz]
        date_only_matching = [col for col in columns if utils.is_matching_regex(str(utils.first_valid_value(self.data[col])), r'\d{4}:\d{2}:\d{2}')]


        for col in without_tz:
            # Convert the column to datetime format
            self.data[col] = pd.to_datetime(self.data[col], format='%Y:%m:%d %H:%M:%S', errors='coerce')
            # Change the datetime format to 'DD/MM/YYYY hh:mm:ss'
            self.data[col] = self.data[col].dt.strftime('%d/%m/%Y %H:%M:%S')

        for col in with_tz:
            format = "%d/%m/%Y %H:%M:%S %z"

            datetime_part = self.data[col].str.extract(r'(\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2})')[0]
            timezone_part = self.data[col].str.extract(r'(\+\d{2}:\d{2})')[0]
            
            datetime_part = pd.to_datetime(datetime_part, format='%Y:%m:%d %H:%M:%S', errors='coerce')
            datetime_part = datetime_part.dt.strftime('%d/%m/%Y %H:%M:%S')

            self.data[col] = datetime_part + ' ' +timezone_part
            self.data[col] = self.data[col].apply(lambda x: utils.convert_to_datetime(x, date_format=format))

        for col in date_only_matching:
            self.data[col] = pd.to_datetime(self.data[col], format='%Y:%m:%d', errors='coerce')
            self.data[col] = self.data[col].dt.strftime('%d/%m/%Y')

    def split_string_column(self, column_name, new_column_names, keep=False, separator='x'):
        """
        Split a string column into multiple columns based on a separator.

        Parameters:
        - data: DataFrame, the input dataset
        - column_name: str, the name of the column to split
        - separator: str, the separator to split the column on

        Returns:
        - data: DataFrame, the dataset with the column split into multiple columns
        """
        # Split the column into multiple columns based on the separator
        split_data = self.data[column_name].str.split(separator, expand=True)
        

        # Add the new columns to the original dataset
        self.data = pd.concat([self.data, split_data], axis=1)

        dict_names = {i: new_column_names[i] for i in range(len(new_column_names))}
        self.data.rename(columns=dict_names, inplace=True)
        if not keep:
            self.data.drop(columns=[column_name], inplace=True)

    def modify_column(self, column_name, possible_prefixes, possible_suffixes, bracket_content):
        # Remove possible prefixes
        for prefix in possible_prefixes:
            self.data[column_name] = self.data[column_name].str.replace(prefix, '')
        
        # Remove possible suffixes
        for suffix in possible_suffixes:
            self.data[column_name] = self.data[column_name].str.replace(suffix, '')

        if bracket_content:
            new_column_name = f"{column_name} ({bracket_content})"
        else:
            new_column_name = column_name
        
        
        new_column_name = f"{column_name} {bracket_content}"
        self.data.rename(columns={column_name: new_column_name}, inplace=True)

    def remove_prefix(self, column_name, prefix):
        self.data[column_name] = self.data[column_name].str.replace(prefix, '')
    


