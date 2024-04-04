import sys
import os
import json
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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

def create_co_occurrence_graphxr_dataset(annot_mat_path, 
                                        label_categories_path, 
                                        URL_base,
                                        node_name="Node",
                                        neighbor_name="Neighbor",
                                        weight_name="Weight",
                                        save = True,
                                        graphxr_dataset_configs_path=None,
                                        **kwargs):

    # Read label categories and annotation matrix
    label_categories = pd.read_csv(label_categories_path, header=0, index_col=0)
    obj_annot_mat = pd.read_csv(annot_mat_path, header=0, index_col=0)

    
    # Extract stuff and thing categories so that we can identify the type of relationship between the objects
    stuff = label_categories[label_categories["Type"] == "stuff"][node_name].tolist()
    thing = label_categories[label_categories["Type"] == "thing"][node_name].tolist()

    # Calculate co-occurrence matrix and convert to list
    obj_co_occ_matrix = utils.calculate_co_occurrence_matrix(obj_annot_mat)
    obj_co_occ_list = utils.matrix_to_list(obj_co_occ_matrix, node_name=node_name, neighbor_name=neighbor_name, weight_name=weight_name)

    # Identify TT relationships
    tt_mask = obj_co_occ_list[node_name].isin(thing) & obj_co_occ_list[neighbor_name].isin(thing)
    obj_co_occ_list.loc[tt_mask, 'LinkType'] = 'TT'

    # Identify SS relationships
    ss_mask = obj_co_occ_list[node_name].isin(stuff) & obj_co_occ_list[neighbor_name].isin(stuff)
    obj_co_occ_list.loc[ss_mask, 'LinkType'] = 'SS'

    # Identify ST relationships (rest of the cases)
    obj_co_occ_list.loc[(~tt_mask) & (~ss_mask), 'LinkType'] = 'ST'

    # Extract node weights for object nodes
    obj_node_features = utils.extract_diagonal(obj_co_occ_matrix, [node_name, "NodeWeight"])
    obj_node_features = pd.merge(label_categories, obj_node_features, on=node_name, how="right")

    # Calculate co-occurrence matrix for image nodes and convert to list
    img_annot_mat = obj_annot_mat.copy().T
    img_co_occ_matrix = utils.calculate_co_occurrence_matrix(img_annot_mat)
    img_co_occ_list = utils.matrix_to_list(img_co_occ_matrix, node_name=node_name, neighbor_name=neighbor_name, weight_name=weight_name)

    # Extract node weights for image nodes
    img_node_features = utils.extract_diagonal(img_co_occ_matrix, [node_name, "NodeWeight"])
    img_node_features["URL"] = URL_base + img_node_features[node_name]

    obj_prefix = 'object_'
    obj_node_features = add_prefix_suffix(obj_node_features, prefix=obj_prefix)
    obj_co_occ_list = add_prefix_suffix(obj_co_occ_list, prefix=obj_prefix)

    img_prefix = 'image_'
    img_node_features = add_prefix_suffix(img_node_features, prefix=img_prefix)
    img_co_occ_list = add_prefix_suffix(img_co_occ_list, prefix=img_prefix)

    obj_img_occ_list = utils.matrix_to_list(obj_annot_mat, node_name=img_prefix+node_name, neighbor_name=obj_prefix+node_name, weight_name=weight_name)

    # Save the dataset
    if save:
        configs = utils.load_json_file(graphxr_dataset_configs_path)
        save_configs = configs["save_paths"]
        index_configs = configs["indexes"]
        obj_node_features.to_csv(save_configs["object_node_features"], index=index_configs["obj_node_features"])
        obj_co_occ_list.to_csv(save_configs["object_co_occ_list"], index=index_configs["obj_co_occ_list"])
        img_node_features.to_csv(save_configs["image_node_features"], index=index_configs["img_node_features"])
        img_co_occ_list.to_csv(save_configs["image_co_occ_list"], index=index_configs["img_co_occ_list"])
        obj_img_occ_list.to_csv(save_configs["object_image_co_occ_list"], index=index_configs["obj_img_co_occ_list"])

    return obj_co_occ_list, img_co_occ_list

annot_path = "/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/2. processed_annotations.csv"
label_categories_path = "/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/label_categories.csv"
base_url = "https://raw.githubusercontent.com/Billal-MOKHTARI/Image-Analysis-and-Object-Detection-Using-Deep-Learning/main/data/test/"
create_co_occurrence_graphxr_dataset(annot_path, label_categories_path, base_url)

# def preprocess_image_metadata(metatada_path, json_file_path, save_path=None):
#     pass