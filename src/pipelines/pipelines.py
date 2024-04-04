import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def create_co_occurrence_graphxr_dataset(annot_mat_path, 
                                        label_categories_path, 
                                        URL_base, 
                                        save_path_obj = None, 
                                        save_path_img = None, 
                                        **kwargs):
    """
    Create a co-occurrence graph dataset for GraphXR.

    Parameters:
        annot_mat_path (str): Path to the annotation matrix file.
        label_categories_path (str): Path to the label categories file.

    Returns:
        pd.DataFrame: DataFrame containing the co-occurrence graph dataset.
    """

    index_obj = kwargs.get('index_obj', False)
    index_img = kwargs.get('index_img', False)

    # Read label categories and annotation matrix
    label_categories = pd.read_csv(label_categories_path, header=0, index_col=0)
    obj_annot_mat = pd.read_csv(annot_mat_path, header=0, index_col=0)

    
    # Extract stuff and thing categories so that we can identify the type of relationship between the objects
    stuff = label_categories[label_categories["Type"] == "stuff"]["Node"].tolist()
    thing = label_categories[label_categories["Type"] == "thing"]["Node"].tolist()

    # Calculate co-occurrence matrix and convert to list
    obj_co_occ_matrix = utils.calculate_co_occurrence_matrix(obj_annot_mat)
    obj_to_obj_co_occ_list = utils.matrix_to_list(obj_co_occ_matrix)

    # Identify TT relationships
    tt_mask = obj_to_obj_co_occ_list['Node'].isin(thing) & obj_to_obj_co_occ_list['Neighbor'].isin(thing)
    obj_to_obj_co_occ_list.loc[tt_mask, 'LinkType'] = 'TT'

    # Identify SS relationships
    ss_mask = obj_to_obj_co_occ_list['Node'].isin(stuff) & obj_to_obj_co_occ_list['Neighbor'].isin(stuff)
    obj_to_obj_co_occ_list.loc[ss_mask, 'LinkType'] = 'SS'

    # Identify ST relationships (rest of the cases)
    obj_to_obj_co_occ_list.loc[(~tt_mask) & (~ss_mask), 'LinkType'] = 'ST'

    # Extract node weights for object nodes
    obj_node_features = utils.extract_diagonal(obj_co_occ_matrix, ["Node", "NodeWeight"])
    obj_node_features = pd.merge(label_categories, obj_node_features, on="Node", how="right")

    # Calculate co-occurrence matrix for image nodes and convert to list
    img_annot_mat = obj_annot_mat.copy().T
    img_co_occ_matrix = utils.calculate_co_occurrence_matrix(img_annot_mat)
    img_co_occ_list = utils.matrix_to_list(img_co_occ_matrix)

    # Extract node weights for image nodes
    img_node_features = utils.extract_diagonal(img_co_occ_matrix, ["Node", "NodeWeight"])
    img_node_features["URL"] = URL_base + img_node_features["Node"]



    if save_path_obj is not None:
        obj_to_obj_co_occ_list.to_csv(save_path_obj, index=index_obj)

    if save_path_img is not None:
        img_co_occ_list.to_csv(save_path_img, index=index_img)

    return obj_to_obj_co_occ_list, img_co_occ_list

annot_path = "/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/1. annotations.csv"
label_categories_path = "/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/label_categories.csv"
base_url = "https://raw.githubusercontent.com/Billal-MOKHTARI/Image-Analysis-and-Object-Detection-Using-Deep-Learning/main/data/test/"
create_co_occurrence_graphxr_dataset(annot_path, label_categories_path, base_url)

# def preprocess_image_metadata(metatada_path, json_file_path, save_path=None):
#     pass