import sys
import os
# Get the current directory of pipelines.py
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def create_co_occurrence_graphxr_dataset(annot_mat_path, 
                                        label_categories_path, 
                                        URL_base, 
                                        save_path_overall = None, 
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

    index_overall = kwargs.get('index_overall', False)
    index_obj = kwargs.get('index_obj', False)
    index_img = kwargs.get('index_img', False)

    # Read label categories and annotation matrix
    label_categories = pd.read_csv(label_categories_path, header=0, index_col=0)
    obj_annot_mat = pd.read_csv(annot_mat_path, header=0, index_col=0)

    # Calculate co-occurrence matrix and convert to list
    obj_co_occ_matrix = utils.calculate_co_occurrence_matrix(obj_annot_mat)
    obj_co_occ_list = utils.matrix_to_list(obj_co_occ_matrix)

    # Extract node weights for object nodes
    node_weight_obj = obj_co_occ_list[obj_co_occ_list["Node"] == obj_co_occ_list["Neighbor"]]
    node_weight_obj.drop(columns=["Neighbor"], inplace=True)
    node_weight_obj.rename(columns={"Weight": "nodeWeight"}, inplace=True)

    # Merge label categories and node weights with object co-occurrence list
    obj_co_occ_list = pd.merge(obj_co_occ_list, label_categories, on="Node", how="left")
    obj_co_occ_list = pd.merge(obj_co_occ_list, node_weight_obj, on="Node", how="left")

    # Convert annotation matrix to list and rename columns
    annot_list = utils.matrix_to_list(obj_annot_mat).rename(columns={"Node": "Node image", "Neighbor": "Node"})

    # Merge object co-occurrence list with annotation list
    obj_co_occ_list = pd.merge(obj_co_occ_list, annot_list.drop(columns=["Weight"]), on="Node", how="right")


    # Calculate co-occurrence matrix for image nodes and convert to list
    img_annot_mat = obj_annot_mat.copy().T
    img_co_occ_matrix = utils.calculate_co_occurrence_matrix(img_annot_mat)
    img_co_occ_list = utils.matrix_to_list(img_co_occ_matrix)

    # Extract node weights for image nodes
    node_weight_img = img_co_occ_list[img_co_occ_list["Node"] == img_co_occ_list["Neighbor"]]
    node_weight_img.drop(columns=["Neighbor"], inplace=True)
    node_weight_img.rename(columns={"Weight": "nodeWeight"}, inplace=True)

    # Merge node weights with image co-occurrence list
    img_co_occ_list = pd.merge(img_co_occ_list, node_weight_img, on="Node", how="left")



    # Merge object and image co-occurrence lists
    co_occ_list = pd.merge(obj_co_occ_list, img_co_occ_list, how="left", left_on="Node image", right_on="Node",
                           suffixes=("_object", "_image"))
    del co_occ_list["Node image"]

    co_occ_list["Image URL"] = URL_base + co_occ_list["Node_image"]
    

    if save_path_overall is not None:
        co_occ_list.to_csv(save_path_overall, index=index_overall)
    
    if save_path_obj is not None:
        obj_co_occ_list.to_csv(save_path_obj, index=index_obj)

    if save_path_img is not None:
        img_co_occ_list.to_csv(save_path_img, index=index_img)

    return co_occ_list, {obj_co_occ_list, img_co_occ_list}

