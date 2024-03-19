import sys
import os
# Importing user-defined module
sys.path.append(os.path.join("..", "utils"))
import utils
import visualization
import pandas as pd
import matplotlib.pyplot as plt

def create_object_co_occurrence_graphxr_dataset(annot_mat_path, label_categories_path):
    """
    Creates a dataset for object co-occurrence graph XR based on annotation matrix and label categories.

    Parameters:
        annot_mat (DataFrame): The annotation matrix.
        label_categories_path (str): Path to the label categories file.

    Returns:
        DataFrame: Object-image co-occurrence adjacency list dataset.
    """
    # Read label categories
    label_categories = pd.read_csv(label_categories_path, header=0, index_col=0)
    obj_annot_mat = pd.read_csv(annot_mat_path, header=0, index_col=0)

    # Create adjacency co-occurrence matrix and list
    obj_co_occ_matrix = utils.calculate_co_occurrence_matrix(obj_annot_mat)
    obj_co_occ_list = utils.matrix_to_list(obj_co_occ_matrix)

    # Create Node Weight column for objects in terms of images
    node_weight_obj = obj_co_occ_list[obj_co_occ_list["Node"]==obj_co_occ_list["Neighbor"]]
    node_weight_obj.drop(columns=["Neighbor"], inplace=True)
    node_weight_obj.rename(columns={"Weight": "nodeWeight"}, inplace=True)

    # Add the types of objects and their occurrence frequency
    obj_co_occ_list = pd.merge(obj_co_occ_list, label_categories, on="Node", how="left")
    obj_co_occ_list = pd.merge(obj_co_occ_list, node_weight_obj, on="Node", how="left")

    # Create annotation list
    annot_list = utils.matrix_to_list(obj_annot_mat).rename(columns={"Node":"Node image", "Neighbor": "Node"})

    # Create a dataset by adding the images in which each object occurs
    obj_co_occ_list = pd.merge(obj_co_occ_list, 
                                       annot_list.drop(columns=["Weight"]), 
                                       on="Node", 
                                       how="right")

    
    img_annot_mat = obj_annot_mat.copy().T
    img_co_occ_matrix = utils.calculate_co_occurrence_matrix(img_annot_mat)
    img_co_occ_list = utils.matrix_to_list(img_co_occ_matrix)

    # Create Node Weight column for images in terms of objects
    node_weight_img = img_co_occ_list[img_co_occ_list["Node"]==img_co_occ_list["Neighbor"]]
    node_weight_img.drop(columns=["Neighbor"], inplace=True)
    node_weight_img.rename(columns={"Weight": "nodeWeight"}, inplace=True)

    img_co_occ_list = pd.merge(img_co_occ_list, node_weight_img, on="Node", how="left")

    # Merge both object and image co-occurrence lists
    co_occ_list = pd.merge(obj_co_occ_list, 
                           img_co_occ_list, 
                           how="left", 
                           left_on="Node image", 
                           right_on="Node",
                           suffixes=("_object", "_image"))
    del co_occ_list["Node image"]
    
    pd.set_option('display.max_columns', None)
    print(co_occ_list)

    # img_co_occ


    return 

path = "/workspaces/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/2. processed_annotations.csv"
cat = "/workspaces/Image-Analysis-and-Object-Detection-Using-Deep-Learning/data/output/label_categories.csv"
print(create_object_co_occurrence_graphxr_dataset(path, cat))
# mat = pd.read_csv(path, header=0, index_col=0).T

