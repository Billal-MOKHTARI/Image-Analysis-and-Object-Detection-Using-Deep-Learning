import sys
import os
# Importing user-defined module
sys.path.append(os.path.join("..", "utils"))
import utils
import pandas as pd

def create_object_co_occurrence_graphxr_dataset(annot_mat, label_categories_path):
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

    # Create adjacency co-occurrence matrix and list
    adj_co_occ_matrix = utils.calculate_co_occurrence_matrix(annot_mat)
    adj_co_occ_list = utils.matrix_to_list(adj_co_occ_matrix)

    # Create Node Weight Column
    node_weight_df = adj_co_occ_list[adj_co_occ_list["Node"]==adj_co_occ_list["Neighbor"]]
    node_weight_df.drop(columns=["Neighbor"], inplace=True)
    node_weight_df.rename(columns={"Weight": "nodeWeight"}, inplace=True)

    # Add the types of objects and their occurrence frequency
    adj_co_occ_list = pd.merge(adj_co_occ_list, label_categories, on="Node", how="left")
    adj_co_occ_list = pd.merge(adj_co_occ_list, node_weight_df, on="Node", how="left")

    # Create annotation list
    annot_list = utils.matrix_to_list(annot_mat).rename(columns={"Node":"Image", "Neighbor": "Node"})

    # Create a dataset by adding the images in which each object occurs
    obj_img_co_occ_adj_list = pd.merge(adj_co_occ_list, 
                                       annot_list.drop(columns=["Weight"]), 
                                       on="Node", 
                                       how="right")

    return obj_img_co_occ_adj_list