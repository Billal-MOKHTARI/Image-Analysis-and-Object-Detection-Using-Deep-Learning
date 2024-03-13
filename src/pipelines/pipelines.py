import sys
import os
sys.path.append(os.path.join("..", "utils"))
import utils
import pandas as pd
import numpy as np

def get_statistics(annot_mat, label_categories_path):
    # Read label_categories
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

    # Create a dataset
    obj_img_co_occ_adj_list = pd.merge(adj_co_occ_list, 
                                       annot_list.drop(columns=["Weight"]), 
                                       on="Node", 
                                       how="right")

    return obj_img_co_occ_adj_list

df = pd.read_csv("../../data/output/2. processed_annotations.csv", header=0, index_col=0)

print(get_statistics(df, "../../data/output/label_categories.csv"))

    



