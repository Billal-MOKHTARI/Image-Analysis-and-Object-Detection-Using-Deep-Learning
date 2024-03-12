import sys
import os
sys.path.append(os.path.join("..", "utils"))
import utils

def get_statistics(annot_mat):

    # Create adjacency co-occurrence matrix and list
    adj_co_occ_matrix = utils.calculate_co_occurrence_matrix(annot_mat)
    adj_co_occ_list = utils.matrix_to_list(adj_co_occ_matrix)

    # Create annotation list
    annot_list = utils.matrix_to_list(annot_mat)

    



