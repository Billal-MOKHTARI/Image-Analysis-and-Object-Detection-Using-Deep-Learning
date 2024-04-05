import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from fractions import Fraction
from utils import utils, scrapper
import pandas as pd

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

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

def use_absolute_path(path, use_abs_path):
    if use_abs_path:
        return os.path.abspath(path)
    return path

def create_co_occurrence_graphxr_dataset(graphxr_dataset_configs_path):
    configs = utils.load_json_file(graphxr_dataset_configs_path)
    use_abs_path = configs["use_absolute_path"]


    data_saver = configs["data_saver"]
    save_data = data_saver["save_data"]

    metadata_extractor = configs["metadata_extractor"]
    metadata_preprocess_arguments = metadata_extractor["metadata_preprocess_arguments"]
    extractor_script = use_absolute_path(metadata_extractor["extractor_script"], use_abs_path)
    metadata_image_folder = use_absolute_path(metadata_extractor["image_folder"], use_abs_path)
    tmp_metadata_path = use_absolute_path(metadata_extractor["tmp_path"], use_abs_path)

    graphxr_data_configs = configs["graphxr_data_configs"]
    obj_prefix = graphxr_data_configs["object_prefix"]
    img_prefix = graphxr_data_configs["image_prefix"]

    annot_mat_path = use_absolute_path(graphxr_data_configs["annotation_matrix_path"], use_abs_path)
    label_categories_path = use_absolute_path(graphxr_data_configs["label_categories_path"], use_abs_path)
    URL_base = graphxr_data_configs["image_base_URL"]
    node_name = graphxr_data_configs["node_column_name"]
    neighbor_name = graphxr_data_configs["neighbor_column_name"]
    weight_name = graphxr_data_configs["weight_column_name"]
    # Read metadata
    scrapper.get_exif_data(metadata_image_folder, tmp_metadata_path, extractor_script)
    metadata = pd.read_csv(tmp_metadata_path, header=0, index_col=0)
    os.remove(tmp_metadata_path)

    # Preprocess metadata
    metadata = utils.preprocess_image_metadata(metadata, metadata_preprocess_arguments)

    # Read label categories and annotation matrix
    label_categories = pd.read_csv(os.path.abspath(label_categories_path), header=0, index_col=0)
    obj_annot_mat = pd.read_csv(os.path.abspath(annot_mat_path), header=0, index_col=0)

    
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

    obj_node_features = add_prefix_suffix(obj_node_features, prefix=obj_prefix)
    obj_co_occ_list = add_prefix_suffix(obj_co_occ_list, prefix=obj_prefix)

    img_node_features = add_prefix_suffix(img_node_features, prefix=img_prefix)
    img_node_features = pd.merge(img_node_features, metadata, left_on = 'image_Node', right_on='FileName', how='left')


    img_co_occ_list = add_prefix_suffix(img_co_occ_list, prefix=img_prefix)
    obj_img_occ_list = utils.matrix_to_list(obj_annot_mat, node_name=img_prefix+node_name, neighbor_name=obj_prefix+node_name, weight_name=weight_name)

    # Save the dataset
    if save_data:
        save_configs = data_saver["save_paths"]
        temporary_folder = use_absolute_path(data_saver["temporary_folder"], use_abs_path)
        create_folder(temporary_folder)

        for key, value in save_configs.items():
            save_configs[key]["path"] = use_absolute_path(value["path"], use_abs_path)

        name = "obj_node_features"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        obj_node_features.to_csv(path, index=save_configs[name]["index"])

        name = "obj_co_occ_list"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        obj_co_occ_list.to_csv(path, index=save_configs[name]["index"])

        name = "img_node_features"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        img_node_features.to_csv(path, index=save_configs[name]["index"])

        name = "img_co_occ_list"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        img_co_occ_list.to_csv(path, index=save_configs[name]["index"])

        name = "obj_img_occ_list"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        obj_img_occ_list.to_csv(path, index=save_configs[name]["index"])

        name = "obj_co_occ_matrix"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        obj_co_occ_matrix.to_csv(path, index=save_configs[name]["index"])

        name = "img_co_occ_matrix"
        make_it_tmp = save_configs[name]["make_it_temporary"]
        print(make_it_tmp)
        print(os.path.join(temporary_folder, os.path.basename(original_path)))
        original_path = save_configs[name]["path"]
        path = original_path if not make_it_tmp else os.path.join(temporary_folder, os.path.basename(original_path))
        img_co_occ_matrix.to_csv(path, index=save_configs[name]["index"])



    return obj_co_occ_list, img_co_occ_list

create_co_occurrence_graphxr_dataset('/home/bimokhtari1/Documents/Image-Analysis-and-Object-Detection-Using-Deep-Learning/configs/graphxr_input_configs.json')