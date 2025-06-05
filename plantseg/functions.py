import os
import re
import ast
from itertools import product
from pathlib import Path
from tqdm import tqdm 
import numpy as np
import pandas as pd
from tifffile import imwrite

base_dir = "/mnt/e/PROJECTS-01/Adrian/Bachelor_project/GridSearch/Mouse_model"

'''
Architecture of watershed folder name

_Threshold_Value_SigmaSeed_value_SigmaWeight_value_MinSize_value_Alpha_value_PixelPitch_value
     1    _  2  _     3   _   4 _      5    _  6  _    7  _  8  _   9 _ 10  _    11    _ 12  

Architecture of Cluster folder name     
_Mode_value_Beta_value_MinSize2_value
   1 _  2  _  3 _  4  _    5   _  6

Architecture of PostProcessed folder name  
_Treshold2_value_Instances_value
     1   _  2  _    3    _  4

Possible combi 
_Threshold_Value_SigmaSeed_value_SigmaWeight_value_MinSize_value_Alpha_value_PixelPitch_value_Mode_value_Beta_value_MinSize2_value_Treshold2_value_Instances_value
     1    _  2  _     3   _   4 _      5    _  6  _    7  _  8  _   9 _ 10  _    11    _ 12  _ 13 _ 14  _ 15 _  16 _   17   _  18 _    19   _  20 _   21    _  22 

Number of parameters 22                              

Float and int should be written as 0.0
tuples should be written as -a,b,c- i.e (1.0,9.3,3.2) --> -1.0,9.3,3.2-   
'''
def pause():
    """
    Pause the program until the user presses the <ENTER> key.
    """
    programPause = input("Press the <ENTER> key to continue...")

def convert_value(value):
    """Convert a string value to its appropriate type (None, float, boolean, tuple, or string).
    Args:
        value (str): The string value to convert.
    Returns:
        The converted value, which can be None, float, boolean, tuple, or string.
    """
    if value == "None" or pd.isna(value):  # Handle None
        return None
    elif value =="False": #Hanfle Booleans
        return False
    elif value =="True":
        return True
    elif value.replace(".", "", 1).isdigit():  # Handle floats
        return float(value)
    elif value.startswith("-") and value.endswith("-"):  # Handle tuples
        try:
            elements=value[1:-1]
            str_elements="("+elements+")"
            return ast.literal_eval(str_elements)  # Convert to tuple safely
        except :
            print("Could not convert the tuple in "+value)
    else:
        return value  # Keep as string
    
def assemble_combis(dicts):
    """Assemble a list of dictionaries into a single dictionary.
    Args:
        dicts (list): A list of dictionaries to combine.
    Returns:
        A single dictionary containing all key-value pairs from the input dictionaries.
    """
    return {key:value for d in dicts for key,value in d.items()}

def create_dict(folder,in_folders):
    """Create a dictionary from the folder name and its subfolders.
    Args:
        folder (str): The name of the folder.
        in_folders (list): A list of subfolders within the folder.
    Returns:
        A list of dictionaries containing the folder values and the folder name --> the combinations.
    """
    # How to analyse folder name
    pattern = re.compile(r"_([^_]+)_([^_]+)")  
    matches = pattern.findall(folder)
    # Convert matches to a dictionary with values
    folder_values={key: convert_value(value) for key, value in matches}
    if in_folders : #check if the folder is not empty
        folders_values=[]
        for folder in in_folders:
            dicts=[folder_values,folder]
            folders_values.append(assemble_combis(dicts))
        return folders_values
    else: 
        return [folder_values]
    

def my_combinations_(path=base_dir):
    """Get all combinations from the folders in the given path.
    Args:
        path (str): The path to the base directory containing the folders.
    Returns:
        A list of dictionaries containing the combinations from all folders.
    """
    folders_data = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            #Look the combinations done in the folder
            in_folders=my_combinations_(path=folder_path)
            #Create the overall combinations with the last step
            folders_values=create_dict(folder,in_folders)
            #append the list of combinations
            folders_data+=folders_values
    return folders_data

def filter_combinations_(new_combinations,done_combinations):
    """Filter out combinations that are already done.
    Args:
        new_combinations (list): A list of new combinations to filter.
        done_combinations (list): A list of already done combinations.
    Returns:
        A list of new combinations that are not in the done combinations.
    """
    filtered_combinations=[combo for combo in new_combinations if combo not in done_combinations ]
    return filtered_combinations


def new_combinations_(Threshold,SigmaSeed,SigmaWeight,MinSize,Alpha,PixelPitch,Mode,Beta,MinSize2,Treshold2,Instances):
    """Generate all combinations of the given parameters and filter them based on existing combinations.
    Args:
        Threshold (list): List of threshold values.
        SigmaSeed (list): List of sigma seed values.
        SigmaWeight (list): List of sigma weight values.
        MinSize (list): List of minimum size values.
        Alpha (list): List of alpha values.
        PixelPitch (list): List of pixel pitch values.
        Mode (list): List of mode values.
        Beta (list): List of beta values.
        MinSize2 (list): List of second minimum size values.
        Treshold2 (list): List of second threshold values.
        Instances (list): List indicating whether instances are used or not.
    Returns:
        A list of filtered combinations that are not already done.
    """
    all_combinations = product(Threshold,
                               SigmaSeed,
                               SigmaWeight,
                               MinSize,
                               Alpha,
                               PixelPitch,
                               Mode,
                               Beta,
                               MinSize2,
                               Treshold2,
                               Instances)
    combination_list=[{
    'Threshold': t, 
    'SigmaSeed': ss, 
    'SigmaWeight': sw, 
    'MinSize': ms, 
    'Alpha': a, 
    'PixelPitch': pp, 
    'Mode': m, 
    'Beta': b, 
    'MinSize2': ms2,
    'Treshold2': t2,
    'Instances': i} for t,ss,sw,ms,a,pp,m,b,ms2,t2,i in all_combinations]
    print("Total number of asked combinations is : "+str(len(combination_list)))
    filtered_combinations = filter_combinations_(combination_list,my_combinations_())
    print("The Number of filtered combinations is : "+str(len(filtered_combinations)))
    return filtered_combinations


def divide_combi(combi):
    """Divide a combination dictionary into three parts: watershed, cluster, and post-processed.
    Args:
        combi (dict): A dictionary containing the combination parameters.
    Returns:
        A list containing three dictionaries: watershed_combi, cluster_combi, and post_processed_combi.
    """
    items=list(combi.items())
    watershed_combi={key:value for key, value in items[:7]}
    cluster_combi={key:value for key, value in items[7:9]}
    post_processed_combi={key:value for key, value in items[9:]}
    return [watershed_combi,cluster_combi,post_processed_combi]


def dict_str(combination):
    """Convert a combination dictionary to a string representation.
    Args:
        combination (dict): A dictionary containing the combination parameters.
    Returns:
        A dictionary with string representations of the values in the combination.
    """
    def format_value(value):
        if value is None:
            return "None"
        elif value is True:
            return "True"
        elif value is False:
            return "False"
        elif isinstance(value, tuple):
            # Convert tuple elements to strings and join them without spaces
            return "-" + ",".join(map(str, value)) + "-"
        elif isinstance(value,int):
            return str(float(value))
        else:
            return str(value)
    return  { key: format_value(value) for key, value in combination.items() } # convert everything to string 

def from_dict_to_folder_path(combination,path=base_dir):
    """Convert a combination dictionary to a folder path string.
    Args:
        combination (dict): A dictionary containing the combination parameters.
        path (str): The base directory path.
    Returns:
        A string representing the folder path based on the combination parameters.
    """
    combi_path="_"+"_".join(f"{key}_{value}" for key, value in combination.items()) #from dictionary to string
    return os.path.join(path, combi_path)

def get_image_path(path,condition):
    """Get the path of an image file in the given directory that meets a specific condition.
    Args:
        path (str): The directory path to search for the image file.
        condition (function): A function that takes a file name and returns True if the file meets the condition.
    Returns:
        The path of the first image file that meets the condition, or None if no such file is found.
    """
    for file_name in os.listdir(path):
        if condition(file_name):
            file_path=os.path.join(path,file_name)
            return file_path
    return None 

def get_image(path,semantic_type='raw'):
    """Get an image from the specified path based on the semantic type.
    Args:
        path (str): The directory path to search for the image file.
        semantic_type (str): The type of semantic image to retrieve (default is 'raw').
    Returns:
        An image object if found, or None if no image file meets the condition.
    """
    from plantseg.tasks.io_tasks import import_image_task
    condition=lambda file_name:not file_name.startswith('WRONG') and (file_name.endswith('.tiff') or file_name.endswith('.tif'))
    path=get_image_path(path,condition)
    if path is not None:
        return import_image_task(input_path=Path(path),semantic_type=semantic_type,stack_layout='ZYX')

def image_exists(combination, path=base_dir):
    """Check if images exist for the given combination of parameters in the specified path.
    Args:
        combination (dict): A dictionary containing the combination parameters.
        path (str): The base directory path.
    Returns:
        A list of boolean values indicating whether images exist for each step in the combination.
    """
    formatted_params = dict_str(combination)
    bool_list = []
    for combi in divide_combi(formatted_params):    
        folder_path = from_dict_to_folder_path(combi,path )
        there_is_image=False
        # Check if the folder exists
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.tiff') or file_name.endswith('.tif'):
                    there_is_image=True
        bool_list.append(there_is_image)
        path = folder_path
    return bool_list


def save_segs(path,image,name='{file_name}_export'):
    """Save a segmentation image to the specified path with the given name.
    Args:
        path (str): The directory path where the image will be saved.
        image (numpy.ndarray or plansteg Image): The segmentation image to save.
        name (str): The name pattern for the saved image file (default is '{file_name}_export').
    """
    os.makedirs(path, exist_ok=True)
    if name.startswith('WRONG'):
        # Create an empty TIFF file (e.g., a 1x1 black pixel)
        empty_image = np.zeros((1, 1), dtype=np.uint16)  # Adjust size if needed
        imwrite(os.path.join(path, f"{name}.tiff"), empty_image)
    else:
        from plantseg.tasks.io_tasks import export_image_task 
        export_image_task(image=image,
                    export_directory=Path(path),
                    name_pattern=name,
                    key=None,
                    scale_to_origin=True,
                    export_format='tiff',
                    data_type='uint16')

def remove_spaces(path=base_dir):
    """Recursively rename directories in the specified path by removing spaces from their names.
    Args:
        path (str): The base directory path where directories will be renamed.
    """
    for root, dirs, files in os.walk(path, topdown=False):  # bottom-up so renaming inner dirs first
        for name in dirs:
            if " " in name:
                old_path = os.path.join(root, name)
                new_name = name.replace(" ", "")
                new_path = os.path.join(root, new_name)
                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)


