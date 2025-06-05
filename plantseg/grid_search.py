from functions import new_combinations_,divide_combi,image_exists,save_segs,get_image,from_dict_to_folder_path,dict_str,pause,remove_spaces,base_dir
from plantseg.tasks.segmentation_tasks import dt_watershed_task, clustering_segmentation_task, lmc_segmentation_task
from plantseg.tasks.dataprocessing_tasks import remove_false_positives_by_foreground_probability_task,set_biggest_instance_to_zero_task
from plantseg.tasks.io_tasks import import_image_task
from pathlib import Path
from tqdm import tqdm
# This script is used to run a grid search for the watershed, clustering segmentation tasks and the post processing tasks.

# You can change the parameters in the lists below to test different combinations.
Threshold=[0.4,0.5,0.8] # range 0 1
SigmaSeed=[0.1,0.2]
SigmaWeight=[0.0,0.5]
MinSize=[90.0,100.0]
Alpha=[0.7,0.8,0.9]
PixelPitch=[(1.441,1,1)]
Mode=['gasp','multicut']#,'mutex_ws']
Beta=[0.6,0.7,0.8,0.9]#,0.5,1]
MinSize2=[75.0,100.0,120.0]#,100.0]
Treshold2=[0.1,0.2]
Instances=[False,True]

# The combinations will be generated based on the parameters above.
combinations=new_combinations_(Threshold,SigmaSeed,SigmaWeight,MinSize,Alpha,PixelPitch,Mode,Beta,MinSize2,Treshold2,Instances)

# Import the prediction image for the 3D-unet model and the raw image
prediction_image=get_image(base_dir,semantic_type='prediction')
rectangle_image=import_image_task(input_path=Path(r"/mnt/e/PROJECTS-01/Adrian/Bachelor_project/Raw_data/t0001_ch1_bi1.tiff"),semantic_type="raw",stack_layout="ZYX")
count=0
for combi in combinations :
    count+=1
    print(combi)
    images=[]
    # Divide the combination into three parts: watershed, clustering and post processing
    watershed_combi,cluster_combi,post_processed_combi=divide_combi(combi)

    # Create the paths for the three parts
    watershed_combi_str=dict_str(watershed_combi)
    watershed_path=from_dict_to_folder_path(watershed_combi_str)

    cluster_combi_str=dict_str(cluster_combi)
    cluster_path=from_dict_to_folder_path(cluster_combi_str,watershed_path)

    post_processed_combi_str=dict_str(post_processed_combi)
    post_processed_path=from_dict_to_folder_path(post_processed_combi_str,cluster_path)

    # Create the directories if they do not exist to avoid making the same task multiple times
    list_bool=image_exists(combi, path=base_dir)
    print(list_bool)
    
    # Wateshed task
    if list_bool[0]:
        images.append(get_image(watershed_path,semantic_type='segmentation'))
    else:
        try :
            images.append(dt_watershed_task(image=prediction_image, 
                            threshold=combi["Threshold"], 
                            sigma_seeds=combi["SigmaSeed"],
                            stacked=False, 
                            sigma_weights=combi["SigmaWeight"], 
                            min_size=combi["MinSize"], 
                            alpha=combi["Alpha"], 
                            pixel_pitch=combi["PixelPitch"], 
                            apply_nonmax_suppression=True))
            name='watershed_export'
    
        except Exception as e:
            print(e)
            images.append(None)
            name='WRONG_watershed_export'
        save_segs(watershed_path,images[0],name)

    # Clustering task
    if list_bool[1]:
        images.append(get_image(cluster_path,semantic_type='segmentation'))
    else:
        try:
            images.append(clustering_segmentation_task(image=rectangle_image,
                                    over_segmentation=images[0],
                                    mode=combi["Mode"],
                                    beta=combi["Beta"],
                                    post_min_size=combi["MinSize2"]))
            name='cluster_export'
        except Exception as e:
            print(e)
            images.append(None)
            name='WRONG_cluster_export'
        save_segs(cluster_path,images[1],name)

    # Post processing task
    if list_bool[2]:
        print("if this is true, there is something wrong")
        images.append(get_image(post_processed_path,semantic_type='segmentation'))
        
    else:
        try: 
            image_post_processed=remove_false_positives_by_foreground_probability_task(segmentation=images[1], 
                                                                foreground=prediction_image, 
                                                                threshold=combi["Treshold2"])
            
            images.append(set_biggest_instance_to_zero_task(image=image_post_processed,
                                                                instance_could_be_zero=combi["Instances"]))
            name='post_processed_export'
        except Exception as e:
            print(e)
            images.append(None)
            name='WRONG_post_processed_export'
        save_segs(post_processed_path,images[2],name)
    print(count,"/",len(combinations))

# Reformat the names of the folders to remove spaces
remove_spaces()
