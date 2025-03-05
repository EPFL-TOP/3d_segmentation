from plantseg.tasks.io_tasks import import_image_task
from pathlib import Path

import torch
print("================================== CUDAAAAA. ",torch.cuda.is_available())

# Transform image into PlantsegImage
path=Path(r"E:\PROJECTS-01\Adrian\Bachelor_project\Raw_data\t0001_ch1.tif")
semantic_type="raw" # options are 'raw', 'segmentation' and 'prediction'
stack_layout= "ZYX" # options are 'ZYX', 'YX', '2D_time'
image=import_image_task(input_path=path,semantic_type=semantic_type,stack_layout=stack_layout)
raw_image=image


from plantseg.functionals.dataprocessing.dataprocessing import image_gaussian_smoothing,scale_image_to_voxelsize
from plantseg.tasks.dataprocessing_tasks import image_cropping_task,set_voxel_size_task,gaussian_smoothing_task
import numpy as np
# Cropping, change intervals to what you need
#[77:195,437:1446,654:]
rectangle_bi= np.array([
    [77, 437, 654],  # Premier coin (z, y, x)
    [77, 1446, 654],  # Deuxième coin (z, y, x)
    [77,1446 ,2304]   # Troisième coin (z, y, x)
])
#[82:121,1171:1448,1263:1545]
rectangle_sm=np.array([
    [83, 1171, 1263],  # Premier coin (z, y, x)
    [83, 1448, 1263],  # Deuxième coin (z, y, x)
    [83,1448 ,1545]   # Troisième coin (z, y, x)
])
z_interval=(82,121)
image=image_cropping_task(image=image,rectangle=rectangle_sm,crop_z=z_interval)
rectangle_image=image
# scale image
new_voxel_size=(None,None,None)
#image=set_voxel_size_task(image,new_voxel_size)

# Normalise


# Gaussian smoothing, change sigma
sigma=2.5
image=gaussian_smoothing_task(image=image,sigma=sigma)


from plantseg.tasks.prediction_tasks import unet_prediction_task
from plantseg.tasks.segmentation_tasks import dt_watershed_task, clustering_segmentation_task, lmc_segmentation_task

# Segmentation / Prediction
model='lightsheet_3D_unet_mouse_embryo_cells' #generic_light_sheet_3D_unet or lightsheet_3D_unet_mouse_embryo_cells or zebrafish
model_id=None
device= 'cpu' # cpu or cuda
image=unet_prediction_task(image=image,model_id=model_id,model_name=model,device=device)[-1]
prediction_image=image


# Watershed task
threshold=0.5 # default 0.5
sigma_seeds=0.2 # default 1.0
stacked= False # default False 
sigma_weights=0 # default 2.0
min_size=100 # default 100
alpha= 1.25 # default 1.0
pixel_pitch= None # default None
apply_nonmax_suppression= True # default False
image=dt_watershed_task(image=image, 
                        threshold=threshold, 
                        sigma_seeds=sigma_seeds,
                        stacked=stacked, 
                        sigma_weights=sigma_weights, 
                        min_size=min_size, 
                        alpha=alpha, 
                        pixel_pitch=pixel_pitch, 
                        apply_nonmax_suppression=apply_nonmax_suppression)



# Cluster segmentation task
over_segmentation = image # over-segementation image
mode = 'gasp' # gasp, multicut or mutex_ws
beta = 1.0 # default 0.5
post_min_size = 100 # default 100
image=clustering_segmentation_task(image=rectangle_image,
                                   over_segmentation=
                                   over_segmentation,
                                   mode=mode,
                                   beta=beta,
                                   post_min_size=post_min_size)

# Lifted Multicut task
boundary_pmap=None # cell boundart prediction (Z,Y,X) normalised
superpixels=None # superpixels/over-segmentation. Must have the same shape as boundary_pmap.
nuclei=None # a nuclear segmentation or prediction map. Must have the same shape as boundary_pmap.
beta=None # default 0.5
post_min_size=None # default 100

#image=lmc_segmentation_task(boundary_pmap: PlantSegImage, superpixels: PlantSegImage, nuclei: PlantSegImage, beta: float = 0.5, post_min_size: int = 100)
