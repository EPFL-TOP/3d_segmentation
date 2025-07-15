from cellpose import core, utils, io, models, denoise, metrics, plot
from cellpose.io import imread
import torch
import tifffile as tiff
import os, shutil
import glob
import numpy as np
from pathlib import Path
import argparse
import os

# Set umask so new files are group-writable
os.umask(0o0002)

def main():
    parser = argparse.ArgumentParser(
        description="Cellpose argparse"
    )
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("output", help="Path to the input file")
    parser.add_argument("model", help="Path to the output file")
    parser.add_argument("--diameter", "-d", default = 15, help="diameter for the model", type=float)
    parser.add_argument("--anisotropy", "-a", default = 1.5, help="anisotropy", type=float)
    parser.add_argument("--minsize", "-m", default = -1., help="minimun size", type=float)
    parser.add_argument("--channels", "-c", default = [0,0], help="channels for the model", type=int, nargs='+')
    parser.add_argument("--image", "-i", default = "", help="input name of the image to match between NAS and Cluster", type=str)
    parser.add_argument("--verbose", "-v",action="store_true",help="Increase output verbosity")
    
    args = parser.parse_args()
    print(args.channels)
    
    if args.verbose:
        print("Verbose mode enabled")
    
    print("Processing file  : ", args.input)
    print("Processing model : ", args.model)

    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')
    print("CUDA device ", torch.cuda.torch.cuda.get_device_properties())
    
    # Setup logging for Cellpose
    io.logger_setup()
    
    ### MODEL
    model_path = str(args.model)
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    ### PARAMETERS
    # Define channels
    channels = [args.channels]  # cytoplasm: 1, nucleus: 2
    
    # Segmentation parameters
    diameter = args.diameter  # in pixels
    cellprob_threshold = -1
    
    # Input folder containing images
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist
    
    # Process all files in the folder
    for img_file in input_folder.glob("*.tif"):
        if args.image!="" and args.image!=img_file.name:continue
        # Read image
        img = imread(img_file)
        print(f'Loaded image {img_file.name} with shape: {img.shape} and data type {img.dtype}')
    
        # Generate output file name
        output_file_base = output_folder / img_file.stem  # Use the base name without extension
        print("========================output_file_base ",output_file_base)
        


        # computes flows from 2D slices and combines into 3D flows to create masks
        masks, flows, _ = model.eval(img,
                                     cellprob_threshold=cellprob_threshold,
                                     normalize={"tile_norm_blocksize": 128},
                                     batch_size=32,
                                     anisotropy=args.anisotropy,
                                     channel_axis=1,
                                      z_axis=0,
                                      do_3D=True, flow3D_smooth=1,
                                      diameter=diameter,
                                      min_size=args.minsize)

        output_mask_name = os.path.join(args.output, "{}".format(os.path.split(img_file)[-1].replace(".tif","_masks.tif")))
        print("output_mask_name  ",output_mask_name) 
        #io.imsave(str(output_file_base) + "_masks", masks)

        io.imsave(output_mask_name, masks)
      
        
        os.chmod(args.output, 0o2775)
        os.chmod(os.path.join(output_mask_name), 0o664)


if __name__ == "__main__":
    main()


