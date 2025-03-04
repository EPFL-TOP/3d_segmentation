from cellpose import core, utils, io, models, denoise, metrics, plot
from cellpose.io import imread

import tifffile as tiff
import os, shutil
import glob
import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Cellpose argparse"
    )
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("model", help="Path to the input model")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Increase output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode enabled")
    
    print("Processing file  : ", args.input)
    print("Processing model : ", args.model)

    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')
    
    # Setup logging for Cellpose
    io.logger_setup()
    
    ### MODEL
    model_path = str(args.model)
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    ### PARAMETERS
    # Define channels
    channels = [[0, 0]]  # cytoplasm: 1, nucleus: 2
    
    # Segmentation parameters
    diameter = 15  # in pixels
    cellprob_threshold = -1
    
    # Input folder containing images
    input_folder = Path(args.input)
    output_folder = input_folder / "denoise_customnuclei_dia15_masks_rcp"
    output_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist
    
    # Process all files in the folder
    for img_file in input_folder.glob("*.tif"):
    
        # Read image
        img = imread(img_file)
        print(f'Loaded image {img_file.name} with shape: {img.shape} and data type {img.dtype}')
    
        # Generate output file name
        output_file_base = output_folder / img_file.stem  # Use the base name without extension
    
        # USE cellposeDenoiseModel FOR DENOISING THE IMAGE
        from cellpose import denoise
        dn = denoise.DenoiseModel (model_type='denoise_cyto3', 
                                   gpu=True,
                                   chan2=False)
        img = dn.eval(img, channels=channels, diameter=diameter)
    
        # Perform segmentation
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            cellprob_threshold=cellprob_threshold,
            do_3D=True,
            anisotropy=1.5,
            min_size=-1,
        )
    
        # Save output masks to tiffs/pngs or txt files for ImageJ
        io.save_masks(
            img,
            masks,
            flows,
            str(output_file_base),
            channels=channels,
            png=False,  # Save masks as PNGs
            tif=True,  # Save masks as TIFFs
            save_txt=True,  # Save txt outlines for ImageJ
            save_flows=False,  # Save flows as TIFFs
            save_outlines=False,  # Save outlines as TIFFs
            save_mpl=False,  # Make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
        )



if __name__ == "__main__":
    main()
