# README

## Overview
This repository contains scripts and notebooks for 3D segmentation, model assessment, and grid search. Below is a guide to using the key files: `3d_plantseg.ipynb`, `asses_models.ipynb`, and `grid_search.py`.

---

## File Descriptions

### 1. `3d_plantseg.ipynb`
This notebook is used for performing 3D segmentation tasks using the PlantSeg framework for 1 image.

#### Usage:
1. Ensure the PlantSeg repository is downloaded and the required dependencies are installed.
2. Open the notebook in Jupyter or Visual Studio Code.
3. Update the paths to your raw data.
4. Run the cells to preprocess the data, train the model, and perform segmentation.
5. The output will include segmented images saved in the specified directory.

#### Key Notes:
- The notebook might rely on GPU acceleration. Be carfull for memory errors !
- Adjust patch size and anisotropy settings based on your dataset.

---

### 2. `asses_models.ipynb`
This notebook evaluates the performance of trained models by comparing predictions against ground truth annotations.

#### Usage:
1. Ensure the Stardist repository is downloaded and the required dependencies are installed.
2. Open the notebook in Jupyter or Visual Studio Code.
3. Update the paths to the ground truth and prediction files.
4. Run the cells to calculate metrics such as accuracy and mean matched score.
5. Results will be displayed in the notebook and saved to a specified output directory.

#### Key Notes:
- Ensure the ground truth and prediction files are in `.tiff` format.
- The notebook uses the StarDist library for matching metrics. Refer to [StarDist documentation](https://github.com/stardist/stardist/blob/main/stardist/matching.py) for details.

---

### 3. `grid_search.py`
This script performs a grid search to optimize segmentation parameters.

#### Usage:
1. Update the parameter lists in the script (e.g., `Threshold`, `SigmaSeed`, `SigmaWeight`, etc.).
2. Run the script on a Terminal

#### Key Notes:
- Ensure the PlantSeg repository is downloaded and the required dependencies are installed.