"""
Evaluate how different algorithms evaluate segment clouds
"""

# imports
import os
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from skimage.io import imread

import torch
from torch import nn
from torch.functional import F
from torch.utils import data
import glob



# declarations
## directories for reference images
SWIMSEG_PATH = os.path.join(os.path.dirname(__file__), "..","data","SWIMxxx","SWIMSEG")
SWIMSEG_IMAGES_DIR_PATH = os.path.join(SWIMSEG_PATH, "images")
SWIMSEG_LABELS_DIR_PATH = os.path.join(SWIMSEG_PATH, "GTmaps")
SWIMSEG_LABEL_SUFFIX = ".png"

HYTA_PATH= os.path.join(os.path.dirname(__file__), "..","data","HYTA")
HYTA_IMAGES_DIR_PATH = os.path.join(HYTA_PATH, "GTmaps")
HYTA_LABELS_DIR_PATH = os.path.join(HYTA_PATH, "Labels")
HYTA_LABEL_SUFFIX = "_GT.png"

# Functions
## Create PyTorch Dataset
### Get paths to all files in directory
def get_all_paths_in_directory(directory_path):
    return [data_path for data_path in glob.glob(os.path.join(directory_path, "*"))]

### Pair the sky-image segment with the corresponding semantic labels
def pair_images_and_labels(images_dir, labels_dir, label_suffix=".png"):
    """
    Assumes that the image and the corresponding label have the same name, though not neccesarily the same extension
    i.e. image_name = abc123.jpg and label = abc123 + label_suffix
    """


    image_paths = get_all_paths_in_directory(images_dir)
    label_paths = []

    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        label_filename = image_filename.split(".")[0] + label_suffix
        label_paths.append(os.path.join(labels_dir, label_filename))

    return image_paths, label_paths

# Script
img_paths, label_paths = pair_images_and_labels(SWIMSEG_IMAGES_DIR_PATH, SWIMSEG_LABELS_DIR_PATH)

print(img_paths, label_paths)