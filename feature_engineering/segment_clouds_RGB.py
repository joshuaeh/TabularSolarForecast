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
import torch.optim as optim
from torchvision.transforms import ToTensor
import glob


# declarations
## directories for reference images
SWIMSEG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "SWIMxxx", "SWIMSEG"
)
SWIMSEG_IMAGES_DIR_PATH = os.path.join(SWIMSEG_PATH, "images")
SWIMSEG_LABELS_DIR_PATH = os.path.join(SWIMSEG_PATH, "GTmaps")
SWIMSEG_LABEL_SUFFIX = "_GT.png"

HYTA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "HYTA")
HYTA_IMAGES_DIR_PATH = os.path.join(HYTA_PATH, "Images")
HYTA_LABELS_DIR_PATH = os.path.join(HYTA_PATH, "Labels")
HYTA_LABEL_SUFFIX = "_GT.jpg"

CLOUD_LABEL = [1, 1, 1]
SKY_LABEL = [0, 0, 0]

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


def display_random_image_and_label(image_paths, label_paths):
    # ensure lengths are equal
    if len(image_paths) != len(label_paths):
        raise ValueError("image_paths and label_paths are different lengths")

    random_index = np.random.choice(np.arange(len(image_paths)))

    random_image_path = image_paths[random_index]
    random_label_path = label_paths[random_index]
    print(random_image_path)
    print(random_label_path)
    return 0


class SegmentationDataSet(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.mapping = {(0, 0, 0): 0, (255, 255, 255): 1}  # sky  # clouds

    def __len__(self):
        return len(self.inputs)

    def mask_to_class(self, mask):
        mask = torch.from_numpy(np.array(mask))
        mask.squeeze_()

        class_mask = mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]

        mask_out = torch.ampty(h, w, dtype=torch.long)

        for k in self.makking:
            idx = class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(
                1
            ).unsqueeze(2)
            validx = idx.sum(0) == 3
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

        return mask_out

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)

        # get number of classes from target/mask
        obj_ids = np.unique(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y


# TODO create train_test function

## Models
class RBR_Model(nn.Module):
    """Custom PyTorch model for gradient optimization of RBR statistics"""

    def __init__(self):
        super().__init__()  # inherit from parent class
        self.RBR_threshold = torch.Tensor([0.7])

    def forward(self, img):
        print(img.shape)
        R = img[:,:,:,0]
        B = img[:,:,:,2]
        RBR = R / B
        RBR_mask = RBR > self.RBR_threshold
        return RBR_mask

## Loss Functions


# Script
## Determine device
if torch.has_cuda:
    device = torch.device('cuda')
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

## Define dataset and dataloader
train_img_paths, train_label_paths = pair_images_and_labels(
    SWIMSEG_IMAGES_DIR_PATH, SWIMSEG_LABELS_DIR_PATH, label_suffix=SWIMSEG_LABEL_SUFFIX
)
test_img_paths, test_label_paths = pair_images_and_labels(
    HYTA_IMAGES_DIR_PATH, HYTA_LABELS_DIR_PATH, label_suffix=HYTA_LABEL_SUFFIX
)

training_dataset = SegmentationDataSet(
    inputs=train_img_paths, targets=train_label_paths, transform=None
)

training_dataloader = data.DataLoader(
    dataset=training_dataset, batch_size=1, shuffle=True
)

test_dataset = SegmentationDataSet(
    inputs=test_img_paths, targets=test_label_paths, transform=None
)

## initialize model
test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

model = RBR_Model()

