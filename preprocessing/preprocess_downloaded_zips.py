"""
This script will:
- sub-folder by year
- Unzip all zipped files
- rename all images
- take an accounting of covered times
- mask images give a cute smiley face when complete
"""

##################### imports #######################
import os
import zipfile
import cv2
import glob

#################### Declarations ####################

NREL_TSI_FILENAMES = {}
NREL_ASI_FILENAMES = {}


#################### functions #####################

def unzip_and_group_by_year():
    NREL_DATA_Directory = os.path.join(os.path.dirname(__file__), "..","data","NREL")
    img_dirs = ["ASI", "TSI"]
    weather_dirs = ["BMS", "IRRSP"]

    # unzip images and delete zip file
    for dir in img_dirs:
        dir_path = os.path.join(img_dirs, dir)
        zip_names = glob.glob(os.path.join(dir_path, "*.zip"))

    # add csv files to database and erase .csv (?)
    ## Need a list of all csv parameters and dates covered. perhaps one giant csv for now

def unzip_sort_images(zip_path):
    return

def assemble_csvs():
    return

def account_data():
    return

def mask_TSI_images():
    return

def mask_ASI_images():
    return



# script