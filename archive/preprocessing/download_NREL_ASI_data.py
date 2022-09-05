"""
Download zip files and organize into project. Source url for ASI-16 : https://midcdmz.nrel.gov/tsi/SRRLASI/YYYY/YYYYMMDD.zip
Starting September 26, 2017
"""

# Imports
import sys, os, shutil
import shutil
import wget
from datetime import date, timedelta
import cv2
import numpy as np
import ssl
import zipfile

# Configuration and global variables

image_data_dir = os.path.join(os.path.dirname(__file__), '../data/NREL')

preprocess_image_type = ["_11.jpg", "_12.jpg"]

file_type_mappings = {"_11.jpg": "_Raw_NE.jpg",
                       "_11_NE.jpg": "_Projection_NE.jpg",
                       "_12.jpg": "_Raw_UE.jpg",
                       "_12_UE.jpg": "_Projection_UE.jpg",
                       "_1112_BRBG.png": "_Segmented_BRBG.png",
                       "_1112_CDOC.png": "_Segmented_CDOC.png",
                       ".txt": "_Image_Data.txt"}

asi_years = [
    2017, 2018, 2019, 
    2020, 2021, 2022]

# Classes and Functions

def make_dir(dir):
    """
    Make new directory if it doesn't exist
    """
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

def rename_file(file):
    """
    Rename file using file type mappings
    """
    for key, value in file_type_mappings.items():
        if file.endswith(key) and not file.endswith(value):
            os.rename(file, file.replace(key, value))

def ensure_rename_file():
    """
    Recheck and ensure if all the files are renamed proparly after the download (and rename) is finished
    """
    dates = sorted([dir for dir in os.listdir(image_data_dir) if os.path.isdir(os.path.join(image_data_dir, dir))]) # list of all image directories
    for date in dates:
        files = os.listdir(os.path.join(image_data_dir, date)) # all files in the image directory folder
        for file in files:
            filepath = os.path.join(image_data_dir, date, file)
            rename_file(filepath)

def preprocess_image(file):
    """
    Image preprocessing Tasks
    - Mask timestamp
    - Mirror image
    """
    if any([file.endswith(image_type) for image_type in preprocess_image_type]):
        # Read image
        image = cv2.imread(file)
        # Masking timestamp
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        cv2.rectangle(mask, (0, 0), (550, 30), 0, -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        # Mirror image
        image = cv2.flip(image, 1)
        # Write back
        cv2.imwrite(file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

def download_and_extract_data(base_url, date):
    """
    Download the zip file for specific date
    Extract zip file and preprocess images 
    """

    # make target directory
    target_dir = os.path.join(image_data_dir, date)
    make_dir(target_dir)

    # Extract images from the zip file
    full_url = base_url + '/' + date + '.zip'
    zip_file = wget.download(full_url)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(zip_file)

    # preprocess images and rename files
    files = os.listdir(target_dir)
    for file in files:
        file = os.path.join(target_dir, file)
        preprocess_image(file)
        rename_file(file)

def get_dates(year, month):
    """
    Get all dates in a given month and year
    """
    if month == 12:
        num_days = (date(year+1, 1, 1) - date(year, month, 1)).days
    else:
        num_days = (date(year, month+1, 1) - date(year, month, 1)).days
    d1 = date(year, month, 1)
    d2 = date(year, month, num_days)
    delta = d2 - d1
    return [(d1 + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]


def download_data(year, month_start=1, month_end=12):
    """
    Download the data for a given month range of an year (default = full year)
    """
    make_dir("../data/")  # make data directory if it doesn't exist
    make_dir(image_data_dir) # make /data/NREL folder if it doesn't exist
    base_url = 'https://midcdmz.nrel.gov/tsi/SRRLASI/' + str(year)
    for month in range(month_start, month_end+1):
        dates = get_dates(year, month)
        for date in dates:
            try:
                download_and_extract_data(base_url, date)
            except:
                print("\nError in processing: " + date)
    
# Execution script

if __name__ == "__main__":
    # Download ASI Data: Uncomment based on your need
    [download_data(i) for i in asi_years]
    ensure_rename_file()
