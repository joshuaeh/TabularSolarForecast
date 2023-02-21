#/usr/bin/env python3
"""Script to create h5py cache of the windows for model training and testing based on """
import os
import datetime
import h5py

# Anaconda Standards
import pandas as pd
import numpy as np
# import seaborn as sns
from tqdm import tqdm

# user libraries
import constants
import utils

# get cleaned data
df = utils.load_joint_data(constants.JOINT_DATA_H5_PATH)
df = df.dropna(subset=[
    'GHI', 'DNI','DHI', 'BRBG Total Cloud Cover [%]', 'CDOC Total Cloud Cover [%]', 
    'CDOC Thick Cloud Cover [%]', 'CDOC Thin Cloud Cover [%]'])  # drop rows with NaN in any of these columns (no data from ASI-16 or solar data)
df = df.fillna(value=0)

scalar_features = constants.DATA_COLS[1:]
future_features = constants.FUTURE_FEATURES
scalar_responses = constants.SCALAR_RESPONSES
relative_responses = constants.RELATIVE_RESPONSES

n_steps_out = 12

window_cache_path = os.path.join("data", "windows_cache.h5")

for n_steps_in in [1, 2, 3, 4, 5, 6, 8, 10, 12, 13]:
    print(f"n_steps_in: {n_steps_in}")
    
    datetimes = []
    selected_past_features = []
    selected_future_features = []
    selected_scalar_responses = []
    selected_relative_responses = []
    clear_sky_indexes = []
    clear_sky_irradiances = []

    start_idx = 0
    end_idx = n_steps_in + n_steps_out
    count= 0

    for start_idx in tqdm(range(df.shape[0]-n_steps_in-n_steps_out-1)):  # TODO vectorize
        end_idx = start_idx + n_steps_in + n_steps_out
        # check that time is continuous
        if pd.to_timedelta(df.index.values[end_idx] - df.index.values[start_idx]) == (n_steps_in + n_steps_out)*datetime.timedelta(minutes=10):
            count += 1
            window_datetimes = df.index.values[start_idx:end_idx]

            # window and scale
            window_data = df.iloc[start_idx:end_idx]

            # initialize other values
            window_past_features = []
            window_future_features = []
            window_scalar_responses = []
            window_relative_responses = []
            window_clear_sky_indexes = []
            window_clear_sky_irradiances = []

            for feature in scalar_features:
                window_past_features.append(
                    window_data[feature].values[:n_steps_in]
                )
            for feature in future_features:  # future feature
                window_future_features.append(
                    window_data[feature.replace("Future ", "")].values[-n_steps_out:]
                )
        
            for response in scalar_responses:
                window_scalar_responses.append(
                    window_data[response].values[-n_steps_out:]
                )
            for response in relative_responses:
                window_relative_responses.append(
                    window_data[response.replace("Delta ", "")].values[-n_steps_out:] - window_data[response.replace("Delta ", "")].values[-n_steps_out-1]
                )

            window_clear_sky_indexes = window_data["CSI GHI"].values
            window_clear_sky_irradiances = window_data["clearsky ghi"].values

            # append but first change dimensions from n_features, n_steps to n_steps, n_features
            window_past_features = list(map(list, zip(*window_past_features)))
            window_future_features = list(map(list, zip(*window_future_features)))
            window_scalar_responses = list(map(list, zip(*window_scalar_responses)))
            window_relative_responses = list(map(list, zip(*window_relative_responses)))

            datetimes.append(window_datetimes)
            selected_past_features.append(window_past_features)
            selected_future_features.append(window_future_features)
            selected_scalar_responses.append(window_scalar_responses)
            selected_relative_responses.append(window_relative_responses)
            clear_sky_indexes.append(window_clear_sky_indexes)
            clear_sky_irradiances.append(window_clear_sky_irradiances)

    # convert lists to numpy arrays
    datetimes = np.array(datetimes).squeeze()
    selected_past_features = np.array(selected_past_features).squeeze()
    selected_future_features = np.array(selected_future_features).squeeze()
    selected_scalar_responses = np.array(selected_scalar_responses).squeeze()
    selected_relative_responses = np.array(selected_relative_responses).squeeze()
    clear_sky_indexes = np.array(clear_sky_indexes).squeeze()
    clear_sky_irradiances = np.array(clear_sky_irradiances).squeeze()
    
    print(f"n_steps_in: {n_steps_in}")
    print(f"datetimes.shape: {datetimes.shape}")
    print(f"selected_past_features.shape: {selected_past_features.shape}")
    print(f"selected_future_features.shape: {selected_future_features.shape}")
    print(f"selected_scalar_responses.shape: {selected_scalar_responses.shape}")
    print(f"selected_relative_responses.shape: {selected_relative_responses.shape}")
    print(f"clear_sky_indexes.shape: {clear_sky_indexes.shape}")
    print(f"clear_sky_irradiances.shape: {clear_sky_irradiances.shape}")
    
    with h5py.File(window_cache_path, "a") as f:
        f.create_dataset(f"{n_steps_in}/datetimes", data=datetimes.astype(np.int64))  # h5py can't handle datetimes
        f.create_dataset(f"{n_steps_in}/past_features", data=selected_past_features)
        f.create_dataset(f"{n_steps_in}/future_features", data=selected_future_features)
        f.create_dataset(f"{n_steps_in}/scalar_responses", data=selected_scalar_responses)
        f.create_dataset(f"{n_steps_in}/relative_responses", data=selected_relative_responses)
        f.create_dataset(f"{n_steps_in}/clear_sky_indexes", data=clear_sky_indexes)
        f.create_dataset(f"{n_steps_in}/clear_sky_irradiances", data=clear_sky_irradiances)
    