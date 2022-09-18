########## Imports ##########

# Standard Library
import os
import datetime
# import IPython
# import IPython.display
# import pytz
import time
# import json
from itertools import combinations_with_replacement
from typing import Concatenate

# Anaconda / Colab Standards
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as md
import numpy as np
# import seaborn as sns
from tqdm.notebook import tqdm

# Machine Learning
## SKLearn
from sklearn import preprocessing
# from sklearn.metrics import mean_absolute_error
# import imageio

## Tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Concatenate
# from keras.metrics import MAE, MAPE, MSE

## PyTorch


# pip installed libraries

## dask
import dask
import dask.dataframe as dd

## PVLib
# try:
#     import pvlib
#     from pvlib import clearsky, atmosphere, solarposition
#     from pvlib.location import Location
#     from pvlib.iotools import read_tmy3
# except ImportError:
#     ! pip install pvlib
#     import pvlib
#     from pvlib import clearsky, atmosphere, solarposition
#     from pvlib.location import Location
#     from pvlib.iotools import read_tmy3

## Neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import neptune.new as neptune

# Declarations
pd.options.display.max_rows = 300
pd.options.display.max_columns = 300
pd.plotting.register_matplotlib_converters()

import multiprocessing
print(f"CPU Count:{multiprocessing.cpu_count()}")

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
  details = tf.config.experimental.get_device_details(gpu_devices[0])
  details.get('device_name', 'Unknown GPU')
  print(details)

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

########## DECLARATIONS ##########

# cwd = "drive/MyDrive/SolarProphet/Tabular Benchmark/"

NEPTUNE_PROJECT = "HorizonPSE/SP-tab"
NEPTUNE_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNTBhNmQ1Ny04MjAzLTQ2ZjUtODA2MC0yNDllNWUxOWE2ZjkifQ=="
HOME = os.environ["HOME"]
WORK = os.environ["WORK"]
DATA_PATH = os.path.join("/work", "08940", "joshuaeh", "shared", "joint_data.csv")
SOLARPROPHET_PATH = os.path.join(WORK, "projects", "SolarProphet")

class TabularTest():
    def __init__(
        self, 
        neptune_run_name, 
        scaler_type, 
        n_steps_in, 
        n_steps_out, 
        selected_responses:list, 
        neptune_log, 
        model_save_path, 
        selected_features=None,
        selected_groups=None,
        seed=42
    ):
        # Declarations/parameters
        self.csv_cols = ["dateTime",
            # Irradiance    
            'GHI', 'DNI','DHI',
            # Clouds (camera - related)
            'BRBG Total Cloud Cover [%]', 'CDOC Total Cloud Cover [%]', 'CDOC Thick Cloud Cover [%]',
            'CDOC Thin Cloud Cover [%]', 

            'HCF Value', 'Blue/Red_min', 'Blue/Red_mid', 'Blue/Red_max',
            # Clear Sky Irradiance

            'clearsky ghi', 'clearsky dni', 'clearsky dhi', 

            'cs_dev t ghi', 'cs_dev t dni', 'cs_dev t dhi',

            'CSI GHI', 'CSI DNI', 'CSI DHI',

            # Time
            'TOD', 'TOD Sin', 'TOD Cos',

            'TOY', 'TOY Sin', 'TOY Cos',

            'time from sunrise', 'cos time from sunrise', 'sin time from sunrise', 

            'time to solar noon', 'cos time to solar noon', 'sin time to solar noon',

            'time to sunset',  'cos time to sunset',  'sin time to sunset',

            'Day', 'before solar noon',

            # Sun Flag (Camera - related)
            'Flag: Sun not visible', 'Flag: Sun on clear sky', 'Flag: Parts of sun covered',
            'Flag: Sun behind clouds, bright dot visible', 'Flag: Sun outside view', 
            'Flag: No evaluation',

            # Auxilary Weather Data
            'Precipitation [mm]', 'Precipitation (Accumulated) [mm]', 'Station Pressure [mBar]', 
            'Tower Dry Bulb Temp [deg C]', 'Tower RH [%]', 'Airmass',

            'Wind NS [m/s]', 'Wind EW [m/s]',

            'Avg Wind Speed @ 22ft [m/s]', 'Avg Wind Direction @ 22ft [deg from N]', 'Peak Wind Speed @ 22ft [m/s]',

            'Snow Depth [cm]','Snow Depth Quality', 'SE Dry Bulb Temp [deg C]', 'SE RH [%]',
            'Vertical Wind Shear [1/s]', 'Sea-Level Pressure (Est) [mBar]', 'Tower Dew Point Temp [deg C]',
            'Tower Wet Bulb Temp [deg C]', 'Tower Wind Chill Temp [deg C]',


            # Solar Position

            'Zenith Angle [degrees]', 'Azimuth Angle [degrees]', 'Solar Eclipse Shading', 

            'cos zenith', 'cos normal irradiance',

            'apparent_zenith', 'zenith', 'apparent_elevation',
            'elevation', 'azimuth', 'Sun NS', 'Sun EW',  

            # Aerosol
            '315nm POM-01 Photometer [nA]', '400nm POM-01 Photometer [uA]',
            '500nm POM-01 Photometer [uA]', '675nm POM-01 Photometer [uA]',
            '870nm POM-01 Photometer [uA]', '940nm POM-01 Photometer [uA]',
            '1020nm POM-01 Photometer [uA]',  'Albedo (CM3)', 'Albedo (LI-200)',
            'Albedo Quantum (LI-190)', 'Broadband Turbidity',

            # Lagged Variables
            'ghi t-1', 'dni t-1',
            'dhi t-1', 'ghi t-2', 'dni t-2', 'dhi t-2', 'ghi t-3', 'dni t-3',
            'dhi t-3', 'ghi t-4', 'dni t-4', 'dhi t-4', 'ghi t-5', 'dni t-5',
            'dhi t-5', 'ghi t-6', 'dni t-6', 'dhi t-6', 'ghi t-7', 'dni t-7',
            'dhi t-7', 'ghi t-8', 'dni t-8', 'dhi t-8', 'ghi t-9', 'dni t-9',
            'dhi t-9',

            # Lagged Window Stats
            'cs_dev_mean t-10 ghi', 'cs_dev_mean t-10 dni',
            'cs_dev_mean t-10 dhi', 'cs_dev_median t-10 ghi',
            'cs_dev_median t-10 dni', 'cs_dev_median t-10 dhi',
            'csi_stdev t-10 ghi', 'csi_stdev t-10 dni', 'csi_stdev t-10 dhi',
            'cs_dev_mean t-60 ghi', 'cs_dev_mean t-60 dni',
            'cs_dev_mean t-60 dhi', 'cs_dev_median t-60 ghi',
            'cs_dev_median t-60 dni', 'cs_dev_median t-60 dhi',
            'csi_stdev t-60 ghi', 'csi_stdev t-60 dni', 'csi_stdev t-60 dhi',
            'dGHI', 'dDNI', 'dDHI'

            # 'Global CMP22 (vent/cor) [W/m^2]',
            # 'Direct NIP [W/m^2]', 'Direct sNIP [W/m^2]',
            # 'Diffuse CM22-1 (vent/cor) [W/m^2]',
        ]
        self.future_features = ["Future clearsky ghi"]
        self.scalar_response = ["GHI", "DNI", "DHI", 'CSI GHI', 'CSI DNI', 'CSI DHI', 
                                'cs_dev t ghi', 'cs_dev t dni', 'cs_dev t dhi']
        self.relative_response = ["Delta GHI", "Delta DNI", "Delta DHI", "Delta CSI"]

        self.feature_groups = {
            "Irradiance" : [
                "GHI"
            ],
            "Decomposed Irradiance" : [
                "DNI", "DHI"
            ],
            "Lagged 10 min GHI" : [
                'ghi t-1', 'ghi t-2', 'ghi t-3', 'ghi t-4', 'ghi t-5', 'ghi t-6', 'ghi t-7', 'ghi t-8', 'ghi t-9'
            ],
            "Lagged 10 min Decomposed Irradiance" : [
                'dni t-1',
                'dhi t-1', 'dni t-2', 'dhi t-2', 'dni t-3',
                'dhi t-3', 'dni t-4', 'dhi t-4', 'dni t-5',
                'dhi t-5', 'dni t-6', 'dhi t-6', 'dni t-7',
                'dhi t-7', 'dni t-8', 'dhi t-8', 'dni t-9',
                'dhi t-9'
            ],
            "Time of Day" : [
                'TOD'
            ],
            "Trig Time of Day" : [
                'TOD Sin', 'TOD Cos'
            ],
            "Time of Year" : [
                'TOY'
            ],
            "Trig Time of Year" : [
                'TOY Sin', 'TOY Cos'
            ],
            "Time Milestones" : [
                'time from sunrise', 'time to solar noon', 'time to sunset', 'Day', 'before solar noon'
            ],
            "Trig Time Milestones" : [
                'cos time from sunrise', 'sin time from sunrise', 'cos time to solar noon',
                'sin time to solar noon', 'cos time to sunset',  'sin time to sunset', 'Day', 'before solar noon'
            ],
            "Clear Sky" : [
                'clearsky ghi', 'clearsky dni', 'clearsky dhi', 'Solar Eclipse Shading',  'zenith', 'elevation', 'azimuth'
            ],
            "Future Clear Sky" : [
                "Future clearsky ghi"
            ],
            "Prev Hour Stats": [
                'cs_dev t ghi', 'cs_dev t dni', 'cs_dev t dhi', 'CSI GHI', 'CSI DNI', 'CSI DHI',
                'cs_dev_mean t-10 ghi', 'cs_dev_mean t-10 dni', 'cs_dev_mean t-10 dhi', 'cs_dev_median t-10 ghi',
                'cs_dev_median t-10 dni', 'cs_dev_median t-10 dhi','csi_stdev t-10 ghi', 'csi_stdev t-10 dni', 'csi_stdev t-10 dhi',
                'cs_dev_mean t-60 ghi', 'cs_dev_mean t-60 dni', 'cs_dev_mean t-60 dhi', 'cs_dev_median t-60 ghi',
                'cs_dev_median t-60 dni', 'cs_dev_median t-60 dhi', 'csi_stdev t-60 ghi', 'csi_stdev t-60 dni', 'csi_stdev t-60 dhi',
                'dGHI', 'dDNI', 'dDHI'
            ],
            "Meteorological Measurements" : [
                'Precipitation [mm]', 'Precipitation (Accumulated) [mm]', 'Station Pressure [mBar]', 'Tower Dry Bulb Temp [deg C]', 
                'Tower RH [%]', 'Airmass', 'Wind NS [m/s]', 'Wind EW [m/s]', 'Avg Wind Speed @ 22ft [m/s]', 'Avg Wind Direction @ 22ft [deg from N]', 
                'Peak Wind Speed @ 22ft [m/s]', 'Snow Depth [cm]','Snow Depth Quality', 'SE Dry Bulb Temp [deg C]', 'SE RH [%]', 
                'Vertical Wind Shear [1/s]', 'Sea-Level Pressure (Est) [mBar]', 'Tower Dew Point Temp [deg C]','Tower Wet Bulb Temp [deg C]', 
                'Tower Wind Chill Temp [deg C]', '315nm POM-01 Photometer [nA]', '400nm POM-01 Photometer [uA]', '500nm POM-01 Photometer [uA]',
                '675nm POM-01 Photometer [uA]', '870nm POM-01 Photometer [uA]', '940nm POM-01 Photometer [uA]', '1020nm POM-01 Photometer [uA]',  
                'Albedo (CM3)', 'Albedo (LI-200)', 'Albedo Quantum (LI-190)', 'Broadband Turbidity',
            ],
            "ASI-16" : [
                'BRBG Total Cloud Cover [%]', 'CDOC Total Cloud Cover [%]', 'CDOC Thick Cloud Cover [%]',
                'CDOC Thin Cloud Cover [%]', 'HCF Value', 'Blue/Red_min', 'Blue/Red_mid', 'Blue/Red_max',
                'Flag: Sun not visible', 'Flag: Sun on clear sky', 'Flag: Parts of sun covered',
                'Flag: Sun behind clouds, bright dot visible', 'Flag: Sun outside view', 'Flag: No evaluation',
            ]
        } 

        # Declarations
        if selected_groups:
            self.selected_groups = selected_groups
            self.selected_features = self.feature_groups_to_features(self.selected_groups)
        elif selected_features:
            self.selected_features = selected_features
        else:
            ValueError("Either Selected groups or selected responses should be selected, not both.")
        self.selected_responses = selected_responses
        
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

        ### Initialize neptune
        self.neptune_log = neptune_log
        if neptune_log:
            self.run = neptune.init(
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_TOKEN,
            )
        else:
            self.run = {}  

        self.run["name"] = neptune_run_name

        self.run["selected features"] = selected_features
        self.run["selected responses"] = selected_responses
        self.run["scaler type"] = scaler_type

        ## Scaler
        if scaler_type.lower() == "minmax":
            self.scaler = preprocessing.MinMaxScaler         # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
        elif scaler_type.lower() == "normalizer":
            self.scaler = preprocessing.Normalizer          # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
        elif scaler_type.lower() == "powertransformer":
            self.scaler = preprocessing.PowerTransformer     # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
        elif scaler_type.lower() == "quantiletransformer":
            self.scaler = preprocessing.QuantileTransformer  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
        elif scaler_type.lower() == "robustscaler":
            self.scaler = preprocessing.RobustScaler         # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
        elif scaler_type.lower() == "standardscaler":
            self.scaler = preprocessing.StandardScaler       # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
        else:
            ValueError("scaler_type must be one of those listed in __init__")

        self.model_save_path = model_save_path

        tf.random.set_seed(seed)
        return

    def feature_groups_to_features(self, list_of_feature_groups):
        list_of_features = []
        for feature_group in list_of_feature_groups:
            list_of_features.extend(self.feature_groups[feature_group])
        return list_of_features

    def feature_combinations(self): # TODO write feature combination script or not
        return


    def import_csv(self):
        df = dd.read_csv(DATA_PATH, usecols=self.csv_cols, parse_dates=["dateTime"]).set_index("dateTime")
        df = df.compute()
        df = df.dropna(subset=['GHI', 'DNI','DHI',
        'BRBG Total Cloud Cover [%]', 'CDOC Total Cloud Cover [%]', 'CDOC Thick Cloud Cover [%]',
        'CDOC Thin Cloud Cover [%]', ])
        df = df.fillna(value=0)

        print(f"Data from {min(df.index)} to {max(df.index)} containing {df.shape[0]} rows and {df.shape[1]} columns")
        print(f"Duplicated rows: {df.index.duplicated().sum()}")
        print(f"Rows with Na/inf: {df.isna().any(axis=1).sum()}")
        return df

    def split_df(self, df, iso_split_date="2021-09-27", verbose=False):
        """Split and scale dataframe for machine learning
        Args:
        :param df(DataFrame): Data Frame containing all features, responses for both 
            training and testing time periods. The index should be in datetime format
        :param features(list):
        :param response(list):
        :param scaler(SKLearn preprocessing object): {MinMaxScaler, Normalizer, RobustScaler, StandardScaler}

        Returns:

        """
        # Scale the selected features and response
        if verbose:
            print(f"Earliest Date: {min(df.index)}")
            print(f"Latest Date: {max(df.index)}")

        split_date = datetime.date.fromisoformat(iso_split_date)
        if verbose:
            print(f"Split date: {split_date}\n\n")

        train_filter = (df.index < pd.to_datetime(split_date, utc=True))

        df_train = df[train_filter]
        df_val = df[~train_filter]
        if verbose:
            print(f"Train Range: {df_train.index.min()} to {df_train.index.max()}")
            print(f"Train Range: {df_val.index.min()} to {df_val.index.max()}\n\n")

            print(f"Train Data Points: {df_train.shape[0]}")
            print(f"Test Data Points: {df_val.shape[0]}")
        return df_train, df_val

    def preprocess(self, train_validate_date, validate_test_date, verbose):
        # import
        df_total = self.import_csv()

        # split
        self.df_train, df_validate_test = self.split_df(df_total, iso_split_date=train_validate_date, verbose=False)  # "2020-09-27"
        self.df_validate, self.df_test = self.split_df(df_validate_test, iso_split_date=validate_test_date, verbose=False)  # "2021-09-27"
        if verbose:
            for i in [self.df_train, self.df_validate, self.df_test]:
                print(f"Beginning {i.index.values[0]} through {i.index.values[-1]}: {i.shape[0]} points")

        # scale

        # Create Dictionary to store scaling parameters and scaled df
        self.scalers = {}

        # Get Training scales
        for column_name in self.df_train.columns.values:
            column_scaler = self.scaler()
            self.scalers[column_name] = column_scaler.fit(self.df_train[column_name].values.reshape(-1,1))

        self.response_scaler = self.scalers[self.selected_responses[0]]  # TODO change to allow for multiple responses

        return

    def transform_(self, scales, sequence_name, sequence):
        if sequence_name.startswith("Future "):  # future feature: rescale with same scale
            sequence_name = sequence_name.replace("Future ", "")
        return scales[sequence_name].transform(sequence)

    def inverse_transform_(self, scales, sequence_name, sequence, n_steps_in):
        if sequence_name.startswith("Future "):  # future feature: rescale with same scale
            sequence_name = sequence_name.replace("Future ", "")
        elif sequence_name.startswith("Delta "):
            # TODO how to rescale delta_based forecasts?
            rescaled_difference = scales[sequence_name].inverse_transform(sequence)
            t0_value = 0
            return rescaled_difference  # t0_value - rescaled_difference

        return scales[sequence_name].inverse_transform(sequence)

    def fit_relative_response(self, df, scales, relative_response_name, scaler, n_steps_out):
        response_name = relative_response_name.replace("Delta ","")
        response_data = df[response_name]
        response_deltas = np.array([])
        for i in range(n_steps_out):
            response_deltas = np.concatenate((response_deltas, response_data.diff(periods=i+1).fillna(value=0).values))
        column_scaler = self.scaler()
        scales[relative_response_name] = self.scaler.fit(response_deltas)
        return scales

    def split_past_features(self, df, feature, scalers, start_indices, n_steps_in):
        """return an array of shape (# windows, steps in, # features) """
        past_features = df[feature].values[start_indices:start_indices + n_steps_in]
        scaled_past_features = self.transform_(self.scalers, feature, past_features.reshape(-1,1))
        return scaled_past_features

    def split_scalar_response(self, df, response, start_indices, n_steps_in, n_steps_out):
        """return an array of shape (# windows, steps in, # features) 
        Also works for future features"""
        scalar_response = df[response].values[start_indices + n_steps_in:start_indices + n_steps_in + n_steps_out] 
        scaled_scalar_response = self.transform_(self.scalers, response, scalar_response.reshape(-1,1))
        return scaled_scalar_response

    def split_relative_response(self, df, response, start_indices, n_steps_in, n_steps_out):
        """
        """
        sequence = df[response].values
        response = sequence[start_indices + n_steps_in:start_indices + n_steps_in + n_steps_out] 
        relative_response = response - sequence[start_indices+n_steps_in - 1]
        scaled_relative_response = self.transform_(self.scalers, response, relative_response.reshape(-1,1))
        return scaled_relative_response

    # general windowing:
    def window_sequence(self, df, selected_features, selected_responses, scalers, n_steps_in, n_steps_out, step_time=datetime.timedelta(minutes=10)):  # TODO Reformat to allow multiprocessing
        # lists for storage
        datetimes = []
        selected_past_features = []
        selected_future_features = []
        selected_scalar_responses = []
        selected_relative_responses = []
        clear_sky_indexes = []
        clear_sky_irradiances = []

        start_idx = 0
        end_idx = n_steps_in + n_steps_out
        num_windows = 0
        count= 0

        while end_idx < df.shape[0]-1:  # TODO reformat to for loop and tqdm TODO vectorize
            # check that time is continuous
            if pd.to_timedelta(df.index.values[end_idx] - df.index.values[start_idx]) == datetime.timedelta(minutes=(n_steps_in + n_steps_out)*10):
                count += 1
                datetimes.append([df.index.values[start_idx:end_idx]])

                past_features_ = []
                future_features_ = []
                scalar_response_ = []
                relative_response_ = []
                

                for feature in selected_features:
                    if  not feature.startswith("Future "):  # past feature
                        past_features_.append(self.split_past_features(df, feature, scalers, start_idx, n_steps_in))
                    else:                               # future feature
                        future_features_.append(self.split_scalar_response(df, feature, start_idx, n_steps_in, n_steps_out))
            
                for response in selected_responses:
                    if  not response.startswith("Delta "):  # Scalar response
                        scalar_response_.append(self.split_scalar_response(df, response, start_idx, n_steps_in, n_steps_out))
                    else:
                        relative_response_.append(self.split_relative_response(df, response, start_idx, n_steps_in, n_steps_out))

                # append but first change dimensions from n_features, n_steps to n_steps, n_features
                selected_past_features.append(list(map(list, zip(*past_features_))))
                selected_future_features.append(list(map(list, zip(*future_features_))))
                # scalar_responses.append(list(map(list, zip(*scalar_response_))))
                # relative_responses.append(list(map(list, zip(*relative_response_))))
                selected_scalar_responses.append(scalar_response_)
                selected_relative_responses.append(relative_response_)
                clear_sky_indexes.append(df["CSI GHI"].values[start_idx: start_idx+n_steps_in+n_steps_out])
                clear_sky_irradiances.append(df["clearsky ghi"].values[start_idx: start_idx+n_steps_in+n_steps_out])

            start_idx+=1
            end_idx+=1

        datetimes = np.array(datetimes).squeeze()
        selected_past_features = np.array(selected_past_features).squeeze()
        selected_future_features = np.array(selected_future_features).squeeze()
        selected_scalar_responses = np.array(selected_scalar_responses).squeeze()
        selected_relative_responses = np.array(selected_relative_responses).squeeze()
        clear_sky_indexes = np.array(clear_sky_indexes).squeeze()
        clear_sky_irradiances = np.array(clear_sky_irradiances).squeeze()

        return [datetimes, selected_past_features, selected_future_features, selected_scalar_responses, selected_relative_responses, clear_sky_indexes, clear_sky_irradiances]

    def create_windows(self, verbose):
        self.train_dates, self.train_past_features, self.train_future_features, self.train_scalar_responses, self.train_relative_responses, self.train_clear_sky_indexes, self.train_clear_sky_irradiances = self.window_sequence(self.df_train, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)
        self.validate_dates, self.validate_past_features, self.validate_future_features, self.validate_scalar_responses, self.validate_relative_responses, self.validate_clear_sky_indexes, self.validate_clear_sky_irradiances = self.window_sequence(self.df_validate, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)
        self.test_dates, self.test_past_features, self.test_future_features, self.test_scalar_responses, self.test_relative_responses, self.test_clear_sky_indexes, self.test_clear_sky_irradiances = self.window_sequence(self.df_test, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)

        if verbose:
            print(f"Train windows: {len(self.train_dates)}")
            print(f"Validate windows: {len(self.validate_dates)}")
            print(f"Test windows: {len(self.test_dates)}")
        return

    ## Error Stats
    ### MAE
    # def MAE(self, y_true, y_predicted):
    #     n = tf.cast(len(y_true), tf.float32)
    #     print(y_true.dtype, y_predicted.dtype)
    #     return K.sum(K.abs(y_true - y_predicted)) / n

    def MAE(self, y_true, y_predicted):
        return np.absolute(np.subtract(y_true, y_predicted)).mean()

    def MAE_temporal_split(self, y_true, y_predicted):
        return [self.MAE(y_true[:,i], y_predicted[:,i]) for i in range(12)]

    ### nMAE
    def nMAE(self, y_true, y_predicted):
        n = len(y_true)
        ymax = np.amax(y_true)
        return (np.subtract(y_true, y_predicted)/ymax).mean()

    def nMAE_temporal_split(self, y_true, y_predicted):
        return [self.nMAE(y_true[:,i], y_predicted[:,i]) for i in range(12)]

    ### MSE
    def MSE(self, y_true, y_predicted):
        return np.square(np.subtract(y_true,y_predicted)).mean()

    def MSE_temporal_split(self, y_true, y_predicted):
        return [self.MSE(y_true[:,i], y_predicted[:,i]) for i in range(12)]

    ### RMSE
    def RMSE(self, y_true, y_predicted):
        mse = self.MSE(y_true, y_predicted)
        return np.sqrt(mse)

    def RMSE_temporal_split(self, y_true, y_predicted):
        return [self.RMSE(y_true[:,i], y_predicted[:,i]) for i in range(12)]

    ### nRMSE
    def nRMSE(self, y_true, y_predicted):
        return 100 / (np.amax(y_true) - np.amin(y_true)) * (self.MSE(y_true, y_predicted))**(0.5)

    def nRMSE_temporal_split(self, y_true, y_predicted):
        return [self.nRMSE_rescaled_GHI(y_true[:,i], y_predicted[:,i]) for i in range(12)]

    ### MAPE

    ### nMAP
    def nMAP(self, true, predicted):
        n = len(true)
        num = abs(true - predicted)
        den = 1/n * sum(predicted)
        return 1/n * sum(num / den) * 100

    def nMAP_temporal_split(self, y_true, y_predicted):
        return [self.nMAP(y_true[:,i], y_predicted[:,i]) for i in range(12)]

    def persistence_of_cloudiness_prediction(self, clear_sky_indexes, clear_sky_irradiances, n_steps_in, n_steps_out):
        prediction_csi = clear_sky_indexes[:, n_steps_in-1]
        return (prediction_csi * clear_sky_irradiances[:, n_steps_in:].T).T

    def build_CNN1D_LSTM(self,  
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), model_name=None):
        
        n_steps_in = self.n_steps_in
        n_steps_out = self.n_steps_out
        self.n_features = len(self.selected_features)
        self.n_responses = len(self.selected_responses)
  
        # tf.debugging.set_log_device_placement(True)
        self.run["Model Structure"] = "TimeDistributed Conv1D(32,1) > LSTM(16) > Dense(64) > Dense(32) >"

        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            inputs = tf.keras.layers.Input(shape=(self.n_steps_in, self.n_features))
            inputs_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),
                                                input_shape=(self.n_steps_in, self.n_features)) (inputs)
            x = TimeDistributed(Conv1D(32,1,padding="same", activation="relu"))(inputs_expanded)
            x = TimeDistributed(Flatten())(x)
            x = LSTM(16)(x)
            x = Dense(64, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            x = Dense(self.n_steps_out * self.n_responses, activation="relu")(x)
            x = Reshape((self.n_steps_out, self.n_responses), input_shape=(self.n_steps_out * self.n_responses,))(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=x, name=model_name)

            self.model.compile(optimizer, loss='mae'
                        )
        return

    def build_flexible_CNN1D_LSTM(self,  
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), model_name=None):
        """function for creating the model
        inputs must be in the shape (# windows in a batch, , )
        past features will have a length of n_steps_in while future features will have a length of n_steps_out """
        n_steps_in = self.n_steps_in
        n_steps_out = self.n_steps_out
        self.n_past_features = len([x for x in self.selected_features if x not in self.future_features])
        self.n_future_features = len([x for x in self.selected_features if x in self.future_features])
        self.n_responses = len(self.selected_responses)
  
        # tf.debugging.set_log_device_placement(True)
        self.run["Model Structure"] = ""

        # design model structure
        # Compile

        gpus = tf.config.list_logical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy(gpus)
            with strategy.scope():
                past_inputs = tf.keras.layers.Input(shape=(self.n_steps_in, self.n_past_features))
                past_inputs_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),
                                                    input_shape=(self.n_steps_in, self.n_past_features)) (past_inputs)

                past_convolution = TimeDistributed(Conv1D(32,self.n_steps_in,padding="same", activation="relu"))(past_inputs_expanded)  # TODO check Kernel Size TODO SeperableConv1D TODO deep convolutions TODO Feed Conv results with unchanged inputs to deep LSTM
                past_flattened = TimeDistributed(Flatten())(past_convolution)

                x = LSTM(16)(past_flattened)

                x = Dense(64, activation="relu")(x)
                x = Dense(32, activation="relu")(x)
                x = Dense(self.n_steps_out * self.n_responses, activation="relu")(x)
                x = Reshape((self.n_steps_out, self.n_responses), input_shape=(self.n_steps_out * self.n_responses,))(x)

                self.model = tf.keras.Model(inputs=past_flattened, outputs=x, name=model_name)
                self.model.compile(optimizer, loss='mae')
        else:
            past_inputs = tf.keras.layers.Input(shape=(self.n_steps_in, self.n_past_features))
            past_inputs_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),
                                                input_shape=(self.n_steps_in, self.n_past_features)) (past_inputs)

            past_convolution = TimeDistributed(Conv1D(32,self.n_steps_in,padding="same", activation="relu"))(past_inputs_expanded)  # TODO check Kernel Size TODO SeperableConv1D TODO deep convolutions TODO Feed Conv results with unchanged inputs to deep LSTM
            past_flattened = TimeDistributed(Flatten())(past_convolution)

            x = LSTM(16)(past_flattened)

            x = Dense(64, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            x = Dense(self.n_steps_out * self.n_responses, activation="relu")(x)
            x = Reshape((self.n_steps_out, self.n_responses), input_shape=(self.n_steps_out * self.n_responses,))(x)

            self.model = tf.keras.Model(inputs=past_flattened, outputs=x, name=model_name)
            self.model.compile(optimizer, loss='mae')
        return
            

    def create_and_fit(self):
        self.build_flexible_CNN1D_LSTM(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), model_name=None)
        self.model.summary()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0000001,
            patience=200,
            verbose=1,
            mode="min",
            restore_best_weights=True)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=30, min_lr=0.00001)

        callbacks_ = [early_stopping, reduce_lr]

        if self.neptune_log:
            callbacks_.append(NeptuneCallback(run=self.run))


        self.run["history"] = self.model.fit(x=self.train_past_features,
                        y=self.train_scalar_responses,
                    validation_data= (self.validate_past_features, self.validate_scalar_responses),
                    epochs = 1000,
                    batch_size=35000,
                    shuffle=True,
                    callbacks=callbacks_
                    )
        return

    def rescale_to_GHI(self, y, set_clear_sky_irradiances, set_clear_sky_indexes):
        """Take the selected response in {"GHI", 'CSI GHI', 'cs_dev t ghi', "Delta GHI", "Delta CSI"} as the output of the ML model
        and transform to units of WHI (W/m^2)"""
        # Squeeze and inverse transform
        inverse_transform_y = self.response_scaler.inverse_transform(y.squeeze())

        # Convert to GHI
        if "GHI" in self.selected_responses:  # GHI wil already be accounted for TODO refactor for multiple responses TODO check logic
            y_rescaled = inverse_transform_y
        elif "CS GHI" in self.selected_responses:
            y_rescaled = inverse_transform_y * set_clear_sky_irradiances[:,self.n_steps_in:]
        elif "cs_dev t ghi" in self.selected_responses:
            y_rescaled = inverse_transform_y + set_clear_sky_irradiances[:,self.n_steps_in:]
        elif "Delta GHI" in self.selected_responses:
            GHI_t0 = set_clear_sky_irradiances[:,self.n_steps_in] * set_clear_sky_indexes[:,self.n_steps_in]
            y_rescaled = inverse_transform_y + GHI_t0
        elif "Delta CSI" in self.selected_responses:
            y_rescaled = inverse_transform_y + set_clear_sky_indexes[:,self.n_steps_in]
        else:
            ValueError("""selected_responses must include one of {"GHI", 'CSI GHI', 'cs_dev t ghi', "Delta GHI", "Delta CSI"}""")

        return y_rescaled

    def final_error_metrics(self):
        self.y_train_predicted = self.model.predict(self.train_past_features)
        self.y_validate_predicted = self.model.predict(self.validate_past_features)
        self.y_test_predicted = self.model.predict(self.test_past_features)

        # rescale
        self.y_train_predicted_rescaled = self.rescale_to_GHI(self.y_train_predicted, self.train_clear_sky_irradiances, self.train_clear_sky_irradiances, self.train_clear_sky_indexes)
        self.y_validate_predicted_rescaled = self.rescale_to_GHI(self.y_validate_predicted, self.validate_clear_sky_irradiances, self.validate_clear_sky_irradiances, self.validate_clear_sky_indexes)
        self.y_test_predicted_rescaled = self.rescale_to_GHI(self.y_test_predicted, self.test_clear_sky_irradiances, self.test_clear_sky_irradiances, self.test_clear_sky_indexes)

        self.y_train_true_rescaled = self.rescale_to_GHI(self.train_scalar_responses, self.train_clear_sky_irradiances, self.train_clear_sky_irradiances, self.train_clear_sky_indexes)
        self.y_validate_true_rescaled = self.rescale_to_GHI(self.validate_scalar_responses, self.validate_clear_sky_irradiances, self.validate_clear_sky_irradiances, self.validate_clear_sky_indexes)
        self.y_test_true_rescaled = self.rescale_to_GHI(self.test_scalar_responses, self.test_clear_sky_irradiances, self.test_clear_sky_irradiances, self.test_clear_sky_indexes)

        # MAE
        self.run["Train MAE"] = self.MAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate MAE"] = self.MAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test MAE"] = self.MAE(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train MAE t+{i+1}0 min"] = self.MAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate MAE t+{i+1}0 min"] = self.MAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test MAE t+{i+1}0 min"] = self.MAE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # nMAE
        self.run["Train nMAE"] = self.nMAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate nMAE"] = self.nMAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test nMAE"] = self.nMAE(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train nMAE t+{i+1}0 min"] = self.nMAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate nMAE t+{i+1}0 min"] = self.nMAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test nMAE t+{i+1}0 min"] = self.nMAE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # MSE
        self.run["Train MSE"] = self.MSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate MSE"] = self.MSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test MSE"] = self.MSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train MSE t+{i+1}0 min"] = self.MSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate MSE t+{i+1}0 min"] = self.MSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test MSE t+{i+1}0 min"] = self.MSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # RSME
        self.run["Train RMSE"] = self.RMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate RMSE"] = self.RMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test RMSE"] = self.RMSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train RMSE t+{i+1}0 min"] = self.RMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate RMSE t+{i+1}0 min"] = self.RMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test RMSE t+{i+1}0 min"] = self.RMSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # nRMSE
        self.run["Train nRMSE"] = self.nRMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate nRMSE"] = self.nRMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test nRMSE"] = self.nRMSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train nRMSE t+{i+1}0 min"] = self.nRMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate nRMSE t+{i+1}0 min"] = self.nRMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test nRMSE t+{i+1}0 min"] = self.nRMSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # nMAP
        self.run["Train nMAP"] = self.nMAP(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate nMAP"] = self.nMAP(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test nMAP"] = self.nMAP(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train nMAP t+{i+1}0 min"] = self.nMAP(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate nMAP t+{i+1}0 min"] = self.nMAP(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test nMAP t+{i+1}0 min"] = self.nMAP(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # Persistence
        self.train_poc_prediction = self.persistence_of_cloudiness_prediction(self.train_clear_sky_indexes, self.train_clear_sky_irradiances, self.n_steps_in, self.n_steps_out)
        self.validate_poc_prediction = self.persistence_of_cloudiness_prediction(self.validate_clear_sky_indexes, self.validate_clear_sky_irradiances, self.n_steps_in, self.n_steps_out)
        self.test_poc_prediction = self.persistence_of_cloudiness_prediction(self.test_clear_sky_indexes, self.test_clear_sky_irradiances, self.n_steps_in, self.n_steps_out)

        ## Persistence MAE
        self.run["Train Persistence MAE"] = self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate Persistence MAE"] = self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test Persistence MAE"] = self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train Persistence MAE t+{i+1}0 min"] = self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate Persistence MAE t+{i+1}0 min"] = self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test Persistence MAE t+{i+1}0 min"] = self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## Persistence nMAE
        self.run["Train Persistence nMAE"] = self.nMAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate Persistence nMAE"] = self.nMAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test Persistence nMAE"] = self.nMAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train Persistence nMAE t+{i+1}0 min"] = self.nMAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate Persistence nMAE t+{i+1}0 min"] = self.nMAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test Persistence nMAE t+{i+1}0 min"] = self.nMAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## Persistence MSE
        self.run["Train Persistence MSE"] = self.MSE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate Persistence MSE"] = self.MSE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test Persistence MSE"] = self.MSE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train Persistence MSE t+{i+1}0 min"] = self.MSE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate Persistence MSE t+{i+1}0 min"] = self.MSE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test Persistence MSE t+{i+1}0 min"] = self.MSE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## Persistence RSME
        self.run["Train Persistence RMSE"] = self.RMSE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate Persistence RMSE"] = self.RMSE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test Persistence RMSE"] = self.RMSE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train Persistence RMSE t+{i+1}0 min"] = self.RMSE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate Persistence RMSE t+{i+1}0 min"] = self.RMSE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test Persistence RMSE t+{i+1}0 min"] = self.RMSE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## Persistence nRMSE
        self.run["Train Persistence nRMSE"] = self.nRMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate Persistence nRMSE"] = self.nRMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test Persistence nRMSE"] = self.nRMSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train Persistence nRMSE t+{i+1}0 min"] = self.nRMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate Persistence nRMSE t+{i+1}0 min"] = self.nRMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test Persistence nRMSE t+{i+1}0 min"] = self.nRMSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        ## Persistence nMAP
        self.run["Train Persistence nMAP"] = self.nMAP(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run["Validate Persistence nMAP"] = self.nMAP(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)
        self.run["Test Persistence nMAP"] = self.nMAP(self.y_test_true_rescaled, self.y_test_predicted_rescaled)

        for i in range(12):
            self.run[f"Train Persistence nMAP t+{i+1}0 min"] = self.nMAP(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"Validate Persistence nMAP t+{i+1}0 min"] = self.nMAP(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            self.run[f"Test Persistence nMAP t+{i+1}0 min"] = self.nMAP(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i])

        # skill Score
        ## FS MAE
        self.run["Train FS MAE"] = 1 - self.MAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate FS MAE"] = 1 - self.MAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test FS MAE"] = 1 - self.MAE(self.y_test_true_rescaled, self.y_test_predicted_rescaled) / self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train FS MAE t+{i+1}0 min"] = 1 - self.MAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate FS MAE t+{i+1}0 min"] = 1 - self.MAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test FS MAE t+{i+1}0 min"] = 1 - self.MAE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i]) / self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## FS nMAE
        self.run["Train FS nMAE"] = 1 - self.nMAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate FS nMAE"] = 1 - self.nMAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test FS nMAE"] = 1 - self.nMAE(self.y_test_true_rescaled, self.y_test_predicted_rescaled) / self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train FS nMAE t+{i+1}0 min"] = 1 - self.nMAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate FS nMAE t+{i+1}0 min"] = 1 - self.nMAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test FS nMAE t+{i+1}0 min"] = 1 - self.nMAE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i]) / self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## FS MSE
        self.run["Train FS MSE"] = 1 - self.MSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate FS MSE"] = 1 - self.MSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test FS MSE"] = 1 - self.MSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled) / self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train FS MSE t+{i+1}0 min"] = 1 - self.MSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate FS MSE t+{i+1}0 min"] = 1 - self.MSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test FS MSE t+{i+1}0 min"] = 1 - self.MSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i]) / self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## FS RSME
        self.run["Train FS RMSE"] = 1 - self.RMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate FS RMSE"] = 1 - self.RMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test FS RMSE"] = 1 - self.RMSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled) / self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train FS RMSE t+{i+1}0 min"] = 1 - self.RMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate FS RMSE t+{i+1}0 min"] = 1 - self.RMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test FS RMSE t+{i+1}0 min"] = 1 - self.RMSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i]) / self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## FS nRMSE
        self.run["Train FS nRMSE"] = 1 - self.nRMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate FS nRMSE"] = 1 - self.nRMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test FS nRMSE"] = 1 - self.nRMSE(self.y_test_true_rescaled, self.y_test_predicted_rescaled) / self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train FS nRMSE t+{i+1}0 min"] = 1 - self.nRMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate FS nRMSE t+{i+1}0 min"] = 1 - self.nRMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test FS nRMSE t+{i+1}0 min"] = 1 - self.nRMSE(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i]) / self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])

        ## FS nMAP
        self.run["Train FS nMAP"] = 1 - self.nMAP(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / self.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run["Validate FS nMAP"] = 1 - self.nMAP(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / self.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)
        self.run["Test FS nMAP"] = 1 - self.nMAP(self.y_test_true_rescaled, self.y_test_predicted_rescaled) / self.MAE(self.y_test_true_rescaled, self.test_poc_prediction)

        for i in range(12):
            self.run[f"Train FS nMAP t+{i+1}0 min"] = 1 - self.nMAP(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / self.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"Validate FS nMAP t+{i+1}0 min"] = 1 - self.nMAP(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / self.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            self.run[f"Test FS nMAP t+{i+1}0 min"] = 1 - self.nMAP(self.y_test_true_rescaled[:,i], self.y_test_predicted_rescaled[:,i]) / self.MAE(self.y_test_true_rescaled[:,i], self.test_poc_prediction[:,i])
        return

    def save_and_quit(self):
        if self.neptune_log:
            my_run_id = self.run["sys/id"].fetch()
        else:
            my_run_id = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        path = os.path.join(self.model_save_path, my_run_id)
        self.model.save(path)

        if self.neptune_log:
            self.run.stop()

        tf.keras.backend.clear_session()

        return
        

    def do_it_all(self, train_validate_date="2020-09-27", validate_test_date="2021-09-27", verbose=True):
        if verbose:
            print(" Starting ".center(40, "="))
            print(" Preprocessing ".center(40, "-"))
        self.preprocess(train_validate_date, validate_test_date, True)
        if verbose:
            print(" Preprocessing Complete ".center(40, "-"))
            print(" Creating Windows ".center(40, "-"))
        self.create_windows(True)
        if verbose:
            print(" Windows Complete ".center(40, "-"))
            print(" Creating and Fitting Models ".center(40, "-"))
        self.create_and_fit()
        if verbose:
            print(" Fit Complete ".center(40, "-"))
            print(" Calculating Error Metrics ".center(40, "-"))
        self.final_error_metrics()
        if verbose:
            print(" Error Metrics Complete ".center(40, "-"))
            print(" Saving ".center(40, "-"))
        self.save_and_quit()
        if verbose:
            print(" Saved ".center(40, "-"))
            print(" Complete! ".center(40, "="))
        return
    
# Scripting