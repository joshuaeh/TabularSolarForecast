########## Imports ##########
# Standard Library
from copy import copy, deepcopy
import os
import datetime
from joblib import Parallel, delayed
from multiprocessing import Pool
# import IPython
# import IPython.display
# import pytz
import time
import h5py
# import json

# Anaconda / Colab Standards
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as md
import numpy as np
# import seaborn as sns
from tqdm import tqdm

# Machine Learning
## SKLearn
from sklearn import preprocessing
# from sklearn.metrics import mean_absolute_error
# import imageio

## Tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Conv1D, Reshape, Dropout, Concatenate
import dask
import dask.dataframe as dd
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import neptune.new as neptune

# user libraries
import constants
import utils

# Declarations
pd.options.display.max_rows = 300
pd.options.display.max_columns = 300
pd.plotting.register_matplotlib_converters()

# TODO use dataclasses to initialize parameters and handle default values (https://docs.python.org/3/library/dataclasses.html) (Python 3.7+)
class TabularTest():
    """
    
    """
    def __init__(
        self,

        # Required Parameters
        n_steps_in:int,                   # number of timesteps used in input
        n_steps_out:int,                  # number of timesteps used in output prediction
        selected_features:list=None,      # Features to be used as inputs (NOTE: either selected_features or selected_groups should be passed, not both)
        selected_groups:list=None,        # Groups of features to be used as inputs (NOTE: either selected_features or selected_groups should be passed, not both)
        selected_responses:list=["GHI"],  # response variable to be predicted (NOTE: format required is a list but currently only one response variable is supported)
        scaler_type:str="minmax",         # scaler for raw data {minmax, normalizer, powertransformer, quantiletransformer, robustscaler, standardscaler} (default is minmax)
        data_path:str=None,               # path to measurement data. Eiter .h5 or .csv format is accepted. See data gathering script for more details on data format (str or path-like)

        ## Saving and cache parameters (optional)
        model_save_path:str=None, 
        datetimes_cache_path:str=None,  
        window_cache_path:str=None,  # path to saved windows with datetimes, features, responses, etc.
        all_past_features:list=constants.PAST_FEATURES,
        all_future_features:list=constants.FUTURE_FEATURES,
        all_scalar_responses:list=constants.SCALAR_RESPONSES,
        all_relative_responses:list=constants.RELATIVE_RESPONSES,

        # Model (optional)
        model=None,           # specified tensorflow model (default will build CNN-LSTM model) (tf model)
                              # model must have input shape: (batch_size, n_steps_in, n_features)
                              # model must have output shape: (batch_size, n_steps_out)  when multiple regressed features are used: (batch_size, n_steps_out, n_regressed_variables)
        model_name:str=None,  # name of model if desired. Only letters,numbers, and spaces

        # Optimizer and training tuning parameters (optional)
        scale_responses:bool=True,        # scale responses 
        epochs:int=200,                     # maximum number of epochs to train model (default is 200)
        shuffle_training_order:bool=False,  # shuffle training order of windows (default is False)
        batch_size:int=1000,                # batch size for training (default is 1000 windows)
        loss:str="mae",                      # loss function for model {mae or mse} (default is mae)
        optimizer=None,                     # custom tf optimizer if desired (default is Adam optimizer)
        learning_rate:float=1e-3,    # learning rate may be a static value or a tf.optimizers.schedules object
        callbacks = [],                     # list of tf.keras.callbacks to be used during training (default is empty list)
        early_stopping:bool=False,      # implement early stopping (default is False)  
        stopping_patience:int=50,       # (early stopping only) number of epochs to wait without improvement before stopping training (default is 50)
        stopping_min_delta:float=1e-7,  # (early stopping only) minimum objective function improvement to reset the patience counter (default is 1e-7)
        metrics = None,                       # list of metrics to be used during training (default is None)
        dropout_ratio = 0,
        fit_verbose:int=0,                   # verbosity of model.fit() (default is 0)

        # Utility Parameters (optional)
        data_cols:list = constants.DATA_COLS,                     # columns of raw data to be read in (default is constants.CSV_COLS)
        feature_groups:dict = constants.FEATURE_GROUPS,         # dictionary pairing of feature groups and the features the group contains (default is constants.FEATURE_GROUPS)
        scalar_response:list = constants.SCALAR_RESPONSES,        # 
        relative_response:list = constants.RELATIVE_RESPONSES,
        seed:int=42,                                            # random seed for reproducibility used in all rng-based functions (default is 42)
        n_job_workers:int=1,                                    # number of workers to use for parallel processing (default is 1)

        # Neptune Parameters (optional) 
        neptune_log:bool=False,       # log results to neptune (default is False)
        neptune_run_name:str = None,  # name of neptune run (default is None) 
        tags:list=None,                    # tags to be added to neptune run (default is None)
        ):
        """Initialize TabularTest object for irradiance prediction with tabular features"""

        # initialize object attributes
        ## Required Parameters
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        if selected_groups and (selected_features is None):  # if selected_groups is passed and selected_features is None
            self.selected_groups = selected_groups
            self.selected_features = utils.create_features_list_from_groups(selected_groups, selected_responses[0], constants.FEATURE_GROUPS,
                constants.RESPONSE_FEATURES, add_response_variable=False)
        elif selected_features and (selected_groups is None):
            self.selected_features = selected_features
        else:
            error_string = f"Either Selected groups or selected responses should be passed and both."
            error_string += f"\n\tselected_groups: {selected_groups}\n\tselected_features: {selected_features}"
            ValueError(error_string)
        self.selected_responses = selected_responses
        self.n_features = len(self.selected_features)
        self.n_responses = len(self.selected_responses)
        assert os.path.isfile(data_path), f"Data path {data_path} does not exist."
        self.data_path = data_path
        ### Scaler
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

        ## Saving and cache parameters
        self.model_save_path = model_save_path
        self.datetimes_cache_path = datetimes_cache_path
        if datetimes_cache_path:
            assert os.path.isfile(datetimes_cache_path), f"datetimes_cache_path {datetimes_cache_path} does not exist."
            self.cache_datetimes = utils.get_experiment_datetimes(n_steps_in)
            
        self.window_cache_path = window_cache_path
        self.all_past_features = all_past_features
        self.all_future_features = all_future_features
        self.all_scalar_responses = all_scalar_responses
        self.all_relative_responses = all_relative_responses

        ## Model (optional)
        self.model = model
        self.model_name = model_name

        ## Optimizer and training tuning parameters (optional)
        self.scale_responses = scale_responses
        self.epochs = epochs
        self.shuffle_training_order = shuffle_training_order
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.callbacks = callbacks
        self.early_stopping = early_stopping
        self.stopping_patience = stopping_patience
        self.stopping_min_delta = stopping_min_delta
        self.metrics = metrics
        self.dropout_ratio = dropout_ratio
        self.fit_verbose = fit_verbose
        self.metrics_dict = {
            "mae" : self.rescaled_MAE,
            "mae_t" : self.rescaled_MAE_stepwise,
        }

        ## Utility Parameters (optional)
        self.data_cols = data_cols
        self.feature_groups = feature_groups
        self.scalar_response = scalar_response
        self.relative_response = relative_response
        self.seed = seed
        self.n_job_workers = n_job_workers

        tf.random.set_seed(seed)

        ## Neptune Parameters (optional)
        self.neptune_log = neptune_log
        self.neptune_run_name = neptune_run_name
        self.tags = tags
        
        if neptune_log:
            self.run = neptune.init_run(
            project=constants.NEPTUNE_PROJECT,
            api_token=constants.NEPTUNE_TOKEN,
            tags=tags
            )
        else: 
            # TODO do neptune ANONYMOUS mode when not logging
            self.run = {}  

        self.run["name"] = self.neptune_run_name
        self.run["selected features"] = self.selected_features
        self.run["selected groups"] = selected_groups
        self.run["selected responses"] = self.selected_responses
        self.run["scaler type"] = scaler_type
        self.run["loss function"] = loss
        
        # Initialization not directly related to arguments
        self.df_total = None  # import data on initialization so it only happens once
        return None
    
    # TODO Create copy and deep copy methods
    def __copy__(self):
        return type(self)(**self.__dict__)
    
    # TODO understand memoization better
    # def __deepcopy__(self, memo):
    #     """memo is a dict of ids to copies"""
    #     id_self = id(self)        # memoization avoids unnecesary recursion
    #     _copy = memo.get(id_self)
    #     if _copy is None:
    #         _copy = type(self)(
    #             deepcopy(self.a, memo), 
    #             deepcopy(self.b, memo))
    #         memo[id_self] = _copy 
    #     return _copy
    # def __deepcopy__(self):
    #     return deepcopy(self.__dict__)

    def import_data(self, data_path:str) -> pd.DataFrame:
        """
        Import raw data from .h5 of .csv file using load_joint_data function from utils.py

        :param data_path(str or path-like): path to .h5 or .csv file containing raw data
        :(implicit) param self.data_cols(list): list of column names to import from .h5 or .csv file

        :return: pandas dataframe with datetime index
        """
        df = utils.load_joint_data(path=data_path, data_cols=self.data_cols)  # takes .h5 or csv, reads in, and returns df with datetime index
        df = df.dropna(subset=[
            'GHI', 'DNI','DHI', 'BRBG Total Cloud Cover [%]', 'CDOC Total Cloud Cover [%]', 
            'CDOC Thick Cloud Cover [%]', 'CDOC Thin Cloud Cover [%]'])  # drop rows with NaN in any of these columns (no data from ASI-16 or solar data)
        df = df.fillna(value=0)

        print("Data imported")
        print(f"Data from {min(df.index)} to {max(df.index)} containing {df.shape[0]} rows and {df.shape[1]} columns")
        print(f"Duplicated rows: {df.index.duplicated().sum()}")
        print(f"Rows with Na/inf: {df.isna().any(axis=1).sum()}")
        return df

    def split_df(self, df:pd.DataFrame, iso_split_date:str="2021-09-27", verbose:bool=False) -> tuple:
        """
        split a datetime indexed pd.DataFrame into two parts, before and after a given date

        :param df: pd.DataFrame to split
        :param iso_split_date: str, date to split df on, in ISO format (YYYY-MM-DD)
        :param verbose: bool, whether to print info about split

        :return: tuple of pd.DataFrames, (before_df, after_df)
        """
        iso_time_and_tz = " 00:00:00-07:00"  # midnight MST
        before, after = df.loc[:iso_split_date + iso_time_and_tz], df.loc[iso_split_date + iso_time_and_tz:]
        if verbose:
            print(f"Splitting df from {before.index[0]} to {before.index[-1]} and {after.index[0]} to {after.index[-1]}")
        return before, after

    def preprocess_joint_data(self, train_validate_date:str, end_date:str, verbose:bool) -> None:
        """
        import data if unimported, split to before and after a given datem and fit scalers based on the data before the given date

        :param train_validate_date: str, date to split the data into training and validation sets in ISO format (YYYY-MM-DD)
        :param end_date: str, date to end validation set spanning from train_validate_date to end_date in ISO format (YYYY-MM-DD)
        :param verbose: bool, whether to print info about split

        :return: None
        """
        # import data if unimported
        if self.df_total is None:
            self.df_total = self.import_data(self.data_path)
        
        # declare local variable
        df = self.df_total
        # split
        if end_date:
            df, _ = self.split_df(self.df_total, iso_split_date=end_date, verbose=False)
        self.df_train, self.df_validate = self.split_df(df, iso_split_date=train_validate_date, verbose=verbose)  # "2020-09-27
        if verbose:
            for i in [self.df_train, self.df_validate]:
                print(f"Beginning {i.index[0]} through {i.index[-1]}: {i.shape[0]} points")

        # scale
        ## Create Dictionary to store scaling parameters and scaled df
        self.scalers = {}

        ## Get Training scales - one scaler per feature or data column
        for column_name in self.selected_features:
            column_scaler = self.scaler()
            self.scalers[column_name] = column_scaler.fit(self.df_train[column_name].values.reshape(-1,1))

        if self.selected_responses[0] in self.df_train.columns.values:
            column_scaler = self.scaler()
            self.scalers[self.selected_responses[0]] = column_scaler.fit(self.df_train[self.selected_responses[0]].values.reshape(-1,1))
            self.response_scaler = self.scalers[self.selected_responses[0]]  # TODO change to allow for multiple responses
        else:
            # relative responses scale to the response relative to the value at t=0, and as such use a seperate function
            self.response_scaler = self.fit_relative_response(self.df_train, self.scalers, self.selected_responses[0], self.scaler, self.n_steps_out)
        return None

    def transform_(self, scales:dict, sequence_name:str, sequence:np.ndarray) -> np.ndarray:
        """
        Use a scaler selected from self.scalers to transform an array of data according to selected scaler method

        :param scales(dict): dict, dictionary of scalers to use - one for each data column
        :param sequence_name(str): str, name of the sequence to be scaled (Column name)
        :param sequence(np.ndarray): np.ndarray, 2d array of data to be scaled

        :return: np.ndarray, scaled data
        """
        # if sequence_name.startswith("Future "):  # future feature: rescale with same scale
        #     sequence_name = sequence_name.replace("Future ", "")
        return scales[sequence_name].transform(sequence)[:, 0]  # indexed to [:,0 ] due to dims [# samples, # features] but we do this feature-wise

    def inverse_transform_(self, scales:dict, sequence_name:str, sequence:np.ndarray, n_steps_in:int) -> np.ndarray:
        """
        Convert scaled data back to original units

        :param scales(dict): dict, dictionary of scalers to use - one for each data column
        :param sequence_name(str): str, name of the sequence to be scaled (Column name)
        :param sequence(np.ndarray): np.ndarray, 2d array of data to be scaled
        :param n_steps_in(int): int, number of steps in the past used as input to the model

        :return: np.ndarray, scaled data
        """
        # if sequence_name.startswith("Future "):  # future feature: rescale with same scale
        #     sequence_name = sequence_name.replace("Future ", "")
        if sequence_name.startswith("Delta "):
            # TODO how to rescale delta_based forecasts?
            rescaled_difference = scales[sequence_name].inverse_transform(sequence)
            t0_value = 0
            return rescaled_difference  # t0_value - rescaled_difference

        return scales[sequence_name].inverse_transform(sequence)[:, 0]  # indexed to [:,0 ] due to dims [# samples, # features] but we do this feature-wise

    def fit_relative_response(self, df:pd.DataFrame, scales:dict, relative_response_name:str, n_steps_out:int) -> object:
        """
        Relative responses are relative to the time of prediction. As such, the measured value must be subtracted from the time of prediction.
        This function fits the scaler for use in the relative response.

        :param df: pd.DataFrame, dataframe containing the data to be scaled
        :param scales: dict, dictionary of scalers to use - one for each data column
        :param relative_response_name: str, name of the response to be scaled
        :param n_steps_out: int, number of steps in the future to predict

        :return: scaler, scaler used to scale the data
        """
        response_name = relative_response_name.replace("Delta ","")  # Measured quantity name (Column name)
        response_data = df[response_name]  # measured quantity data
        response_deltas = np.array([])
        for i in range(n_steps_out):  # get the difference between the measured quantity and itself up to n_steps_out in the future shifted
            response_deltas = np.concatenate((response_deltas, response_data.diff(periods=i+1).fillna(value=0).values))
        column_scaler = self.scaler()  # initialize scaler object
        scales[relative_response_name] = column_scaler.fit(response_deltas.reshape(-1,1))  # fit to flattened data
        return column_scaler

    def split_past_features(self, df:pd.DataFrame, feature:str, start_indices:int, n_steps_in:int) -> np.ndarray:
        """
        return a scaled array of shape (# windows, steps in, # features)
        
        :param df(pd.DataFrame): pd.DataFrame, dataframe containing the data to be scaled)
        :param feature(str): str, name of the feature to be scaled (Column name)
        :param start_indices(int): int, index (row) of the window to be scaled
        :param n_steps_in(int): int, number of steps in the past used as input to the model
        :param (implicit) self.scalers(dict): dict, dictionary of scalers to use - one for each data column

        :return: np.ndarray, scaled data
        """
        past_features = df[feature].values[start_indices:start_indices + n_steps_in]
        scaled_past_features = self.transform_(self.scalers, feature, past_features.reshape(-1,1))
        return scaled_past_features

    def split_scalar_response(self, df:pd.DataFrame, response:str, start_indices:int, n_steps_in:int, n_steps_out:int) -> np.ndarray:
        """
        return an array of shape (# windows, steps in, # features). Also works for scalar future features
        
        :param df(pd.DataFrame): pd.DataFrame, dataframe containing the data to be scaled)
        :param response(str): str, name of the response to be scaled (Column name)
        :param start_indices(int): int, index (row) of the window to be scaled
        :param n_steps_in(int): int, number of steps in the past used as input to the model
        :param n_steps_out(int): int, number of steps in the future to predict
        :param (implicit) self.scalers(dict): dict, dictionary of scalers to use - one for each data column
        
        :return: np.ndarray, scaled data
        """
        scalar_response = df[response].values[start_indices + n_steps_in:start_indices + n_steps_in + n_steps_out]
        if self.scale_responses: 
            scaled_scalar_response = self.transform_(self.scalers, response, scalar_response.reshape(-1,1))
            return scaled_scalar_response
        else:
            return scalar_response.reshape(-1,1)

    def split_relative_response(self, df:pd.DataFrame, response_name:str, start_indices:int, n_steps_in:int, n_steps_out:int) -> np.ndarray:
        """
        calculate the relative response and return the data scaled

        :param df(pd.DataFrame): dataframe containing the data to be scaled
        :param response_name(str): name of the response to be scaled (Column name)
        :param start_indices(int): index (row) of the window to be scaled
        :param n_steps_in(int): number of steps in the past used as input to the model
        :param n_steps_out(int): number of steps in the future to predict
        :param (implicit) self.scalers(dict): dictionary of scalers to use - one for each data column

        :return: np.ndarray, scaled data
        """
        scalar_response_name = response_name.replace("Delta ","")
        sequence = df[scalar_response_name].values
        response = sequence[start_indices + n_steps_in:start_indices + n_steps_in + n_steps_out] 
        relative_response = response - sequence[start_indices+n_steps_in-1]
        if self.scale_responses:
            scaled_relative_response = self.transform_(self.scalers, response_name, relative_response.reshape(-1,1))
            return scaled_relative_response
        else:
            return relative_response.reshape(-1,1)

    def create_single_window(self, 
        window_datetimes:list, 
        df:pd.DataFrame,
        selected_features:list, 
        selected_responses:list, 
        scalers:dict, 
        n_steps_in:int, 
        n_steps_out:int) -> list:
        """
        function for use with cached datetimes for the windows already available

        :param window_datetimes(list): list, list of datetimes for the windows
        :param df(pd.DataFrame): pd.DataFrame, dataframe containing the data
        :param selected_features(list): list, list of features to be used
        :param selected_responses(list): list, list of responses to be used
        :param scalers(dict): dict, dictionary containing the scalers for the features
        :param n_steps_in(int): int, number of steps in the past
        :param n_steps_out(int): int, number of steps in the future

        :return: list, list of lists containing the windows
        [datetimes, selected_past_features, selected_future_features, selected_scalar_responses, selected_relative_responses, clear_sky_indexes, clear_sky_irradiances]
        """
        window_data = df.loc[window_datetimes]

        # initialize other values
        window_past_features = []
        window_future_features = []
        window_scalar_responses = []
        window_relative_responses = []
        window_clear_sky_indexes = []
        window_clear_sky_irradiances = []

        for feature in selected_features:
            if  not feature.startswith("Future "):  # past feature
                window_past_features.append(self.split_past_features(window_data, feature, 0, n_steps_in))
            else:                               # future feature
                window_future_features.append(self.split_scalar_response(window_data, feature, 0, n_steps_in, n_steps_out))
    
        for response in selected_responses:
            if  not response.startswith("Delta "):  # Scalar response
                window_scalar_responses.append(self.split_scalar_response(window_data, response, 0, n_steps_in, n_steps_out))
            else:
                window_relative_responses.append(self.split_relative_response(window_data, response, 0, n_steps_in, n_steps_out))

        window_clear_sky_indexes = window_data["CSI GHI"].values
        window_clear_sky_irradiances = window_data["clearsky ghi"].values

        # append but first change dimensions from n_features, n_steps to n_steps, n_features
        window_past_features = list(map(list, zip(*window_past_features)))
        window_future_features = list(map(list, zip(*window_future_features)))

        return window_datetimes, window_past_features, window_future_features, window_scalar_responses, window_relative_responses, window_clear_sky_indexes, window_clear_sky_irradiances

    # general windowing:
    def window_sequential(self, 
        df:pd.DataFrame, 
        selected_features:list[str], 
        selected_responses:list[str], 
        scalers:dict, 
        n_steps_in:int, 
        n_steps_out:int, 
        step_time:datetime.timedelta=datetime.timedelta(minutes=10)) -> list:
        """
        loop through all the data and if the time is continuous and the correct length, scale datetimes, features, responses, clear sky indices, and clear sky irradiances
        and add to lists. Then convert the lists to numpy arrays and return them.

        :param df(pd.DataFrame): pd.DataFrame, dataframe containing the data to be windowed
        :param selected_features(list[str]): list[str], list of features to be used in the model
        :param selected_responses(list[str]): list[str], list of responses to be used in the model
        :param scalers(dict): dict, dictionary of scalers to use - one for each data column
        :param n_steps_in(int): int, number of steps in the past used as input to the model
        :param n_steps_out(int): int, number of steps in the future to predict
        :param step_time(datetime.timedelta): datetime.timedelta, time between each step in the data

        :return: list(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray), tuple of numpy arrays containing the datetimes, features, responses, clear sky indices, and clear sky irradiances
        [datetimes, selected_past_features, selected_future_features, selected_scalar_responses, selected_relative_responses, clear_sky_indexes, clear_sky_irradiances]
        """
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
        count= 0

        for start_idx in tqdm(range(df.shape[0]-n_steps_in-n_steps_out-1)):  # TODO vectorize
            end_idx = start_idx + n_steps_in + n_steps_out
            # check that time is continuous
            if pd.to_timedelta(df.index.values[end_idx] - df.index.values[start_idx]) == (n_steps_in + n_steps_out)*step_time:
                count += 1
                window_datetimes = df.index[start_idx:end_idx]

                # window and scale
                window_results = self.create_single_window(
                    window_datetimes,
                    df,
                    selected_features,
                    selected_responses,
                    scalers,
                    n_steps_in,
                    n_steps_out,
                    )

                datetimes.append(window_results[0])
                selected_past_features.append(window_results[1])
                selected_future_features.append(window_results[2])
                selected_scalar_responses.append(window_results[3])
                selected_relative_responses.append(window_results[4])
                clear_sky_indexes.append(window_results[5])
                clear_sky_irradiances.append(window_results[6])

        # convert lists to numpy arrays
        datetimes = np.array(datetimes).squeeze()
        selected_past_features = np.array(selected_past_features).squeeze()
        selected_future_features = np.array(selected_future_features).squeeze()
        selected_scalar_responses = np.array(selected_scalar_responses).squeeze()
        selected_relative_responses = np.array(selected_relative_responses).squeeze()
        clear_sky_indexes = np.array(clear_sky_indexes).squeeze()
        clear_sky_irradiances = np.array(clear_sky_irradiances).squeeze()

        return [datetimes, selected_past_features, selected_future_features, selected_scalar_responses, selected_relative_responses, clear_sky_indexes, clear_sky_irradiances]

    def window_cached(self, 
        df:pd.DataFrame, 
        selected_features:list[str], 
        selected_responses:list[str], 
        scalers:dict, 
        n_steps_in:int, 
        n_steps_out:int) -> list:
        """
        loop through all the data and if the time is continuous and the correct length, scale datetimes, features, responses, clear sky indices, and clear sky irradiances
        and add to lists. Then convert the lists to numpy arrays and return them.

        :param df(pd.DataFrame): pd.DataFrame, dataframe containing the data to be windowed
        :param selected_features(list[str]): list[str], list of features to be used in the model
        :param selected_responses(list[str]): list[str], list of responses to be used in the model
        :param scalers(dict): dict, dictionary of scalers to use - one for each data column
        :param n_steps_in(int): int, number of steps in the past used as input to the model
        :param n_steps_out(int): int, number of steps in the future to predict
        :param (implicit) cache_datetimes_path(str): str, path to the cache file containing the datetimes  TODO utils does not use this path, only # steps

        :return: list(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray), tuple of numpy arrays containing the datetimes, features, responses, clear sky indices, and clear sky irradiances
        [datetimes, selected_past_features, selected_future_features, selected_scalar_responses, selected_relative_responses, clear_sky_indexes, clear_sky_irradiances]
        """
        # lists for storage
        datetimes = []
        selected_past_features = []
        selected_future_features = []
        selected_scalar_responses = []
        selected_relative_responses = []
        clear_sky_indexes = []
        clear_sky_irradiances = []

        cache_datetimes = utils.get_experiment_datetimes(n_steps_in).to_numpy()
        # compare as datetime64[ns] to avoid type issues
        after_df_start_mask = cache_datetimes[:,0] >= df.index[0]
        before_df_end_mask = cache_datetimes[:,-1] <= df.index[-1]
        cache_datetimes = cache_datetimes[after_df_start_mask & before_df_end_mask]
        n_windows, _ = cache_datetimes.shape

        for i in tqdm(range(n_windows)):  # TODO vectorize

            window_datetimes = cache_datetimes[i]

            # window and scale
            window_results = self.create_single_window(
                window_datetimes,
                df,
                selected_features,
                selected_responses,
                scalers,
                n_steps_in,
                n_steps_out,
                )

            datetimes.append(window_results[0])
            selected_past_features.append(window_results[1])
            selected_future_features.append(window_results[2])
            selected_scalar_responses.append(window_results[3])
            selected_relative_responses.append(window_results[4])
            clear_sky_indexes.append(window_results[5])
            clear_sky_irradiances.append(window_results[6])

        # convert lists to numpy arrays
        datetimes = np.array(datetimes).squeeze()
        selected_past_features = np.array(selected_past_features).squeeze()
        selected_future_features = np.array(selected_future_features).squeeze()
        selected_scalar_responses = np.array(selected_scalar_responses).squeeze()
        selected_relative_responses = np.array(selected_relative_responses).squeeze()
        clear_sky_indexes = np.array(clear_sky_indexes).squeeze()
        clear_sky_irradiances = np.array(clear_sky_irradiances).squeeze()

        return [datetimes, selected_past_features, selected_future_features, selected_scalar_responses, selected_relative_responses, clear_sky_indexes, clear_sky_irradiances]

    def create_windows(self, verbose):
        """
        Create windows for prediction either in parallel or serially based on n_workers argument
        :
        """
        if self.datetimes_cache_path is not None:
            if verbose:
                print(f"Forming windows from cache".center(40,"="))
                print(f"Training Set")
            self.train_dates, self.train_past_features, self.train_future_features, self.train_scalar_responses, self.train_relative_responses, self.train_clear_sky_indexes, self.train_clear_sky_irradiances \
                = self.window_cached(self.df_train, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)
            if verbose:
                print(f"Validation Set")
            self.validate_dates, self.validate_past_features, self.validate_future_features, self.validate_scalar_responses, self.validate_relative_responses, self.validate_clear_sky_indexes, self.validate_clear_sky_irradiances \
                = self.window_cached(self.df_validate, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)
        
        else:
            if verbose:
                print("Forming Windows Sequentially".center(40,"="))
                print(f"Training Set")
            self.train_dates, self.train_past_features, self.train_future_features, self.train_scalar_responses, self.train_relative_responses, self.train_clear_sky_indexes, self.train_clear_sky_irradiances \
                = self.window_sequential(self.df_train, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)
            if verbose:
                print(f"Validation Set")
            self.validate_dates, self.validate_past_features, self.validate_future_features, self.validate_scalar_responses, self.validate_relative_responses, self.validate_clear_sky_indexes, self.validate_clear_sky_irradiances \
                = self.window_sequential(self.df_validate, self.selected_features, self.selected_responses, self.scalers, self.n_steps_in, self.n_steps_out)

        # convenience attributes
        # pass either relative or scalar responses to model
        if self.selected_responses[0] in self.relative_response:
            print("Relative Response")
            self.y_train_true = self.train_relative_responses
            self.y_validate_true = self.validate_relative_responses
        else:
            self.y_train_true = self.train_scalar_responses
            self.y_validate_true = self.validate_scalar_responses    
            
        if verbose:
            print(f"Train windows: {len(self.train_dates)}")

            print(f"\n{self.train_dates.shape=}")
            print(f"\n{self.train_past_features.shape=}")
            print(f"\n{self.train_future_features.shape=}")
            print(f"\n{self.train_scalar_responses.shape=}")
            print(f"\n{self.train_relative_responses.shape=}")
            print(f"\n{self.train_clear_sky_indexes.shape=}")
            print(f"\n{self.train_clear_sky_irradiances.shape=}")

            print(f"Validate windows: {len(self.validate_dates)}")
        return None
    
    def import_preprocess_cached_windows(self,
        train_validate_date,
        end_date,
        verbose):
        """
        Use h5 cache with precomputed windows for the given window range. Load, split, scale, and return windows.
        
        :param window_cache_path(str): path to h5py cache
        :param train_validate_date(str): date to split train and validate sets in ISO format: YYYY-MM-DD
        :param end_date(str): end date of window range in ISO format: YYYY-MM-DD
        :param all_past_features(list): list of all possible past features. IMPORTANT: needs to match the order of the features in the cache. use constants.py
        :param all_future_features(list): list of all possible future features. See note in all_past_features
        :param all_scalar_responses(list): list of all possible scalar responses. See note in all_past_features
        :param all_relative_responses(list): list of all possible relative responses. See note in all_past_features
        :verbose(bool): print statements
        
        implicit args: self.n_steps_in, self.selected_features, self.selected_responses, self.scaler
        
        :return: None
        
        implicit returns in self. namespace: scalers, train_dates, train_past_features, train_future_features, train_scalar_responses,
            train_relative_responses, train_clear_sky_indexes, train_clear_sky_irradiances, validate_dates, validate_past_features, 
            validate_future_features, validate_scalar_responses, validate_relative_responses, validate_clear_sky_indexes, validate_clear_sky_irradiances
        """
        # read h5py cache
        if verbose:
            print(f" Reading windows from cache ".center(40,"="))
        with h5py.File(self.window_cache_path, 'r') as f:
            datetimes = f[f"{self.n_steps_in}/datetimes"][:]                          # (n_windows, n_steps_in + n_steps_out)
            past_features = f[f"{self.n_steps_in}/past_features"][:]                  # (n_windows, n_steps_in, n_scalar_features(134))
            future_features = f[f"{self.n_steps_in}/future_features"][:]              # (n_windows, n_steps_out, n_future_features(7))
            scalar_responses = f[f"{self.n_steps_in}/scalar_responses"][:]            # (n_windows, n_steps_out, n_scalar_responses(3)
            relative_responses = f[f"{self.n_steps_in}/relative_responses"][:]        # (n_windows, n_steps_out, n_relative_responses(2))
            clear_sky_indexes = f[f"{self.n_steps_in}/clear_sky_indexes"][:]          # (n_windows, n_steps_in + n_steps_out)
            clear_sky_irradiances = f[f"{self.n_steps_in}/clear_sky_irradiances"][:]  # (n_windows, n_steps_in + n_steps_out)
        if verbose:
            print("\nDone")
            
            if verbose >= 2:
                print(f" Selecting and scaling relevant features and responses ".center(40,"="))
                
                print(f"datetimes.shape: {datetimes.shape}")
                print(f"past_features.shape: {past_features.shape}")
                print(f"future_features.shape: {future_features.shape}")
                print(f"scalar_responses.shape: {scalar_responses.shape}")
                print(f"relative_responses.shape: {relative_responses.shape}")
                print(f"clear_sky_indexes.shape: {clear_sky_indexes.shape}")
                print(f"clear_sky_irradiances.shape: {clear_sky_irradiances.shape}")
            
        # convert datetimes from int64 to tz-aware datetime
        datetimes = pd.DatetimeIndex(datetimes).tz_localize("UTC").tz_convert("MST")

        # get indices of relevant datetimes
        train_validate_date = pd.to_datetime(train_validate_date + " 00:00:00").tz_localize("MST")
        end_date = pd.to_datetime(end_date + " 00:00:00").tz_localize("MST")

        train_mask = datetimes[:,-1] < train_validate_date
        validate_mask = (datetimes[:,0] >= train_validate_date) & (datetimes[:,-1] < end_date)

        # list feature and response types
        self.selected_past_features = [feature for feature in self.selected_features if feature in self.all_past_features]
        self.selected_future_features = [feature for feature in self.selected_features if feature in self.all_future_features]
        self.selected_scalar_responses = [response for response in self.selected_responses if response in self.all_scalar_responses]
        self.selected_relative_responses = [response for response in self.selected_responses if response in self.all_relative_responses]
        
        if verbose >= 2:
            print(f"# selected past features: {len(self.selected_past_features)}")
            print(f"# selected future features: {len(self.selected_future_features)}")
            print(f"# selected scalar responses: {len(self.selected_scalar_responses)}")
            print(f"# selected relative responses: {len(self.selected_relative_responses)}")

        # get indices of selected quantities
        selected_past_features_indices = [
            self.all_past_features.index(feature) for feature in self.selected_past_features
            ]
        selected_future_features_indices = [
            self.all_future_features.index(feature) for feature in self.selected_future_features
            ]
        selected_scalar_responses_indices = [
            self.all_scalar_responses.index(response) for response in self.selected_scalar_responses
            ]
        selected_relative_responses_indices = [
            self.all_relative_responses.index(response) for response in self.selected_relative_responses
            ]
        
        if verbose >= 2:
            print(f"{selected_past_features_indices=}")
            print(f"{selected_future_features_indices=}")
            print(f"{selected_scalar_responses_indices=}")
            print(f"{selected_relative_responses_indices=}")

        # new lists will include only relevant datetimes and features/responses in order that they were given
        train_datetimes = datetimes[train_mask]
        validate_datetimes = datetimes[validate_mask]
        train_past_features = past_features[np.ix_(train_mask, np.arange(self.n_steps_in), selected_past_features_indices)]
        validate_past_features = past_features[np.ix_(validate_mask, np.arange(self.n_steps_in), selected_past_features_indices)]
        train_future_features = future_features[np.ix_(train_mask, np.arange(self.n_steps_out), selected_future_features_indices)]
        validate_future_features = future_features[np.ix_(validate_mask, np.arange(self.n_steps_out), selected_future_features_indices)]
        train_scalar_responses = scalar_responses[np.ix_(train_mask, np.arange(self.n_steps_out), selected_scalar_responses_indices)]
        validate_scalar_responses = scalar_responses[np.ix_(validate_mask, np.arange(self.n_steps_out), selected_scalar_responses_indices)]
        train_relative_responses = relative_responses[np.ix_(train_mask, np.arange(self.n_steps_out), selected_relative_responses_indices)]
        validate_relative_responses = relative_responses[np.ix_(validate_mask, np.arange(self.n_steps_out), selected_relative_responses_indices)]
        train_clear_sky_indexes = clear_sky_indexes[train_mask]
        validate_clear_sky_indexes = clear_sky_indexes[validate_mask]
        train_clear_sky_irradiances = clear_sky_irradiances[train_mask]
        validate_clear_sky_irradiances = clear_sky_irradiances[validate_mask]
        
        if verbose >= 2:
            print(f"train_dates.shape: {train_datetimes.shape}")
            print(f"train_past_features.shape: {train_past_features.shape}")
            print(f"train_future_features.shape: {train_future_features.shape}")
            print(f"train_scalar_responses.shape: {train_scalar_responses.shape}")
            print(f"train_relative_responses.shape: {train_relative_responses.shape}")
            print(f"train_clear_sky_indexes.shape: {train_clear_sky_indexes.shape}")
            print(f"train_clear_sky_irradiances.shape: {train_clear_sky_irradiances.shape}")
            
            print(f"validate_dates.shape: {validate_datetimes.shape}")
            print(f"validate_past_features.shape: {validate_past_features.shape}")
            print(f"validate_future_features.shape: {validate_future_features.shape}")
            print(f"validate_scalar_responses.shape: {validate_scalar_responses.shape}")
            print(f"validate_relative_responses.shape: {validate_relative_responses.shape}")
            print(f"validate_clear_sky_indexes.shape: {validate_clear_sky_indexes.shape}")
            print(f"validate_clear_sky_irradiances.shape: {validate_clear_sky_irradiances.shape}")

        # fit scalers to training portion and scale data
        self.scalers = {}

        for i, selected_feature in enumerate(self.selected_past_features):
            feature_scaler = self.scaler()
            feature_data = train_past_features[:, :, i]
            feature_data_shape = feature_data.shape
            train_past_features[:, :, i] = feature_scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(feature_data_shape)
            validate_past_features_data = validate_past_features[:, :, i]
            validate_past_features_shape = validate_past_features_data.shape
            validate_past_features[:, :, i] = feature_scaler.transform(validate_past_features_data.reshape(-1, 1)).reshape(validate_past_features_shape)
            self.scalers[selected_feature] = feature_scaler
        for i, selected_feature in enumerate(self.selected_future_features):
            feature_scaler = self.scaler()
            feature_data = train_future_features[:, :, i]
            feature_data_shape = feature_data.shape
            train_future_features[:, :, i] = feature_scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(feature_data_shape)
            validate_future_features_data = validate_future_features[:, :, i]
            validate_future_features_shape = validate_future_features_data.shape
            validate_future_features[:, :, i] = feature_scaler.transform(validate_future_features_data.reshape(-1, 1)).reshape(validate_future_features_shape)
            self.scalers[selected_feature] = feature_scaler
        if self.scale_responses:          
            for i, selected_response in enumerate(self.selected_scalar_responses):
                response_scaler = self.scaler()
                response_data = train_scalar_responses[:, :, i]
                response_data_shape = response_data.shape
                train_scalar_responses[:, :, i] = response_scaler.fit_transform(response_data.reshape(-1, 1)).reshape(response_data_shape)
                validate_scalar_responses_data = validate_scalar_responses[:, :, i]
                validate_scalar_responses_shape = validate_scalar_responses_data.shape
                validate_scalar_responses[:, :, i] = response_scaler.transform(validate_scalar_responses_data.reshape(-1, 1)).reshape(validate_scalar_responses_shape)
                self.scalers[selected_response] = response_scaler
                self.response_scaler = response_scaler  # TODO make it so multiple responses can be scaled
            for i, selected_response in enumerate(self.selected_relative_responses):
                response_scaler = self.scaler()
                response_data = train_relative_responses[:, :, i]
                response_data_shape = response_data.shape
                train_relative_responses[:, :, i] = response_scaler.fit_transform(response_data.reshape(-1, 1)).reshape(response_data_shape)
                validate_relative_responses_data = validate_relative_responses[:, :, i]
                validate_relative_responses_shape = validate_relative_responses_data.shape
                validate_relative_responses[:, :, i] = response_scaler.transform(validate_relative_responses_data.reshape(-1, 1)).reshape(validate_relative_responses_shape)
                self.scalers[selected_response] = response_scaler
                self.response_scaler = response_scaler  # TODO make it so multiple responses can be scaled
                
        # move to self. namespace
        ## Train
        self.train_dates = train_datetimes
        self.train_past_features = train_past_features
        self.train_future_features = train_future_features
        self.train_scalar_responses = train_scalar_responses
        self.train_relative_responses = train_relative_responses
        self.train_clear_sky_indexes = train_clear_sky_indexes
        self.train_clear_sky_irradiances = train_clear_sky_irradiances
        
        ## Validate
        self.validate_dates = validate_datetimes
        self.validate_past_features = validate_past_features
        self.validate_future_features = validate_future_features
        self.validate_scalar_responses = validate_scalar_responses
        self.validate_relative_responses = validate_relative_responses
        self.validate_clear_sky_indexes = validate_clear_sky_indexes
        self.validate_clear_sky_irradiances = validate_clear_sky_irradiances
        
        ## convenience attributes
        # pass either relative or scalar responses to model
        if self.selected_responses[0] in self.relative_response:
            print("Relative Response")
            self.y_train_true = self.train_relative_responses
            self.y_validate_true = self.validate_relative_responses
        else:
            self.y_train_true = self.train_scalar_responses
            self.y_validate_true = self.validate_scalar_responses
        
        if verbose:
            print(f"Train windows: {len(self.train_dates)}")

            print(f"\n{self.train_dates.shape=}")
            print(f"\n{self.train_past_features.shape=}")
            print(f"\n{self.train_future_features.shape=}")
            print(f"\n{self.train_scalar_responses.shape=}")
            print(f"\n{self.train_relative_responses.shape=}")
            print(f"\n{self.train_clear_sky_indexes.shape=}")
            print(f"\n{self.train_clear_sky_irradiances.shape=}")

            print(f"Validate windows: {len(self.validate_dates)}")
        return None
    

    def create_default_model(self, n_steps_in, n_past_features, model_name):
        """
        Create tensorflow CNN-LSTM model.
        Inputs should be shape (n_samples, n_steps_in, n_past_features)
        Outputs should be shape (n_samples, n_steps_out, n_future_features)

        :param n_steps_in: number of past timesteps to use as input
        :param n_past_features: number of features in the past used as inputs
        :param model_name: name of the model
        :param optimizer: optimizer to use
        :param loss_func: loss function to use
        
        :return: compiled tensorflow model
        """
        layer_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2, seed=self.seed)  # truncated normal initializer

        past_inputs = tf.keras.layers.Input(shape=(n_steps_in, n_past_features), name="PastInputs")
        past_inputs = Dropout(self.dropout_ratio, input_shape=(n_steps_in, n_past_features))(past_inputs)
        past_inputs_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),
                                            input_shape=(n_steps_in, n_past_features), name="PastInputsExpanded") (past_inputs)

        past_convolution = TimeDistributed(Conv1D(32,self.n_steps_in,padding="same", activation="relu"))(past_inputs_expanded)  # TODO check Kernel Size TODO SeperableConv1D TODO deep convolutions TODO Feed Conv results with unchanged inputs to deep LSTM
        past_flattened = TimeDistributed(Flatten())(past_convolution)
        
        x = LSTM(16, kernel_initializer=layer_initializer, name="LSTM")(past_flattened)

        x = Dense(64, activation="relu", kernel_initializer=layer_initializer)(x)
        x = Dense(32, activation="relu", kernel_initializer=layer_initializer)(x)
        x = Dense(self.n_steps_out * self.n_responses, activation="relu", kernel_initializer=layer_initializer)(x)
        x = Reshape((self.n_steps_out, self.n_responses), input_shape=(self.n_steps_out * self.n_responses,))(x)

        self.model = tf.keras.Model(inputs=past_inputs, outputs=x, name=model_name)
        
        return self.model

    def compile_model(self):
        """
        
        """
        # optimizer
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
        # metrics
        if self.metrics is not None:
            metrics_intermediate = []
            for k,v in self.metrics_dict.items():
                if k in self.metrics:
                    metrics_intermediate.append(v)
            self.metrics = metrics_intermediate
        
        # create default model if none is provided
        if self.model is None:
            self.model = self.create_default_model(
                n_steps_in=self.n_steps_in,
                n_past_features=len(self.selected_features),  # TODO account for future features
                model_name=self.model_name,
            )
        
        # compile model on GPU if available TODO add kwarg
        gpus = tf.config.list_logical_devices('GPU')
        if gpus:
            with tf.device(gpus[0]):
                self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
        else:
            self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
            
        self.model.summary()
        return None
    
    def fit_model(self):
        """
        
        """
        # initialize callbacks
        if self.early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=self.stopping_min_delta,
                patience=self.stopping_patience,
                verbose=self.fit_verbose,
                mode="min",
                restore_best_weights=True)
            self.callbacks.append(early_stopping)

        if self.neptune_log:
            self.callbacks.append(NeptuneCallback(run=self.run))
        
        # 
        self.run["history"] = self.model.fit(
            x=self.train_past_features,  # TODO save Model Weights in Neptune
            y=self.y_train_true,
            validation_data= (self.validate_past_features, self.y_validate_true),
            epochs = self.epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_order,
            callbacks=self.callbacks,
            workers=self.n_job_workers,
            verbose=self.fit_verbose
        )
        return None

    def rescaled_MAE_stepwise(self, y_true_scaled, y_predicted_scaled):
        """
        :param y_true: (n_samples, n_steps_out)
        :param y_predicted: (n_samples, n_steps_out)

        :return: (n_steps_out) Mean absolute error for each step

        implicit args: self.response_scaler (sklearn object)

        TODO only suppourts GHI prediction variable currently (?)
        """
        y_true_rescaled = self.response_scaler.inverse_transform(y_true_scaled)
        y_predicted_rescaled = self.response_scaler.inverse_transform(y_predicted_scaled)
        return utils.MAE(y_true_rescaled, y_predicted_rescaled, axis=0)

    def rescaled_MAE(self, y_true_scaled, y_predicted_scaled):
        """
        :param y_true: (n_samples, n_steps_out)
        :param y_predicted: (n_samples, n_steps_out)

        :return: rescaled mean absolute error for the flattened dataset

        implicit args: self.response_scaler (sklearn object)

        TODO only suppourts GHI prediction variable currently (?)
        """
        y_true_rescaled = self.response_scaler.inverse_transform(y_true_scaled)
        y_predicted_rescaled = self.response_scaler.inverse_transform(y_predicted_scaled)
        return utils.MAE(y_true_rescaled, y_predicted_rescaled, axis=None)
    
    def rescale_to_GHI(self, y, set_clear_sky_irradiances, set_clear_sky_indexes):
        """Take the selected response in {"GHI", 'CSI GHI', 'cs_dev t ghi', "Delta GHI", "Delta CSI"} as the output of the ML model
        and transform to units of WHI (W/m^2)"""
        # Squeeze and inverse transform
        inverse_transform_y = self.response_scaler.inverse_transform(y.squeeze())

        # Convert to GHI
        if "GHI" in self.selected_responses:  # GHI wil already be accounted for TODO refactor for multiple responses TODO check logic
            y_rescaled = inverse_transform_y
        elif "CSI GHI" in self.selected_responses:
            y_rescaled = inverse_transform_y * set_clear_sky_irradiances[:,self.n_steps_in:]
        elif "cs_dev t ghi" in self.selected_responses:
            y_rescaled = inverse_transform_y + set_clear_sky_irradiances[:,self.n_steps_in:]
        elif "Delta GHI" in self.selected_responses:
            GHI_t0 = set_clear_sky_irradiances[:,self.n_steps_in] * set_clear_sky_indexes[:,self.n_steps_in]
            y_rescaled = inverse_transform_y + GHI_t0[:,None]
        elif "Delta CSI GHI" in self.selected_responses:
            y_rescaled = (inverse_transform_y + set_clear_sky_indexes[:,self.n_steps_in][:,None]) * set_clear_sky_irradiances[:,self.n_steps_in:]
        else:
            ValueError("""selected_responses must include one of {"GHI", 'CSI GHI', 'cs_dev t ghi', "Delta GHI", "Delta CSI"}""")
        return y_rescaled

    def final_error_metrics(self, cv):
        self.y_train_predicted = self.model.predict(self.train_past_features)
        self.y_validate_predicted = self.model.predict(self.validate_past_features)

        # rescale
        if self.scale_responses:
            self.y_train_predicted_rescaled = self.rescale_to_GHI(self.y_train_predicted, self.train_clear_sky_irradiances,  self.train_clear_sky_indexes)
            self.y_validate_predicted_rescaled = self.rescale_to_GHI(self.y_validate_predicted, self.validate_clear_sky_irradiances, self.validate_clear_sky_indexes)

            self.y_train_true_rescaled = self.rescale_to_GHI(self.y_train_true, self.train_clear_sky_irradiances,  self.train_clear_sky_indexes)
            self.y_validate_true_rescaled = self.rescale_to_GHI(self.y_validate_true, self.validate_clear_sky_irradiances, self.validate_clear_sky_indexes)
            
        else:  # or don't rescale
            self.y_train_predicted_rescaled = self.y_train_predicted  # [:,:,None]  # add third axis for predicted if not rescaling
            self.y_validate_predicted_rescaled = self.y_validate_predicted  # [:,:,None]
            
            if self.selected_responses[0] in self.relative_response:
                self.y_train_true_rescaled = self.train_relative_responses
                self.y_validate_true_rescaled = self.validate_relative_responses

            else:
                self.y_train_true_rescaled = self.train_scalar_responses
                self.y_validate_true_rescaled = self.validate_scalar_responses
        
        # squeeze dimensions to get arrays in the same dimensions
        self.y_train_predicted_rescaled = np.squeeze(self.y_train_predicted_rescaled)
        self.y_validate_predicted_rescaled = np.squeeze(self.y_validate_predicted_rescaled)
        
        self.y_train_true_rescaled = np.squeeze(self.y_train_true_rescaled)
        self.y_validate_true_rescaled = np.squeeze(self.y_validate_true_rescaled)
        
        if cv:
            cv_string = f"CV {cv}: "
        else:
            cv_string = ""

        # MBE
        self.run[f"{cv_string}Train MBE"] = utils.MBE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate MBE"] = utils.MBE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train MBE t+{i+1}0 min"] = utils.MBE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate MBE t+{i+1}0 min"] = utils.MBE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            
        # nMBE
        self.run[f"{cv_string}Train nMBE"] = utils.nMBE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate nMBE"] = utils.nMBE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train nMBE t+{i+1}0 min"] = utils.nMBE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate nMBE t+{i+1}0 min"] = utils.nMBE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])
            
        # MAE
        self.run[f"{cv_string}Train MAE"] = utils.MAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate MAE"] = utils.MAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train MAE t+{i+1}0 min"] = utils.MAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate MAE t+{i+1}0 min"] = utils.MAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])

        # nMAE
        self.run[f"{cv_string}Train nMAE"] = utils.nMAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate nMAE"] = utils.nMAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train nMAE t+{i+1}0 min"] = utils.nMAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate nMAE t+{i+1}0 min"] = utils.nMAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])

        # MSE
        self.run[f"{cv_string}Train MSE"] = utils.MSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate MSE"] = utils.MSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train MSE t+{i+1}0 min"] = utils.MSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate MSE t+{i+1}0 min"] = utils.MSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])

        # RSME
        self.run[f"{cv_string}Train RMSE"] = utils.RMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate RMSE"] = utils.RMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train RMSE t+{i+1}0 min"] = utils.RMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate RMSE t+{i+1}0 min"] = utils.RMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])

        # nRMSE
        self.run[f"{cv_string}Train nRMSE"] = utils.nRMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate nRMSE"] = utils.nRMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train nRMSE t+{i+1}0 min"] = utils.nRMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate nRMSE t+{i+1}0 min"] = utils.nRMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])

        # Persistence
        self.train_poc_prediction = utils.persistence_of_cloudiness_prediction(self.train_clear_sky_indexes, self.train_clear_sky_irradiances, self.n_steps_in, self.n_steps_out)
        self.validate_poc_prediction = utils.persistence_of_cloudiness_prediction(self.validate_clear_sky_indexes, self.validate_clear_sky_irradiances, self.n_steps_in, self.n_steps_out)

        ## Persistence MAE
        self.run[f"{cv_string}Train Persistence MAE"] = utils.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate Persistence MAE"] = utils.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train Persistence MAE t+{i+1}0 min"] = utils.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate Persistence MAE t+{i+1}0 min"] = utils.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])

        ## Persistence nMAE
        self.run[f"{cv_string}Train Persistence nMAE"] = utils.nMAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate Persistence nMAE"] = utils.nMAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train Persistence nMAE t+{i+1}0 min"] = utils.nMAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate Persistence nMAE t+{i+1}0 min"] = utils.nMAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])

        ## Persistence MSE
        self.run[f"{cv_string}Train Persistence MSE"] = utils.MSE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate Persistence MSE"] = utils.MSE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train Persistence MSE t+{i+1}0 min"] = utils.MSE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate Persistence MSE t+{i+1}0 min"] = utils.MSE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])

        ## Persistence RSME
        self.run[f"{cv_string}Train Persistence RMSE"] = utils.RMSE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate Persistence RMSE"] = utils.RMSE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train Persistence RMSE t+{i+1}0 min"] = utils.RMSE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate Persistence RMSE t+{i+1}0 min"] = utils.RMSE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
            
        ## Persistence nRMSE
        self.run[f"{cv_string}Train Persistence nRMSE"] = utils.nRMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled)
        self.run[f"{cv_string}Validate Persistence nRMSE"] = utils.nRMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled)

        for i in range(12):
            self.run[f"{cv_string}Train Persistence nRMSE t+{i+1}0 min"] = utils.nRMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i])
            self.run[f"{cv_string}Validate Persistence nRMSE t+{i+1}0 min"] = utils.nRMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i])

        # Skill Score 1 - prediction error / reference error
        ## FS MAE
        self.run[f"{cv_string}Train FS MAE"] = 1 - utils.MAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / utils.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate FS MAE"] = 1 - utils.MAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / utils.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train FS MAE t+{i+1}0 min"] = 1 - utils.MAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / utils.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate FS MAE t+{i+1}0 min"] = 1 - utils.MAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / utils.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])

        ## FS nMAE
        self.run[f"{cv_string}Train FS nMAE"] = 1 - utils.nMAE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / utils.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate FS nMAE"] = 1 - utils.nMAE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / utils.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train FS nMAE t+{i+1}0 min"] = 1 - utils.nMAE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / utils.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate FS nMAE t+{i+1}0 min"] = 1 - utils.nMAE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / utils.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])

        ## FS MSE
        self.run[f"{cv_string}Train FS MSE"] = 1 - utils.MSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / utils.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate FS MSE"] = 1 - utils.MSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / utils.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train FS MSE t+{i+1}0 min"] = 1 - utils.MSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / utils.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate FS MSE t+{i+1}0 min"] = 1 - utils.MSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / utils.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])

        ## FS RSME
        self.run[f"{cv_string}Train FS RMSE"] = 1 - utils.RMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / utils.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate FS RMSE"] = 1 - utils.RMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / utils.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train FS RMSE t+{i+1}0 min"] = 1 - utils.RMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / utils.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate FS RMSE t+{i+1}0 min"] = 1 - utils.RMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / utils.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
        
        ## FS nRMSE
        self.run[f"{cv_string}Train FS nRMSE"] = 1 - utils.nRMSE(self.y_train_true_rescaled, self.y_train_predicted_rescaled) / utils.MAE(self.y_train_true_rescaled, self.train_poc_prediction)
        self.run[f"{cv_string}Validate FS nRMSE"] = 1 - utils.nRMSE(self.y_validate_true_rescaled, self.y_validate_predicted_rescaled) / utils.MAE(self.y_validate_true_rescaled, self.validate_poc_prediction)

        for i in range(12):
            self.run[f"{cv_string}Train FS nRMSE t+{i+1}0 min"] = 1 - utils.nRMSE(self.y_train_true_rescaled[:,i], self.y_train_predicted_rescaled[:,i]) / utils.MAE(self.y_train_true_rescaled[:,i], self.train_poc_prediction[:,i])
            self.run[f"{cv_string}Validate FS nRMSE t+{i+1}0 min"] = 1 - utils.nRMSE(self.y_validate_true_rescaled[:,i], self.y_validate_predicted_rescaled[:,i]) / utils.MAE(self.y_validate_true_rescaled[:,i], self.validate_poc_prediction[:,i])
        return None

    def save_and_quit(self, cv):
        if self.model_save_path is not None:
            if self.neptune_log:
                my_run_id = self.run["sys/id"].fetch()
            else:
                my_run_id = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")

            if cv:
                cv_string = f"_CV{cv}"
            else:
                cv_string = ""

            path = os.path.join(self.model_save_path, my_run_id+cv_string)
            self.model.save(path)
            
            error_path = os.path.join(self.model_save_path, my_run_id+cv_string+"_residuals.h5")
            hf = h5py.File(error_path, "w")

            # Train
            train_group = hf.create_group("train")
            train_group.create_dataset("true", data=self.y_train_true_rescaled)
            train_group.create_dataset("predicted", data=self.y_train_predicted_rescaled)
            train_group.create_dataset("persistence", data=self.train_poc_prediction)

            validate_group = hf.create_group("validate")
            validate_group.create_dataset("true", data=self.y_validate_true_rescaled)
            validate_group.create_dataset("predicted", data=self.y_validate_predicted_rescaled)
            validate_group.create_dataset("persistence", data=self.validate_poc_prediction)

            hf.close()

        if self.neptune_log and cv==False:
            self.run.stop()

        tf.keras.backend.clear_session()
        return None
        
    def pass_info_to_neptune(self, info_dict):
        """
        
        """
        for k,v in info_dict.items():
            self.run[k] = v
        return None
    
    def do_it_all(self, train_validate_date="2020-09-27", end_date=None, cv=False, verbose=True):
        "Note that CV should start at 1 so that the first iteration, evaluates as 1 or True"
        if verbose:
            print(" Starting ".center(40, "="))
            print(" Preprocessing ".center(40, "-"))
        self.preprocess_joint_data(train_validate_date, end_date=end_date, verbose=verbose)
        if verbose:
            print(" Preprocessing Complete ".center(40, "-"))
            print(" Creating Windows ".center(40, "-"))
        self.create_windows(True)
        if verbose:
            print(" Windows Complete ".center(40, "-"))
            print(" Creating and Fitting Models ".center(40, "-"))
        self.compile_model()
        self.fit_model()  # TODO check that all prediction timesteps predict SOMETHING and if not, penalize
        if verbose:
            print(" Fit Complete ".center(40, "-"))
            print(" Calculating Error Metrics ".center(40, "-"))
        self.final_error_metrics(cv=cv)
        if verbose:
            print(" Error Metrics Complete ".center(40, "-"))
            print(" Saving ".center(40, "-"))
        self.save_and_quit(cv=cv)
        if verbose:
            print(" Saved ".center(40, "-"))
            print(" Complete! ".center(40, "="))
        return
    
    def cross_validated_study(self,
        train_validate_dates=["2019-09-27", "2020-09-27", "2021-09-27"],
        end_dates= ["2020-09-27", "2021-09-27", "2022-09-27"],
        verbose=True
    ):
        for i in range(len(train_validate_dates)):
            if verbose:
                print(f" Starting CV {i+1} ".center(40, "="))
            self.do_it_all(train_validate_dates[i], end_dates[i], cv=i+1, verbose=verbose)
            if verbose:
                print(f" CV {i+1} Complete ".center(40, "="))
        return

    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

        self.model
        return