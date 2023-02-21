import os

import pandas as pd
import h5py
import numpy as np

import dask.dataframe as dd 
import neptune.new as neptune

import constants

possible_experiment_filenames = [i for i in os.listdir(os.path.join(constants.SOLARPROPHET_PATH, "results")) if i.endswith('.h5')]

def features_to_one_hot_feature_dictionary(selected_features):
    included_features_dictionary = {}
    for feature in constants.DATA_COLS:
        if feature in selected_features:
            included_features_dictionary[feature] = 1
        else:
            included_features_dictionary[feature] = 0
    return included_features_dictionary

def features_to_one_hot_groups_dictionary(selected_features, GROUPS_ORDER, FEATURE_GROUPS):
    included_groups_dictionary = {}
    for group in GROUPS_ORDER:
        values = FEATURE_GROUPS[group]
        #if all(group_feature in selected_features for group_feature in values):
        if set(values).issubset(selected_features):
            included_groups_dictionary[group] = 1
        else:
            included_groups_dictionary[group] = 0
    return included_groups_dictionary

def create_features_list_from_groups(selected_groups, response_variable, FEATURE_GROUPS, RESPONSE_FEATURES, add_response_variable=True):
    if add_response_variable:
        features_list = [RESPONSE_FEATURES[response_variable]]
    else:
        features_list = []
    if selected_groups:
        for group in selected_groups:
            group_features = FEATURE_GROUPS[group]
            for feature in group_features:
                if feature not in features_list:
                    features_list.append(feature)
    return features_list

# import data utils
def load_joint_data(path=None, data_cols=constants.DATA_COLS):
    if path.endswith(".h5"):
        # load h5 dataset (takes about 20 seconds on my pc) (7 to load, 13 to convert datetimes)
        print("Loading .h5 dataset")
        data_df = pd.read_hdf(path, key="joint_data")
        data_df.index = pd.DatetimeIndex(data_df.index, tz="MST")

    elif path.endswith(".csv"):
        print("Loading csv dataset")
        # import joint_data csv (takes about 100 seconds on my computer)
        data_df = dd.read_csv(path, usecols=data_cols, parse_dates=["dateTime"]).set_index("dateTime")
        data_df = data_df.compute()
        data_df.index = pd.DatetimeIndex(data_df.index, tz="MST")

        # Save dataset as h5 file for the future (takes 30 seconds on my computer)
        h5_data_df = data_df.copy()
        h5_data_df.index = h5_data_df.index.astype(str)
        h5_data_df.to_hdf(r"data\joint_data.h5", key="joint_data", mode="w")

    else:
        raise Exception("No joint_data.csv or joint_data.h5 file found in [SOLARPROPHET]/data/")
    return data_df

def load_neptune_runs(neptune_cache_path=constants.NEPTUNE_CACHE_PATH, neptune_project=constants.NEPTUNE_PROJECT, neptune_token=constants.NEPTUNE_TOKEN):
    drop_cols = ['sys/description', 'sys/hostname', 'sys/name', 'sys/owner',
       'sys/state', 'sys/tags', 'CV 0: Train nMAP', 'CV 0: Validate nMAP',
       'CV 1: Train nMAP', 'CV 1: Validate nMAP', 'CV 2: Train nMAP',
       'CV 2: Validate nMAP', 'Model Structure', 'Test FS nMAP',
       'Test Persistence nMAP', 'Test nMAP', 'Total Test FS nMAP',
       'Total Test Persistence nMAP', 'Total Test nMAP', 'Total Train FS nMAP',
       'Total Train Persistence nMAP', 'Total Train nMAP', 'Train FS nMAP',
       'Train Persistence nMAP', 'Train nMAP', 'Validate FS nMAP',
       'Validate Persistence nMAP', 'Validate nMAP', 'history',
       'name', 'source_code/entrypoint', 'source_code/git',
       'source_code/integrations/neptune-tensorflow-keras',
       'sys/creation_time', 'sys/modification_time', 'sys/ping_time',
       'monitoring/stderr', 'monitoring/stdout', 'monitoring/traceback']

    str_cols = ['scaler type', 'selected features', 'selected responses', 'loss function', 'sys/id']

    if os.path.isfile(neptune_cache_path):
        # load neptune cache
        runs_table_df = pd.read_hdf(neptune_cache_path, index_col=0)
        for col in str_cols:
            runs_table_df[col] = runs_table_df[col].astype(str)
        runs_table_df.head()

    else:
        # neptune data
        neptune_project = neptune.init_project(
            name=neptune_project,
            api_token=neptune_token,
            mode="read-only"
            )
        runs_table_df = neptune_project.fetch_runs_table().to_pandas()

        runs_table_df.drop(columns=drop_cols, inplace=True)
        runs_table_df.to_hdf(constants.NEPTUNE_CACHE_PATH, key="neptune_cache", mode="w")
    return runs_table_df

# Find missing values
def perdelta(start, end, delta):
    """Function to generate a list of datetime objects between two datetime objects"""
    curr = start
    while curr < end:
        yield curr
        curr += delta

def get_experiment_datetimes(number_of_steps_in, datetime_values_path=constants.DATETIME_VALUES_PATH):
    """datetimes are saved in h5 file that is organized by [# steps in][# windows, # total steps]
    
    """
    with h5py.File(datetime_values_path, 'r') as f:
        intermediates = f[str(int(number_of_steps_in))][:]

    datetimes = pd.DatetimeIndex(intermediates, tz="MST")
    return datetimes

def get_experiment_residuals(experiment_id, pca_df, experiment_path=None):
    # SPTAB-581 to SPTAB-1092 are feature ablation experiments
    #     Total experiments: 512
    #     tf model saved as dir: results/SPTAB-####/
    #     h5 file saved as: results/SPTAB-####_CV_residuals.h5
    #         h5 file contains: [train, test][true, persistence, predicted]
    # SPTAB-1150 to 1272 are PCA experiments
    #     Total experiments: 123
    #     tf model saved as dir: results/SPTAB-####/
    #     h5 file saved as: results/SPTAB-####_residuals.h5
    #         h5 file contains: [train, validate, test][true, persistence, predicted]

    if not experiment_path:
        experiment_number = int(experiment_id.split('-')[1])
        experiment_type = 'feature_ablation' if experiment_number < 1150 else 'pca'
        experiment_h5_filename = experiment_id + '_CV_residuals.h5' if experiment_type == 'feature_ablation' else experiment_id + '_residuals.h5'
        assert experiment_h5_filename in possible_experiment_filenames, "Experiment not found"
        experiment_path = os.path.join(constants.SOLARPROPHET_PATH, "results", experiment_h5_filename)

    with h5py.File(experiment_path, 'r') as f:
        if experiment_type == 'feature_ablation':
            train_intermediate = f["train"]
            test_intermediate = f["test"]
            results_intermediate = [train_intermediate, test_intermediate]
            
        else:
            train_intermediate = f["train"]
            validate_intermediate = f["validate"]
            test_intermediate = f["test"]
            results_intermediate = [train_intermediate, validate_intermediate, test_intermediate]
        results = [np.array([i["true"], i["predicted"], i["persistence"]]) for i in results_intermediate]
    results = np.concatenate(results, axis=1)

    if experiment_id in pca_df["sys/id"].values:
        experiment_n_steps_in = pca_df[pca_df["sys/id"] == experiment_id]["n_steps_in"].values[0]
    else:
        experiment_n_steps_in = 13

    experiment_datetimes = get_experiment_datetimes(experiment_n_steps_in)[:,experiment_n_steps_in:]
    # experiment_datetimes = np.expand_dims(experiment_datetimes, axis=0)

    return experiment_datetimes, results

### Error Metrics ###
### MBE
def MBE(y_true, y_predicted, axis=None):
    """
    Calculate the mean bias error between the two measurements.
    MBE = mean(y_predicted - y_true)
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the MBE of the flattened array
        axis=0 will calculate the MBE of each time step
        axis=1 will calculate the MBE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the MBE over

    :return: The MBE
    """
    return np.subtract(y_predicted, y_true).mean(axis=axis)

### nMBE
def nMBE(y_true, y_predicted, axis=None):
    """
    Calculate the normalized mean bias error between the two measurements.
    nMBE = MBE / mean(y_true)
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the nMBE of the flattened array
        axis=0 will calculate the nMBE of each time step
        axis=1 will calculate the nMBE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the nMBE over

    :return: The nMBE
    """
    numerator = MBE(y_true, y_predicted, axis=axis)
    denominator = y_true.mean(axis=axis)
    return numerator / denominator

### MAE
def MAE(y_true, y_predicted, axis=None):
    """
    Calculate the mean absolute error between the two measurements.
    MAE = mean(abs(y_true - y_predicted))
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the MAE of the flattened array
        axis=0 will calculate the MAE of each time step
        axis=1 will calculate the MAE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the MAE over

    :return: The mean absolute error
    """
    return np.absolute(np.subtract(y_true, y_predicted)).mean(axis=axis)

### nMAE
def nMAE(y_true, y_predicted, axis=None):
    """
    Calculate the normalized mean absolute error between the two measurements.
    nMAE = MAE / y_true.mean()
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the nMAE of the flattened array
        axis=0 will calculate the nMAE of each time step
        axis=1 will calculate the nMAE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the nMAE over

    :return: The nMAE
    """
    numerator = MAE(y_true, y_predicted, axis=axis)
    denominator = y_true.mean(axis=axis)
    return numerator / denominator

### MSE
def MSE(y_true, y_predicted, axis=None):
    """
    Calculate the mean squared error between the two measurements.
    MSE = mean((y_predicted-y_true)**2)
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the MSE of the flattened array
        axis=0 will calculate the MSE of each time step
        axis=1 will calculate the MSE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the MSE over

    :return: The mean squared error
    """
    return np.square(np.subtract(y_predicted, y_true)).mean(axis=axis)

### RMSE
def RMSE(y_true, y_predicted, axis=None):
    """
    Calculate the root mean squared error between the two measurements.
    RMSE = sqrt((y_predicted-y_true)**2).mean()
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the RMSE of the flattened array
        axis=0 will calculate the RMSE of each time step
        axis=1 will calculate the RMSE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the RMSE over

    :return: The root mean squared error
    """
    mse = MSE(y_true, y_predicted, axis=axis)
    return np.sqrt(mse)

### nRMSE
def nRMSE(y_true, y_predicted, axis=None):
    """
    Calculate the normalized root mean squared error between two measurements.
    nRMSE = sqrt((y_predicted-y_true)**2).mean() / y_true.mean()
    Assuming each row is a different measurement, and each column is a different time step,
        axis=None will calculate the nRMSE of the flattened array
        axis=0 will calculate the nRMSE of each time step
        axis=1 will calculate the nRMSE of each measurement

    :param y_true: The true values
    :param y_predicted: The predicted values
    :param axis: The axis to calculate the nRMSE over

    :return: The normalized root mean squared error
    """
    numerator = RMSE(y_true, y_predicted, axis=axis)
    denominator = y_true.mean(axis=axis)
    return numerator / denominator

### Persistence of Cloudiness calculation ###

def persistence_of_cloudiness_prediction(clear_sky_indexes, clear_sky_irradiances, n_steps_in, n_steps_out):
    """
    Calculate the persistence of cloudiness prediction for the given clear sky indexes and clear sky irradiances.
    Given a prediction at time i, the persistence of cloudiness prediction of time j is:
        POC_j = Clear_Sky_Index_i * Clear_Sky_Irradiance_j

    :param clear_sky_indexes: (,n_steps_in+n_steps_out) The measured clear sky indexes for each time step
    :param clear_sky_irradiances: (,n_steps_in+n_steps_out) The ideal clear sky irradiances for each time step
    :param n_steps_in: The number of time steps to use for the input
    :param n_steps_out: The number of time steps to use for the prediction

    :return: (,n_steps_out) The persistence of cloudiness prediction for each predicted time step
    """
    prediction_csi = clear_sky_indexes[:, n_steps_in-1]  # clear sky index at the time of prediction
    return (prediction_csi * clear_sky_irradiances[:, n_steps_in:].T).T