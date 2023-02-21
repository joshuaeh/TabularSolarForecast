# imports
import os
from solarprophet import TabularTest
import constants
import utils
import tensorflow as tf



# script
FEATURE_GROUPS_TESTS = [
    # The autocorrelated variable itself
    [],
    # Just GHI
    ["Irradiance"], 
    # GHI and other Irradiance Measures
    ["Irradiance", "Decomposed Irradiance"],
    ["Irradiance", "Lagged 10 min GHI"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance"],
    # Time components
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Year"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Year"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time Milestones"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time Milestones"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day", "Time of Year"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones"],
    # Clear Sky
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Prev Hour Stats"],
    # ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Future Clear Sky"],
    # ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Future Clear Sky"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Prev Hour Stats"],
    # ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Future Clear Sky", "Prev Hour Stats"],
    # ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Future Clear Sky", "Prev Hour Stats"],
    # MET DATA
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Meteorological Measurements"],
    # ASI
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "ASI-16"],
    # Combined Data
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky",  "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Prev Hour Stats", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    # Remove one feature
    ["Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year",  "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats",  "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats",  "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements"],
    # other
    ["Trig Time of Day"],
    ["Trig Time of Year"],
    ["Trig Time of Day", "Trig Time of Year" ],
    ["Trig Time Milestones"],
    ["Clear Sky"],
    ["Meteorological Measurements"],
    ["ASI-16"],
    ["Trig Time Milestones", "Clear Sky"],
    ["Trig Time Milestones", "Meteorological Measurements"],
    ["Trig Time Milestones", "ASI-16"],
    ["Trig Time Milestones", "Clear Sky", "Meteorological Measurements"],
    ["Trig Time Milestones", "Clear Sky", "ASI-16"],
    ["Trig Time Milestones", "Meteorological Measurements", "ASI-16"],
    ["Trig Time Milestones", "Clear Sky", "Meteorological Measurements",  "ASI-16"]
]

FEATURE_GROUPS_TESTS = [
    ["Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year",  "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones",  "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats",  "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats",  "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky",  "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements"]
]

def get_keys_from_values(values_list, key_dictionary):
    keys = []
    for value in values_list:
        for d_key, d_value in key_dictionary.items():
            if d_value == value:
                keys.append(d_key)
    return keys

tests = []

for loss_func in["mae", "mse"]:
    for response_variable in constants.RESPONSE_ORDER:
        for feature_groups in FEATURE_GROUPS_TESTS:
            test = [loss_func, response_variable, feature_groups]
            tests.append(test)

for i, test in enumerate(tests):
    test_loss_func, test_response_variable, test_selected_feature_groups = test
    
    lr = tf.optimizers.schedules.CosineDecayRestarts(
        1e-2,     # initial learning rate
        500,     # number of steps to initially decay over
        t_mul=4,  # number of times to restart
        m_mul=0.5, # restart maximum learning rate multiplier
        alpha=0
        )

    # initialize experiment object
    tt = TabularTest(
        # Required Parameters
        n_steps_in=13,                  
        n_steps_out=12,                  
        selected_features=None,    
        selected_groups=test_selected_feature_groups,        
        selected_responses=[test_response_variable],  
        scaler_type="minmax",       
        data_path=constants.JOINT_DATA_H5_PATH,            

        ## Saving and cache parameters (optional)
        model_save_path="results",       
        datetimes_cache_path=constants.DATETIME_VALUES_PATH,  
        window_cache_path=os.path.join("data", "windows_cache.h5"),
        all_past_features=constants.PAST_FEATURES,
        all_future_features=constants.FUTURE_FEATURES,
        all_scalar_responses=constants.SCALAR_RESPONSES,
        all_relative_responses=constants.RELATIVE_RESPONSES,

        # Model (optional)
        model=None,  
        model_name="TimeDistributed_CNN-LSTM-MLP",  

        # Optimizer and training tuning parameters (optional)
        scale_responses=True,
        epochs=2000, 
        shuffle_training_order=False, 
        batch_size=5000,
        loss=test_loss_func,
        optimizer=None,
        learning_rate=lr,  # this should be static or a tf.optimizers.schedules object
        callbacks = [],
        early_stopping=True, 
        stopping_patience=500,
        stopping_min_delta=1e-4,
        metrics = [],  # todo possibly replace
        fit_verbose=1,

        # Utility Parameters (optional)
        data_cols = constants.DATA_COLS, 
        feature_groups = constants.FEATURE_GROUPS, 
        scalar_response = constants.SCALAR_RESPONSES, 
        relative_response = constants.RELATIVE_RESPONSES,
        seed=42,
        n_job_workers=10,

        # Neptune Parameters (optional) 
        neptune_log=True,
        neptune_run_name = f"drop one feature group: {i}",
        tags="-1"
        )


    train_validate_dates=["2019-09-27", "2020-09-27", "2021-09-27"]
    end_dates= ["2020-09-27", "2021-09-27", "2022-09-27"]
    for i in range(len(train_validate_dates)):
        print(f" Starting CV {i+1} ".center(40, "="))
        tt.import_preprocess_cached_windows(
        train_validate_date=train_validate_dates[i],
        end_date=end_dates[i],
        verbose=2
        )
        tt.compile_model()
        tt.fit_model()    
        print(" Fit Complete ".center(40, "-"))
        print(" Calculating Error Metrics ".center(40, "-"))
        tt.final_error_metrics(cv=i+1)
        print(" Error Metrics Complete ".center(40, "-"))
        print(" Saving ".center(40, "-"))
        tt.save_and_quit(cv=i+1)
        print(" Saved ".center(40, "-"))
        print(" Complete! ".center(40, "="))
    tt.run.stop()