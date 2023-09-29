# imports
import os
from solarprophet import TabularTest
import constants
import utils
import tensorflow as tf
        
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
    
lr = tf.optimizers.schedules.CosineDecayRestarts(
    1e-2,     # initial learning rate
    500,     # number of steps to initially decay over
    t_mul=4,  # number of times to restart
    m_mul=0.5, # restart maximum learning rate multiplier
    alpha=0
    )

rmse = tf.keras.metrics.RootMeanSquaredError()

# initialize experiment object
tt = TabularTest(
    # Required Parameters
    n_steps_in=1,                  
    n_steps_out=12,                  
    selected_features=None,    
    selected_groups=["Time of Day", "Time of Year", "Decomposed Irradiance", "Lagged 10 min GHI",
                        "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements",
                        "ASI-16"],        
    selected_responses=["Delta CSI GHI"],  
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
    epochs=3000, 
    shuffle_training_order=False, 
    batch_size=4000,
    loss='mean_squared_error',
    optimizer=None,
    learning_rate=lr,  # this should be static or a tf.optimizers.schedules object
    callbacks = [],
    early_stopping=True, 
    stopping_patience=1000,
    stopping_min_delta=1e-4,
    metrics = [],  # todo possibly replace
    fit_verbose=1,

    # Utility Parameters (optional)
    data_cols = constants.DATA_COLS, 
    feature_groups = constants.FEATURE_GROUPS, 
    scalar_response = constants.SCALAR_RESPONSES, 
    relative_response = constants.RELATIVE_RESPONSES,
    seed=42,
    n_job_workers=1,

    # Neptune Parameters (optional) 
    neptune_log=True,
    tags=["CV3", "RMSE Training"]
    )

original_epochs = tt.epochs

train_validate_date = "2021-09-27"
end_date = "2022-09-27"

print(f" Starting CV 3 ".center(40, "="))
tt.import_preprocess_cached_windows(
    train_validate_date=train_validate_date,
    end_date=end_date,
    verbose=2
    )
tt.compile_model()
tt.fit_model() 
    
print(" Fit Complete ".center(40, "-"))
print(" Calculating Error Metrics ".center(40, "-"))
tt.final_error_metrics(cv=3)
print(" Error Metrics Complete ".center(40, "-"))
print(" Saving ".center(40, "-"))
tt.save_and_quit(cv=3)
print(" Saved ".center(40, "-"))
print(" Complete! ".center(40, "="))

tt.run.stop()