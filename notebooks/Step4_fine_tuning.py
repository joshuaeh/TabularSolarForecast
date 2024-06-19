# imports
import os
from solarprophet import TabularTest
import constants
import utils
import tensorflow as tf
import pandas as pd

target_var = "Delta CSI GHI"  # {cs_dev, CSI GHI, Delta CSI GHI, Delta GHI, GHI}
filename = f"feature_importance_{target_var}_CV3_iteration1.csv"
feature_impostance_results_path = os.path.join('results', 'feature_importance', filename)
df = pd.read_csv(feature_impostance_results_path)
df.rename(columns={'Unnamed: 0': 'feature'}, inplace=True)

def fine_tuned_iteration(fine_tuned_features):
    
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
        n_steps_in=1,                  
        n_steps_out=12,                  
        selected_features=fine_tuned_features,    
        selected_groups=None,   
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
        loss="mae",
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
        neptune_run_name = f"Fine Tuning Feature number",
        tags=["CV2", "Time Steps"]
        )

    original_epochs = tt.epochs
    
    train_validate_dates=["2021-09-27"]
    end_dates= ["2022-09-27"]
    for i in range(len(train_validate_dates)):
        print(f" Starting Fine Tuning ".center(40, "="))
        tt.import_preprocess_cached_windows(
        train_validate_date=train_validate_dates[i],
        end_date=end_dates[i],
        verbose=2
        )
        
        # compile the model to fit for 500 epochs with mse then mae
        
        remaining_epochs = max(500, original_epochs - 500)
        tt.loss = "mse"
        tt.epochs = 500
        tt.compile_model()
        tt.fit_model()  
        
        # extract the weights and recompile with new loss function
        weights = tt.model.get_weights()
        tt.loss = "mae"
        tt.epochs = remaining_epochs
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

num_features = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
for n in num_features:
    fine_tuned_features = df["feature"].iloc[:n].to_list()
    fine_tuning_tags = f"Fine Tuning {n} Features"
    fine_tuned_iteration(fine_tuned_features)
