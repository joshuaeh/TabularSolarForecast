# imports
import os
from solarprophet import TabularTest, SOLARPROPHET_PATH
# Functions

# Scripts
tt = TabularTest(neptune_run_name="Flexible Model Test",
    scaler_type="minmax",
    n_steps_in=13,
    n_steps_out=12,
    selected_responses=["GHI"],
    neptune_log=False,
    model_save_path=os.path.join(SOLARPROPHET_PATH, "results"),
    selected_groups=["Irradiance", "Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"])
tt.do_it_all()