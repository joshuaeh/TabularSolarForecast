# imports
import os
from solarprophet import TabularTest, SOLARPROPHET_PATH
from constants import RESPONSE_ORDER, FEATURE_GROUPS, CSV_COLS, GROUP_ABBREVIATIONS, RESPONSE_FEATURES, SOLARPROPHET_PATH

# Functions
def features_to_one_hot_feature_dictionary(selected_features):
    included_features_dictionary = {}
    for feature in CSV_COLS:
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

def create_features_list_from_groups(response_variable, FEATURE_GROUPS, RESPONSE_FEATURES, selected_groups=None, add_response_variable=True):
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
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Future Clear Sky"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Future Clear Sky"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Future Clear Sky", "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Clear Sky", "Future Clear Sky", "Prev Hour Stats"],
    # MET DATA
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Meteorological Measurements"],
    # ASI
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "ASI-16"],
    # Combined Data
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day",  "Time of Year", "Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"]
]

def get_keys_from_values(values_list, key_dictionary):
    keys = []
    for value in values_list:
        for d_key, d_value in key_dictionary.items():
            if d_value == value:
                keys.append(d_key)
    return keys

for response_variable in RESPONSE_ORDER[1:]:
    feature_groups_abbreviated = ['TTOD', 'TTOY', 'ASI']
    feature_groups = get_keys_from_values(feature_groups_abbreviated, GROUP_ABBREVIATIONS)
    selected_features = create_features_list_from_groups(response_variable, FEATURE_GROUPS, RESPONSE_FEATURES, selected_groups=feature_groups, add_response_variable=True)

    tt = TabularTest(neptune_run_name="Flexible Model Test",
            scaler_type="minmax",
            n_steps_in=13,
            n_steps_out=12,
            selected_responses=[response_variable],
            neptune_log=True,
            model_save_path=os.path.join(SOLARPROPHET_PATH, "results"),
            selected_features=selected_features,
            tags=["Ablation Study"],
            n_job_workers=False
            )
    tt.do_it_all()