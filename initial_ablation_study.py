# imports
import os
from xml.sax.handler import feature_external_ges
from solarprophet import TabularTest

# Functions

# script
feature_external_groups =  ["Irradiance","Decomposed Irradiance","Lagged 10 min GHI","Lagged 10 min Decomposed Irradiance","Time of Day","Trig Time of Day",
    "Time of Year","Trig Time of Year","Time Milestones","Trig Time Milestones","Clear Sky","Future Clear Sky","Prev Hour Stats","Meteorological Measurements","ASI-16"]
responses = ["GHI", 'CSI GHI', 'cs_dev t ghi', "Delta GHI", "Delta CSI"]
feature_tests=[
    ["Irradiance"],
    ["Irradiance", "Decomposed Irradiance"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Year"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time Milestones"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "ASI-16"],
    ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Trig Time of Day", "Trig Time of Year", "Trig Time Milestones", "Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"]
]
for response in responses:
    for test in feature_tests:
        print(f"Response: {response} Features:{test}")