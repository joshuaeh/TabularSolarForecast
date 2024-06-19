#!/usr/bin/env python3
# Author: Joshua Hammond
# Static parameters for the project

import os
import datetime
from pathlib import Path

import config

# TODO make API tokens and other specifications in another file called "auth" or something similar that is not commited
# TODO make file structure standardized
# TODO I'd like to refactor to either specify what constants are imported from constants.py or use constants.CONSTANT syntax to assist in debugging

# Directories
# HOME = os.environ["HOME"]
# WORK = os.environ["WORK"]
SOLARPROPHET_PATH = Path.cwd()
DATA_DIR = os.path.join(SOLARPROPHET_PATH, "data")
JOINT_DATA_H5_PATH = os.path.join(DATA_DIR, "joint_data.h5")

# Neptune connection
NEPTUNE_PROJECT = config.NEPTUNE_PROJECT
NEPTUNE_TOKEN = config.NEPTUNE_TOKEN
NEPTUNE_CACHE_PATH = os.path.join(DATA_DIR, "neptune_cache.h5")

# Misc Saved results and and caches
DATETIME_VALUES_PATH = os.path.join(DATA_DIR, "datetimevalues.h5")
IDEAL_SERIES_PATH = os.path.join(DATA_DIR, "ideal_series.csv")

# Experimental parameters
BEGIN_DATE = datetime.date(2017, 9, 27)
CV0_DATE = datetime.date(2019, 9, 27)
CV1_DATE = datetime.date(2020, 9, 27)
CV2_DATE = datetime.date(2021, 9, 27)
CV3_DATE = datetime.date(2022, 9, 27)
END_DATE = datetime.date(2023, 9, 27)

# Data
NREL_DATA_DIR = Path(DATA_DIR, "NREL")
ASI_DATA_DIR = Path(NREL_DATA_DIR, "ASI")
WEATHER_DATA_DIR = Path(NREL_DATA_DIR, "weather")
MIDC_STATIONS = {  # non-exhaustive list, see https://midcdmz.nrel.gov/apps/data_api_doc.pl
    # Also station list api: https://midcdmz.nrel.gov/apps/data_api_doc.pl?_idtextlist_
    # Also metadata api: https://midcdmz.nrel.gov/apps/aim_api_doc.pl?
    "SRRLASI": "NREL SRRL All-Sky Imager (ASI-16)",
    "AOCS": "NREL Solar Radiation Research Laboratory (AOCS)",
    "AODSRRLBM": "NREL Solar Radiation Research Laboratory (AOD ESR-MRI L2)",
    "AODSRRLAM": "NREL Solar Radiation Research Laboratory (AOD ESR-MRI L2A)",
    "AODSRRL0E": "NREL Solar Radiation Research Laboratory (AOD ESR-Sunrad L0)",
    "AODSRRL1E": "NREL Solar Radiation Research Laboratory (AOD ESR-Sunrad L1.1)",
    "AODSRRLBE": "NREL Solar Radiation Research Laboratory (AOD ESR-Sunrad L2)",
    "AODSRRLAE": "NREL Solar Radiation Research Laboratory (AOD ESR-Sunrad L2A)",
    "AODSRRLBC": "NREL Solar Radiation Research Laboratory (AOD SR-CEReS L2)",
    "AODSRRL0S": "NREL Solar Radiation Research Laboratory (AOD SkyRad L0)",
    "AODSRRL1S": "NREL Solar Radiation Research Laboratory (AOD SkyRad L1.1)",
    "ATI": "NREL Solar Radiation Research Laboratory (ATI RSP)",
    "BMS": "NREL Solar Radiation Research Laboratory (BMS)",
    "BSRN": "NREL Solar Radiation Research Laboratory (BSRN)",
    "PWVSRRL": "NREL Solar Radiation Research Laboratory (GPS-based PWV)",
    "IRRSP": "NREL Solar Radiation Research Laboratory (Irradiance Inc. RSP v2)",
    "RAZON": "NREL Solar Radiation Research Laboratory (RaZON+)",
    "RSP": "NREL Solar Radiation Research Laboratory (Schott RSP)",
    "MFR": "NREL Solar Radiation Research Laboratory (YES MFR)",
    "TSR": "NREL Solar Radiation Research Laboratory (YES TSR)",
    "AUXB": "NREL Solar Radiation Research Laboratory (1-Axis Tracker Status)",
    "AUXC": "NREL Solar Radiation Research Laboratory (High Wind Alerts)",
    "AUXE": "NREL Solar Radiation Research Laboratory (Research Channels #1)",
    "AUXF": "NREL Solar Radiation Research Laboratory (Research Channels #2)",
    "AUXD": "NREL Solar Radiation Research Laboratory (Tracker and TEC Status)",
    "AUXA": "NREL Solar Radiation Research Laboratory (Ancillary Data)",
    "PVSBMSMP": "NREL Solar Radiation Research Laboratory (BMS-Samples)",
    "PVSFTSMP": "NREL Solar Radiation Research Laboratory (FT-Samples)",
    "PVSFT": "NREL Solar Radiation Research Laboratory (Fixed Tilt)",
    "PVSGHSMP": "NREL Solar Radiation Research Laboratory (GH-Samples)",
    "PVSGH": "NREL Solar Radiation Research Laboratory (Global Horiz)",
    "PVSSASMP": "NREL Solar Radiation Research Laboratory (ST-Samples)",
    "PVSSA": "NREL Solar Radiation Research Laboratory (PV Resource)",
    "SSIM": "NREL Solar Radiation Research Laboratory (SolarSIM-D2+ Direct Normal)",
    "SSIMG": "NREL Solar Radiation Research Laboratory (SolarSIM-G Global Horizontal)",
    "AODSTAC0E": "Solar Technology Acceleration Center (AOD ESR Level 0)",
    "AODSTAC1E": "Solar Technology Acceleration Center (AOD ESR Level 1.1)",
    "AODSTAC0S": "Solar Technology Acceleration Center (AOD SkyRad Level 0)",
    "AODSTAC1S": "Solar Technology Acceleration Center (AOD SkyRad Level 1.1)",
    "PWVSTAC": "Solar Technology Acceleration Center (GPS-based PWV)",
    "STAC": "Solar Technology Acceleration Center (SolarTAC)",
}
ASI_URL_TEMPLATE = "https://midcdmz.nrel.gov/tsi/SRRLASI/{YYYY}/{YYYYMMDD}.zip"
BMS_URL_TEMPLATE = (
    "https://midcdmz.nrel.gov/apps/data_api.pl?site=BMS&begin={YYYYMMDD}&end={YYYYMMDD}"
)
BMS_DATES = [
    [datetime.date(1981, 7, 15), datetime.date(1983, 6, 30)],
    [datetime.date(1984, 10, 25), datetime.date(1985, 4, 7)],
    [datetime.date(1985, 4, 8), datetime.date(1988, 2, 29)],
    [datetime.date(1988, 3, 14), datetime.date(1999, 1, 11)],
    [datetime.date(1999, 1, 13), datetime.date(2000, 7, 31)],
    [datetime.date(2000, 8, 1), datetime.date(2001, 10, 31)],
    [datetime.date(2000, 11, 1), datetime.date(2001, 8, 3)],
    [datetime.date(2001, 8, 4), datetime.date(2003, 12, 31)],
    [datetime.date(2004, 1, 1), datetime.date(2005, 12, 31)],
    [datetime.date(2006, 1, 1), datetime.date(2008, 8, 31)],
    [datetime.date(2008, 9, 1), datetime.date(2012, 3, 31)],
    [datetime.date(2012, 6, 1), datetime.date(2014, 12, 31)],
    [datetime.date(2015, 1, 1), datetime.date(2017, 11, 30)],
    [datetime.date(2017, 12, 1), datetime.date(2019, 12, 31)],
    [datetime.date(2020, 1, 1), datetime.date(2023, 12, 31)],
    [datetime.date(2024, 1, 1), datetime.date.today() - datetime.timedelta(days=2)],
]
BMS_CONFIGURATION_CHANGE_DATES = [date_pair[-1] for date_pair in BMS_DATES]
IMAGE_FILE_TYPE_MAPPINGS = {
    "_11.jpg": "_Raw_NE.jpg",
    "_11_NE.jpg": "_Projection_NE.jpg",
    "_12.jpg": "_Raw_UE.jpg",
    "_12_UE.jpg": "_Projection_UE.jpg",
    "_1112_BRBG.png": "_Segmented_BRBG.png",
    "_1112_CDOC.png": "_Segmented_CDOC.png",
    ".txt": "_Image_Data.txt",
}

DATA_COLS = [
    "dateTime",
    # Irradiance
    "GHI",
    "DNI",
    "DHI",
    # Clouds (camera - related)
    "BRBG Total Cloud Cover [%]",
    "CDOC Total Cloud Cover [%]",
    "CDOC Thick Cloud Cover [%]",
    "CDOC Thin Cloud Cover [%]",
    "HCF Value",
    "Blue/Red_min",
    "Blue/Red_mid",
    "Blue/Red_max",
    # Clear Sky Irradiance
    "clearsky ghi",
    "clearsky dni",
    "clearsky dhi",
    "cs_dev t ghi",
    "cs_dev t dni",
    "cs_dev t dhi",
    "CSI GHI",
    "CSI DNI",
    "CSI DHI",
    # Time
    "TOD",
    "TOD Sin",
    "TOD Cos",
    "TOY",
    "TOY Sin",
    "TOY Cos",
    "time from sunrise",
    "cos time from sunrise",
    "sin time from sunrise",
    "time to solar noon",
    "cos time to solar noon",
    "sin time to solar noon",
    "time to sunset",
    "cos time to sunset",
    "sin time to sunset",
    "Day",
    "before solar noon",
    # Sun Flag (Camera - related)
    "Flag: Sun not visible",
    "Flag: Sun on clear sky",
    "Flag: Parts of sun covered",
    "Flag: Sun behind clouds, bright dot visible",
    "Flag: Sun outside view",
    "Flag: No evaluation",
    # Auxilary Weather Data
    "Precipitation [mm]",
    "Precipitation (Accumulated) [mm]",
    "Station Pressure [mBar]",
    "Tower Dry Bulb Temp [deg C]",
    "Tower RH [%]",
    "Airmass",
    "Wind NS [m/s]",
    "Wind EW [m/s]",
    "Avg Wind Speed @ 22ft [m/s]",
    "Avg Wind Direction @ 22ft [deg from N]",
    "Peak Wind Speed @ 22ft [m/s]",
    "Snow Depth [cm]",
    "Snow Depth Quality",
    "SE Dry Bulb Temp [deg C]",
    "SE RH [%]",
    "Vertical Wind Shear [1/s]",
    "Sea-Level Pressure (Est) [mBar]",
    "Tower Dew Point Temp [deg C]",
    "Tower Wet Bulb Temp [deg C]",
    "Tower Wind Chill Temp [deg C]",
    # Solar Position
    "Zenith Angle [degrees]",
    "Azimuth Angle [degrees]",
    "Solar Eclipse Shading",
    "cos zenith",
    "cos normal irradiance",
    "apparent_zenith",
    "zenith",
    "apparent_elevation",
    "elevation",
    "azimuth",
    "Sun NS",
    "Sun EW",
    # Aerosol
    "315nm POM-01 Photometer [nA]",
    "400nm POM-01 Photometer [uA]",
    "500nm POM-01 Photometer [uA]",
    "675nm POM-01 Photometer [uA]",
    "870nm POM-01 Photometer [uA]",
    "940nm POM-01 Photometer [uA]",
    "1020nm POM-01 Photometer [uA]",
    "Albedo (CM3)",
    "Albedo (LI-200)",
    "Albedo Quantum (LI-190)",
    "Broadband Turbidity",
    # Lagged Variables
    "ghi t-1",
    "dni t-1",
    "dhi t-1",
    "ghi t-2",
    "dni t-2",
    "dhi t-2",
    "ghi t-3",
    "dni t-3",
    "dhi t-3",
    "ghi t-4",
    "dni t-4",
    "dhi t-4",
    "ghi t-5",
    "dni t-5",
    "dhi t-5",
    "ghi t-6",
    "dni t-6",
    "dhi t-6",
    "ghi t-7",
    "dni t-7",
    "dhi t-7",
    "ghi t-8",
    "dni t-8",
    "dhi t-8",
    "ghi t-9",
    "dni t-9",
    "dhi t-9",
    # Lagged Window Stats
    "cs_dev_mean t-10 ghi",
    "cs_dev_mean t-10 dni",
    "cs_dev_mean t-10 dhi",
    "cs_dev_median t-10 ghi",
    "cs_dev_median t-10 dni",
    "cs_dev_median t-10 dhi",
    "csi_stdev t-10 ghi",
    "csi_stdev t-10 dni",
    "csi_stdev t-10 dhi",
    "cs_dev_mean t-60 ghi",
    "cs_dev_mean t-60 dni",
    "cs_dev_mean t-60 dhi",
    "cs_dev_median t-60 ghi",
    "cs_dev_median t-60 dni",
    "cs_dev_median t-60 dhi",
    "csi_stdev t-60 ghi",
    "csi_stdev t-60 dni",
    "csi_stdev t-60 dhi",
    "dGHI",
    "dDNI",
    "dDHI",
    # 'Global CMP22 (vent/cor) [W/m^2]',
    # 'Direct NIP [W/m^2]', 'Direct sNIP [W/m^2]',
    # 'Diffuse CM22-1 (vent/cor) [W/m^2]',
]

FEATURE_GROUPS = {
    "Irradiance": ["GHI"],
    "Decomposed Irradiance": ["DNI", "DHI"],
    "Lagged 10 min GHI": [
        "ghi t-1",
        "ghi t-2",
        "ghi t-3",
        "ghi t-4",
        "ghi t-5",
        "ghi t-6",
        "ghi t-7",
        "ghi t-8",
        "ghi t-9",
    ],
    "Lagged 10 min Decomposed Irradiance": [
        "dni t-1",
        "dhi t-1",
        "dni t-2",
        "dhi t-2",
        "dni t-3",
        "dhi t-3",
        "dni t-4",
        "dhi t-4",
        "dni t-5",
        "dhi t-5",
        "dni t-6",
        "dhi t-6",
        "dni t-7",
        "dhi t-7",
        "dni t-8",
        "dhi t-8",
        "dni t-9",
        "dhi t-9",
    ],
    "Time of Day": ["TOD"],
    "Trig Time of Day": ["TOD Sin", "TOD Cos"],
    "Time of Year": ["TOY"],
    "Trig Time of Year": ["TOY Sin", "TOY Cos"],
    "Time Milestones": [
        "time from sunrise",
        "time to solar noon",
        "time to sunset",
        "Day",
        "before solar noon",
    ],
    "Trig Time Milestones": [
        "cos time from sunrise",
        "sin time from sunrise",
        "cos time to solar noon",
        "sin time to solar noon",
        "cos time to sunset",
        "sin time to sunset",
        "Day",
        "before solar noon",
    ],
    "Clear Sky": [
        "clearsky ghi",
        "clearsky dni",
        "clearsky dhi",
        "Solar Eclipse Shading",
        "zenith",
        "elevation",
        "azimuth",
    ],
    "Prev Hour Stats": [
        "cs_dev t ghi",
        "cs_dev t dni",
        "cs_dev t dhi",
        "CSI GHI",
        "CSI DNI",
        "CSI DHI",
        "cs_dev_mean t-10 ghi",
        "cs_dev_mean t-10 dni",
        "cs_dev_mean t-10 dhi",
        "cs_dev_median t-10 ghi",
        "cs_dev_median t-10 dni",
        "cs_dev_median t-10 dhi",
        "csi_stdev t-10 ghi",
        "csi_stdev t-10 dni",
        "csi_stdev t-10 dhi",
        "cs_dev_mean t-60 ghi",
        "cs_dev_mean t-60 dni",
        "cs_dev_mean t-60 dhi",
        "cs_dev_median t-60 ghi",
        "cs_dev_median t-60 dni",
        "cs_dev_median t-60 dhi",
        "csi_stdev t-60 ghi",
        "csi_stdev t-60 dni",
        "csi_stdev t-60 dhi",
        "dGHI",
        "dDNI",
        "dDHI",
    ],
    "Meteorological Measurements": [
        "Precipitation [mm]",
        "Precipitation (Accumulated) [mm]",
        "Station Pressure [mBar]",
        "Tower Dry Bulb Temp [deg C]",
        "Tower RH [%]",
        "Airmass",
        "Wind NS [m/s]",
        "Wind EW [m/s]",
        "Avg Wind Speed @ 22ft [m/s]",
        "Avg Wind Direction @ 22ft [deg from N]",
        "Peak Wind Speed @ 22ft [m/s]",
        "Snow Depth [cm]",
        "Snow Depth Quality",
        "SE Dry Bulb Temp [deg C]",
        "SE RH [%]",
        "Vertical Wind Shear [1/s]",
        "Sea-Level Pressure (Est) [mBar]",
        "Tower Dew Point Temp [deg C]",
        "Tower Wet Bulb Temp [deg C]",
        "Tower Wind Chill Temp [deg C]",
        "315nm POM-01 Photometer [nA]",
        "400nm POM-01 Photometer [uA]",
        "500nm POM-01 Photometer [uA]",
        "675nm POM-01 Photometer [uA]",
        "870nm POM-01 Photometer [uA]",
        "940nm POM-01 Photometer [uA]",
        "1020nm POM-01 Photometer [uA]",
        "Albedo (CM3)",
        "Albedo (LI-200)",
        "Albedo Quantum (LI-190)",
        "Broadband Turbidity",
    ],
    "ASI-16": [
        "BRBG Total Cloud Cover [%]",
        "CDOC Total Cloud Cover [%]",
        "CDOC Thick Cloud Cover [%]",
        "CDOC Thin Cloud Cover [%]",
        "HCF Value",
        "Blue/Red_min",
        "Blue/Red_mid",
        "Blue/Red_max",
        "Flag: Sun not visible",
        "Flag: Sun on clear sky",
        "Flag: Parts of sun covered",
        "Flag: Sun behind clouds, bright dot visible",
        "Flag: Sun outside view",
        "Flag: No evaluation",
    ],
}  # TODO add future features
# TODO use .ANONYMOUS as an argument when we are debugging
# TODO save model weights, go through Neptune best practices
RESPONSE_FEATURES = {  # to compare in a standardized manner, we should always include the value being predicted as a feature. We will make sure to do that here
    # TODO figure how to not include a feature twice
    "GHI": "GHI",
    "CSI GHI": "CSI GHI",
    "cs_dev t ghi": "cs_dev t ghi",
    "Delta GHI": "GHI",
    "Delta CSI GHI": "CSI GHI",
}
GROUP_ABBREVIATIONS = {
    "Irradiance": "GHI",  # I like this one the least
    "Decomposed Irradiance": "DI",  # This should in reality include I implicitly
    "Lagged 10 min GHI": "L10GHI",
    "Lagged 10 min Decomposed Irradiance": "L10DI",
    "Time of Day": "TOD",
    "Trig Time of Day": "TTOD",
    "Time of Year": "TOY",
    "Trig Time of Year": "TTOY",
    "Time Milestones": "TM",
    "Trig Time Milestones": "TTM",
    "Clear Sky": "CS",
    "Prev Hour Stats": "L60Stats",
    "Meteorological Measurements": "MET",
    "ASI-16": "ASI",
}
GROUPS_ORDER = [
    "Irradiance",
    "Decomposed Irradiance",
    "Lagged 10 min GHI",
    "Lagged 10 min Decomposed Irradiance",
    "Time of Day",
    "Trig Time of Day",
    "Time of Year",
    "Trig Time of Year",
    "Time Milestones",
    "Trig Time Milestones",
    "Clear Sky",
    "Prev Hour Stats",
    "Meteorological Measurements",
    "ASI-16",
]
RESPONSE_ORDER = ["GHI", "CSI GHI", "cs_dev t ghi", "Delta GHI", "Delta CSI GHI"]

PAST_FEATURES = DATA_COLS[1:]
FUTURE_FEATURES = [
    "Future clearsky ghi",
    "Future clearsky dni",
    "Future clearsky dhi",
    "Future Solar Eclipse Shading",
    "Future zenith",
    "Future elevation",
    "Future azimuth",
]  # clear sky values are known ahead of time
SCALAR_RESPONSES = ["GHI", "CSI GHI", "cs_dev t ghi"]
RELATIVE_RESPONSES = ["Delta GHI", "Delta CSI GHI"]
GROUP_FULL_NAMES = {}
RESPONSE_FULL_NAMES = {
    "GHI": "Irradiance",
    "Delta CSI GHI": "Change in Clear Sky Index",
    "Delta GHI": "Change in Irradiance",
    "cs_dev t ghi": "Deviation from Clear Sky",
    "CSI GHI": "Clear Sky Index",
}
TIME_REPRESENTATION_ABBREVIATIONS = {
    "Time Milestones": "TM",
    "Time of Year": "ToY",
    "Time of Day": "ToD",
    "Trig Time Milestones": r"$\angle$ TM",
    "Trig Time of Year": r"$\angle$ ToY",
    "Trig Time of Day": r"$\angle$ ToD",
}
TARGET_REPRESENTATION_ABBREVIATIONS = {
    "Irradiance": "GHI",
    "Clear Sky Index": "CSI",
    "Deviation from Clear Sky": "CS Dev.",
    "Change in Irradiance": r"$\Delta$ GHI",
    "Change in Clear Sky Index": r"$\Delta$ CSI",
}
