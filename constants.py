import os

# TODO make API tokens and other specifications in another file called "auth" or something similar that is not commited
# TODO make file structure standardized
# TODO I'd like to refactor to either specify what constants are imported from constants.py or use constants.CONSTANT syntax to assist in debugging

# Directories
# HOME = os.environ["HOME"]
# WORK = os.environ["WORK"]
# DATA_PATH = os.path.join("/work", "08940", "joshuaeh", "shared", "joint_data.csv")
# SOLARPROPHET_PATH = os.path.join(WORK, "projects", "SolarProphet")
SOLARPROPHET_PATH = os.getcwd()
JOINT_DATA_H5_PATH = os.path.join(SOLARPROPHET_PATH, "data", "joint_data.h5")
JOINT_DATA_CSV_PATH = os.path.join(SOLARPROPHET_PATH, "data", "joint_data.csv")
DATA_PATH = JOINT_DATA_CSV_PATH
# Neptune connection
NEPTUNE_PROJECT = "HorizonPSE/SP-tab"
NEPTUNE_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNTBhNmQ1Ny04MjAzLTQ2ZjUtODA2MC0yNDllNWUxOWE2ZjkifQ=="

CSV_COLS = ["dateTime",
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

FEATURE_GROUPS = {
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
# TODO use .ANONYMOUS as an argument when we are debugging
# TODO save model weights, go through Neptune best practices
RESPONSE_FEATURES = {  # to compare in a standardized manner, we should always include the value being predicted as a feature. We will make sure to do that here
    # TODO figure how to not include a feature twice
    "GHI" : "GHI",
    "CSI" : "CSI",
    "CS DEV" : "cs_dev t ghi",
    "Delta GHI" : "GHI",
    "Delta CSI" : "CSI"
}
GROUP_ABBREVIATIONS = {
    "Irradiance" : "GHI",  # I like this one the least
    "Decomposed Irradiance" : "DI",  # This should in reality include I implicitly
    "Lagged 10 min GHI" : "L10GHI",
    "Lagged 10 min Decomposed Irradiance" : "L10DI",
    "Time of Day" : "TOD",
    "Trig Time of Day" : "TTOD",
    "Time of Year" : "TOY",
    "Trig Time of Year" : "TTOY",
    "Time Milestones" : "TM",
    "Trig Time Milestones" : "TTM",
    "Clear Sky" : "CS",
    "Future Clear Sky" : "FCS",
    "Prev Hour Stats" : "L60Stats",
    "Meteorological Measurements" : "MET",
    "ASI-16" : "ASI"
}
GROUPS_ORDER = ["Irradiance", "Decomposed Irradiance", "Lagged 10 min GHI", "Lagged 10 min Decomposed Irradiance", "Time of Day", "Trig Time of Day", "Time of Year",
    "Trig Time of Year", "Time Milestones", "Trig Time Milestones", "Clear Sky", "Future Clear Sky", "Prev Hour Stats", "Meteorological Measurements", "ASI-16"]
RESPONSE_ORDER = ["GHI", "CSI GHI", "cs_dev t ghi", "Delta GHI", "Delta CSI"]