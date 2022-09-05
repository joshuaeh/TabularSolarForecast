"""
Preprocess later, download now. for each day since July 1, 2004, try to grab the TSI, ASI and BMS values

Will start at the first date the TSI-880 is available and scrape available data for the TSI-880, as well
as meteorological data. After September 26, 2017 it will also scrape the ASI-16 data
"""

#TODO make a master df or other data storage to determine which dates have been scraped. Maybe a start/stop date is sufficient for each data type
#TODO Do our variables include wind speed, direction, and pressure?
#TODO script currently requires CWD to be preprecessing. Make references universal


# imports
import os
from datetime import date, datetime, timedelta
import wget
import ssl
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context

################ Declarations ##################
# Initial start date 2004,7,1

start_date = datetime(2004,7,1)
day_delta = timedelta(days=1)
end_date = datetime(2022,7,10)

N_JOBS = 8  # Edit with how many processses

################## Functions ########################

def download_date_specified(download_date):
    try:
        # download
        Y = download_date.year
        YYYYMMDD = download_date.strftime("%Y%m%d")
        print(YYYYMMDD)

        ## TSI - Total Sky Imager
        nrel_tsi_url = f"https://midcdmz.nrel.gov/tsi/SRRL/{Y}/{YYYYMMDD}.zip"
        tsi_path = f"../data/NREL/TSI/{YYYYMMDD}.zip"
        if not (os.path.exists(tsi_path) or os.path.isdir(f"../data/NREL/TSI/{YYYYMMDD}/")):
            try:
                wget.download(nrel_tsi_url, out=tsi_path)
            except Exception as e:
                print(f"error {e} with TSI download: {YYYYMMDD}")
                pass

        ## BMS - Baseline Measurement System
        nrel_bms_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=BMS&begin={YYYYMMDD}&end={YYYYMMDD}"
        bms_path = f"../data/NREL/BMS/{YYYYMMDD}.csv"
        if not os.path.isfile(bms_path):
            wget.download(nrel_bms_url, out=bms_path)

        ## irrsp - Rotating Shadowband Pyranometer V2 - low cost option for global and diffuse irradiance
        nrel_irrsp_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=IRRSP&begin={YYYYMMDD}&end={YYYYMMDD}"
        irrsp_path = f"../data/NREL/IRRSP/{YYYYMMDD}.csv"
        if not os.path.isfile(irrsp_path):
            wget.download(nrel_irrsp_url, out=irrsp_path)

        ## SSIM - Direct Normal and diffuse Horizontal irradiance measurement 
        if download_date >= datetime(2016,9,1):
            nrel_ssim_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=SSIM&begin={YYYYMMDD}&end={YYYYMMDD}"
            ssim_path = f"../data/NREL/SSIM/{YYYYMMDD}.csv"
            if not os.path.isfile(ssim_path):
                try:
                    wget.download(nrel_ssim_url, out=ssim_path)
                except Exception as e:
                    print(e)
                    pass

        ## SSIMG - Horizontal irradiance measurements every 20 sec from April 2021
        if download_date >= datetime(2016,9,1):
            nrel_ssimg_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=SSIMG&begin={YYYYMMDD}&end={YYYYMMDD}"
            ssimg_path = f"../data/NREL/SSIMG/{YYYYMMDD}.csv"
            if not os.path.isfile(ssimg_path):
                try:
                    wget.download(nrel_ssimg_url, out=ssimg_path)
                except Exception as e:
                    print(e)
                    pass
        ## PVWSSRL - Precipitable Water Vapor
        if download_date >= datetime(2012,6,13):
            nrel_pwvssrl_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=PWVSSRL&begin={YYYYMMDD}&end={YYYYMMDD}"
            pwvssrl_path = f"../data/NREL/PWVSSRL/{YYYYMMDD}.csv"
            if not os.path.isfile(pwvssrl_path):
                try:
                    wget.download(nrel_pwvssrl_url, out=pwvssrl_path)
                except Exception as e:
                    print(e)
                    pass

        ## ASI - EKO ASI-16 Skycamera
        if download_date > datetime(2017,9,25):
            nrel_asi_url = f"https://midcdmz.nrel.gov/tsi/SRRLASI/{Y}/{YYYYMMDD}.zip"
            asi_path = f"../data/NREL/ASI/{YYYYMMDD}.zip"
            if not(os.path.exists(asi_path) or os.path.isdir(f"../data/NREL/BMS/{YYYYMMDD}/")):
                try:
                    wget.download(nrel_asi_url, out=asi_path)
                except Exception as e:
                    print(f"error {e} with ASI {YYYYMMDD}")
                    pass

        ## AODSSRL1E - ESR Aerosol Optical Depth
        if download_date >= datetime(2013,3,1):
            nrel_AODSSRL1E_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=AODSSRL1EL&begin={YYYYMMDD}&end={YYYYMMDD}"
            aod_path = f"../data/NREL/AODSSRL1E/{YYYYMMDD}.csv"
            if not os.path.isfile(aod_path):
                try:
                    wget.download(nrel_AODSSRL1E_url, out=aod_path)
                except Exception as e:
                    print(e)
                    pass

        ## RAZON - GHI, DNI, DHI from Feb 2017
        if download_date >= datetime(2017,2,1):
            nrel_RAZON_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=RAZONL&begin={YYYYMMDD}&end={YYYYMMDD}"
            razon_path = f"../data/NREL/RAZON/{YYYYMMDD}.csv"
            if not os.path.isfile(razon_path):
                try:
                    wget.download(nrel_RAZON_url, out=razon_path)
                except Exception as e:
                    print(e)
                    pass

    except Exception as e:
        print(e)
        pass
    
    return

################# Scripts #################
datetime_list = []

dt = start_date

while dt <= end_date:
    datetime_list.append(dt)
    dt += timedelta(days=1)

# parallelize downloads
Parallel(n_jobs=N_JOBS, verbose=5)(delayed(download_date_specified)(dt) for dt in datetime_list)
