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

ssl._create_default_https_context = ssl._create_unverified_context

################ Declarations ##################
# Initial start date 2004,7,1

start_date = datetime(2006,6,9)
day_delta = timedelta(days=1)
end_date = datetime(2022,7,10)

################## Loop ########################

download_date = start_date

while download_date < end_date:
    try:
    # download
    Y = download_date.year
    YYYYMMDD = download_date.strftime("%Y%m%d")
    print(YYYYMMDD)

    ## TSI
    nrel_tsi_url = f"https://midcdmz.nrel.gov/tsi/SRRL/{Y}/{YYYYMMDD}.zip"
    tsi_path = f"../data/NREL/TSI/{YYYYMMDD}.zip"
    if not (os.path.exists(tsi_path) or os.path.isdir(f"../data/NREL/TSI/{YYYYMMDD}/")):
        try:
            wget.download(nrel_tsi_url, out=tsi_path)
        except Exception as e:
            print(f"error {e} with TSI download: {YYYYMMDD}")
            pass

    ## BMS
    nrel_bms_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=BMS&begin={YYYYMMDD}&end={YYYYMMDD}"
    bms_path = f"../data/NREL/BMS/{YYYYMMDD}.csv"
    if not os.path.isfile(bms_path):
        wget.download(nrel_bms_url, out=bms_path)

    ## irrsp
    nrel_irrsp_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=IRRSP&begin={YYYYMMDD}&end={YYYYMMDD}"
    irrsp_path = f"../data/NREL/IRRSP/{YYYYMMDD}.csv"
    if not os.path.isfile(irrsp_path):
        wget.download(nrel_irrsp_url, out=irrsp_path)

    ## ASI
    if download_date > datetime(2017,9,25):
        nrel_asi_url = f"https://midcdmz.nrel.gov/tsi/SRRLASI/{Y}/{YYYYMMDD}.zip"
        asi_path = f"../data/NREL/BMS/{YYYYMMDD}.csv"
        if not(os.path.exists(asi_path) or os.path.isdir(f"../data/NREL/BMS/{YYYYMMDD}/")):
            try:
                wget.download(nrel_asi_url, out=asi_path)
            except:
                print(f"error {e} with ASI {YYYYMMDD}")
                pass

    download_date += day_delta