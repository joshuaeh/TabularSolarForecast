"""
Preprocess later, download now. for each day since July 1, 2004, try to grab the TSI, ASI and BMS values
"""
# imports
import os
from datetime import date, datetime, timedelta
import wget


start_date = datetime(2004,7,1)
day_delta = timedelta(days=1)
end_date = date.today()

download_date = start_date

while download_date < end_date:
    # check that folders do not exist

    # download
    Y = download_date.year
    YYYYMMDD = download_date.strftime("%Y%m%d")
    print(YYYYMMDD)

    ## TSI
    nrel_tsi_url = f"https://midcdmz.nrel.gov/tsi/SRRL/{Y}/{YYYYMMDD}.zip"
    tsi_path = "../data/NREL/TSI/{YYYYMMDD}.zip"
    wget.download(nrel_tsi_url, tsi_path)


    ## BMS
    nrel_bms_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site=BMS&begin={YYYYMMDD}&end={YYYYMMDD}"
    bms_path = "../data/NREL/BMS/{YYYYMMDD}.csv"
    wget.download(nrel_bms_url, bms_path)

    ## ASI
    if download_date > datetime(2017,9,25):
        nrel_asi_url = f"https://midcdmz.nrel.gov/tsi/SRRLASI/{Y}/{YYYYMMDD}.zip"
        asi_path = "../data/NREL/BMS/{YYYYMMDD}.csv"
        wget.download(nrel_asi_url, asi_path)