"""Script for downloading images and weather data from NREL SRRL BMS"""
# Author: Joshua Hammond

### Imports ###
import os
import wget
from download_ASI_data_NREL import get_dates, make_dir
import ssl
import pandas as pd
ssl._create_default_https_context = ssl._create_unverified_context
from datetime import datetime, date, timedelta
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

ssl._create_default_https_context = ssl._create_unverified_context

### Declarations ###
start_date = date(2017, 9, 27)
end_date = date(2022, 9, 26)
n_workers = 10

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'NREL')

params = {"2018_2019": {"start_url": "https://midcdmz.nrel.gov/apps/plot.pl?site=BMS&start=20171201&edy=31&emo=12&eyr=2019&zenloc=209&amsloc=211",
                        "end_url": "&time=0&inst=3&inst=57&inst=124&inst=127&inst=131&inst=132&inst=135&inst=139&inst=141&inst=149&inst=154&inst=158&inst=159&inst=160&inst=163&type=data&wrlevel=6&preset=0&first=3&math=0&second=-1&value=0.0&global=-1&direct=-1&diffuse=-1&user=0&axis=1"},
          "2020_2021": {"start_url": "https://midcdmz.nrel.gov/apps/plot.pl?site=BMS&start=20200101&edy=31&emo=12&eyr=2021&zenloc=222&amsloc=224",
                        "end_url": "&time=0&inst=3&inst=62&inst=131&inst=134&inst=138&inst=139&inst=142&inst=146&inst=148&inst=156&inst=161&inst=165&inst=166&inst=167&inst=170&type=data&wrlevel=6&preset=0&first=3&math=0&second=-1&value=0.0&global=-1&direct=-1&diffuse=-1&user=0&axis=1"
                        }
        }
BMS_dates = [  # dates where the BMS data is continuous and can be downloaded in 100 day chunks
    [date(1981,7,15), date(1983,6,30)],
    [date(1984,10,25), date(1985,4,7)],
    [date(1985,4,8), date(1988,2,29)],
    [date(1988,3,14), date(1999,1,11)],
    [date(1999,1,13), date(2000,7,31)],
    [date(2000,8,1), date(2001,10,31)],
    [date(2000,11,1), date(2001,8,3)],
    [date(2001,8,4), date(2003,12,31)],
    [date(2004,1,1), date(2005,12,31)],
    [date(2006,1,1), date(2008,8,31)],
    [date(2008,9,1), date(2012,3,31)],
    [date(2012,6,1), date(2014,12,31)],
    [date(2015,1,1), date(2017,11,30)],
    [date(2017,12,1), date(2019,12,31)],
    [date(2020,1,1), date.today() - timedelta(days=2)]
]

### Functions ###

def make_dir(dir):
    """
    Make new directory if it doesn't exist
    """
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
        
def get_dates(year, month):
    """
    Get all dates in a given month and year
    """
    if month == 12:
        num_days = (date(year+1, 1, 1) - date(year, month, 1)).days
    else:
        num_days = (date(year, month+1, 1) - date(year, month, 1)).days
    d1 = date(year, month, 1)
    d2 = date(year, month, num_days)
    delta = d2 - d1
    return [(d1 + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]

def download_data(year, month_start=1, month_end=12):
    """
    Download the data for a given month range of an year (default = full year)
    """
    make_dir(data_dir) # make /data/NREL folder if it doesn't exist
    for month in range(month_start, month_end+1):
        dates = get_dates(year, month)
        for date in dates:
            try:
                
                day = int(date[-2:])
                # create mid_url
                if year in [2018,2019,2020,2021]:
                    mid_url = "&year=%s&month=%s&day=%s&endyear=%s&endmonth=%s&endday=%s" % (year,month,day, year,month,day)
                else:
                    raise ValueError("Year can only be from 2018, 2019, 2020 or 2021.")

                # create full_url
                if year in [2018,2019]:
                    full_url = params["2018_2019"]["start_url"] + mid_url + params["2018_2019"]["end_url"]
                elif year in [2020,2021]:
                    full_url = params["2020_2021"]["start_url"] + mid_url + params["2020_2021"]["end_url"]

                # download data files
                wget.download(full_url, out=data_dir)
                os.rename(os.path.join(data_dir, 'plot.pl'), os.path.join(data_dir, str(date)+'.csv'))
            except:
                pass
def download_date(date):
    """
    """
    # download
    attempts = 0
    success = False
    while attempts < 3 and not success:
        try:
            download_url = nrel_bms_query_url(date, date)
            downloaded_filename = wget.download(download_url, out=data_dir)
            source_path = os.path.join(data_dir, downloaded_filename)
            target_path = os.path.join(data_dir, f"{date.strftime('%Y%m%d')}.csv")
            os.rename(source_path, target_path)
            success = True
        except Exception as e:
            attempts += 1
            print(f"Error downloading {date.strftime('%Y%m%d')}: {e}")
            pass
    return target_path

def download_data_parallel(start_date, end_date, n_workers=None):
    """
    
    """
    pbar = tqdm(range((end_date - start_date).days + 1))
    pbar.set_description("Determining dates to download")
    # TODO create cache file that stores successful dates so that they don't have to be downloaded again, then check before adding
    dates = [start_date + timedelta(days=x) for x in pbar]
    pbar.close()
    if n_workers:
        with Pool(n_workers) as p:
            r = list(tqdm(p.imap(download_date, dates)))
    else:
        for date in tqdm(dates):
            download_date(date)


def find_bms_date_pairs(start_date, end_date):
    """
    
    """
    bms_date_pairs = []
    while start_date < end_date:
        for BMS_date in BMS_dates:
            if start_date >= BMS_date[0] and start_date <= BMS_date[1]:
                if end_date <= BMS_date[1]:
                    bms_date_pairs.append([start_date, end_date])
                    start_date = end_date
                else:
                    bms_date_pairs.append([start_date, BMS_date[1]])
                    start_date = BMS_date[1] + timedelta(days=1)
                break
    return bms_date_pairs

def nrel_bms_query_url(start_date, end_date):
    """
    """
    site = "BMS"
    begin = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    api_url = f"https://midcdmz.nrel.gov/apps/data_api.pl?site={site}&begin={begin}&end={end}"
    return api_url

def download_continuous_bms_data_chunk(start_date, end_date):
    """
    """
    # check that chunk is continuous
    make_dir(data_dir) # make /data/NREL folder if it doesn't exist
    for BMS_date_pair in BMS_dates:
        if start_date <= BMS_date_pair[1] and end_date >= BMS_date_pair[1]:  # check if span is over data discontinuity
            ValueError("The start and end date must be within the same BMS data chunk.\n" +
                       f"There is a discontinuity in the data at {BMS_date_pair[1].strftime('%Y%m%d')}.\n"+
                       f"Please split the query into two parts.\n" +
                       f"Maybe: {start_date.strftime('%Y%m%d')} to {BMS_date_pair[1].strftime('%Y%m%d')} and {BMS_date_pair[1].strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    # download
    download_url = nrel_bms_query_url(start_date, end_date)
    downloaded_filename = wget.download(download_url, out=data_dir)
    source_path = os.path.join(data_dir, downloaded_filename)
    target_path = os.path.join(data_dir, f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv")
    os.rename(source_path, target_path)
    return target_path

def split_downloaded_csv(csv_path):
    """
    """
    # read csv
    df = pd.read_csv(csv_path, header=0)
    split_dates = []
    # split into two dataframes
    df_split = {year_doy:group for year_doy, group in df.groupby(["Year", "DOY"])}
    for k,v in df_split.items():
        group_date = datetime.strptime(f"{k[0]}-{k[1]}", "%Y-%j")
        v.to_csv(os.path.join(data_dir, f"{group_date.strftime('%Y%m%d')}.csv"), index=False)
        split_dates.append(group_date.strftime('%Y%m%d'))
    return split_dates

def download_data_range(start_date, end_date):
    """
    
    """
    continuous_date_pairs = find_bms_date_pairs(start_date, end_date)
    downloaded_files = []
    pbar = tqdm(continuous_date_pairs)
    for continuous_date_pair in pbar:
        try:
            pbar.set_description(f"Downloading {continuous_date_pair[0].strftime('%Y%m%d')} to {continuous_date_pair[1].strftime('%Y%m%d')}", refresh=True)
            downloaded_files.append(download_continuous_bms_data_chunk(continuous_date_pair[0], continuous_date_pair[1]))
        except Exception as e:
            print(e)
            print(f"Failed to download data for {continuous_date_pair[0].strftime('%Y%m%d')} to {continuous_date_pair[1].strftime('%Y%m%d')}")
            pass
    pbar.close()
    return downloaded_files

### Main ###

# Download weather data: Uncomment based on your need
if __name__ == "__main__":
    download_data_parallel(start_date, end_date, n_workers=n_workers) 