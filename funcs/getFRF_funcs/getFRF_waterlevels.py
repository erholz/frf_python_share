import numpy as np
from netCDF4 import Dataset
import pandas as pd
import requests

def getthredds_waterlevels(full_path):

    """
    :param full_path: consists of floc (path after .../thredds/dodsC/frf/) + filename
    :return: waterlevel_noaa, time_noaa
    """


    ## Get the date information from the input file name
    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"

    ## Water Level Dataset
    ds = Dataset(frf_base + full_path, "r")
    waterlevel_noaa = ds.variables["waterLevel"][:]
    time_noaa = np.asarray(ds.variables["time"][:])

    is_waterlevel_masked = np.ma.is_masked(waterlevel_noaa)
    if is_waterlevel_masked:
        waterlevel_noaa = waterlevel_noaa.filled(fill_value=np.NaN)


    return waterlevel_noaa, time_noaa


def getlocal_waterlevels(full_path):

    """
    :param :param: full_path - consists of floc (full path to file) + filename
    :return: waterlevel_noaa, time_noaa
    """

    ## Water Level Dataset
    ds = Dataset(full_path, "r")
    waterlevel_noaa = ds.variables["waterLevel"][:]
    time_noaa = np.asarray(ds.variables["time"][:])

    is_waterlevel_masked = np.ma.is_masked(waterlevel_noaa)
    if is_waterlevel_masked:
        waterlevel_noaa = waterlevel_noaa.filled(fill_value=np.NaN)

    return waterlevel_noaa, time_noaa



def getnoaatidewithpred_func(gauge, datum, start_year, end_year):
    wl = []
    time = []
    pred = []
    matlabTimePred = []
    datetimePred = []

    for yr in range(start_year, end_year + 1):
        print(yr)

        # NOAA API URLs for water levels and predictions
        website = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={yr}0101&end_date={yr}1231&station={gauge}&product=hourly_height&datum={datum}&time_zone=gmt&units=metric&format=csv'
        website2 = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={yr}0101&end_date={yr}1231&station={gauge}&product=predictions&datum={datum}&time_zone=gmt&units=metric&format=csv'

        try:
            # Download hourly height data
            response = requests.get(website, timeout=15)
            with open('tempwaves.csv', 'w') as f:
                f.write(response.text)
            data2 = pd.read_csv('tempwaves.csv')

            # Parse datetime
            data2['datetime'] = pd.to_datetime(data2['Date Time'], format='%Y-%m-%d %H:%M')
            wl.extend(data2[' Water Level'].values)
            time.extend(data2['datetime'].apply(lambda x: x.toordinal() + x.hour / 24 + x.minute / 1440).values)

            # Download predictions data
            response2 = requests.get(website2, timeout=15)
            with open('tempwaves2.csv', 'w') as f:
                f.write(response2.text)
            data = pd.read_csv('tempwaves2.csv')

            # Parse datetime for predictions
            data['datetime'] = pd.to_datetime(data['Date Time'], format='%Y-%m-%d %H:%M')
            pred.extend(data[' Prediction'].values)
            matlabTimePred.extend(data['datetime'].apply(lambda x: x.toordinal() + x.hour / 24 + x.minute / 1440).values)
            datetimePred.extend(data['datetime'].values)

        except Exception as e:
            print(f"Error for year {yr}: {e}")
            continue

    # Output data as a dictionary
    tideout = {
        'wltime': np.array(time),
        'wl': np.array(wl, dtype=float),
        'predtimeMatlabTime': np.array(matlabTimePred),
        'predtimeDateTime': np.array(datetimePred),
        'pred': np.array(pred, dtype=float)
    }

    return tideout