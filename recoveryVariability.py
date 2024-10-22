import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from run_lidarcollect import *
from run_hydrocollect import *
from funcs.create_contours import *
from funcs.lidar_check import *
from funcs.calculate_beachvol import *
from funcs.lidar_fillgaps import *
from run_makeplots import *
import pickle
import os
# DEFINE WHERE FRF DATA FILES ARE LOCATED
# local_base = 'D:/FRF_data/'
local_base = '/volumes/macDrive/FRF_data/'

# DEFINE TIME PERIOD OF INTEREST
time_beg = '2016-01-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2024-10-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4

# DEFINE CONTOUR ELEVATIONS OF INTEREST
cont_elev = np.arange(-1.75,4.00,0.25)    # <<< MUST BE POSITIVELY INCREASING

# DEFINE NUMBER OF PROFILES TO PLOT
num_profs_plot = 15

# DEFINE SUBDIR WITH LIDAR FILES
lidarfloc = local_base + 'dune_lidar/lidar_transect/'
lidarext = 'nc'  # << change not recommended; defines file type to look for

# DEFINE SUBDIR WITH NOAA WATERLEVEL FILES
noaawlfloc = local_base + 'waterlevel/'
noaawlext = 'nc'  # << change not recommended; defines file type to look for

# DEFINE SUBDIR WITH LIDAR HYDRO FILES
lidarhydrofloc = local_base + 'waves_lidar/lidar_hydro/'
lidarhydroext = 'nc'  # << change not recommended; defines file type to look for




# -------------------- BEGIN RUN_CODE.PY --------------------

# convert period of interest to datenum
time_format = '%Y-%m-%dT%H:%M:%S'
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)
# Save timing variables
with open('timeinfo.pickle','wb') as file:
    pickle.dump([tzinfo,time_format,time_beg,time_end,epoch_beg,epoch_end,TOI_duration], file)

# run file run_lidarcollect.py
lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext)

# Remove weird data (first order filtering)
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
lidarelev[tmpii] = np.nan

# run file create_contours.py
elev_input = lidarelev
cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)


lidarTime = lidartime
lidarProfiles = lidarelev
lidarContours = cont_ts
dts = [dt.datetime.utcfromtimestamp(ts) for ts in lidarTime]

del lidartime
del lidarelev
del cont_ts
del cmean
del cstd

import pickle
clusterPickle = 'alongshoreAverages.pickle'
with open(clusterPickle, "rb") as input_file:
    inputCusps = pickle.load(input_file)
profileTimeAlongshore = inputCusps['profileTime']
alongshoreAverageTime = [dt.datetime.utcfromtimestamp(ts) for ts in profileTimeAlongshore]
# lidar_yFRFAlongshore = inputCusps['lidar_yFRF']
lidar_xFRFAlongshore = inputCusps['lidar_xFRF']
ysCAlongshore = inputCusps['ysC']
alongshoreAverage = inputCusps['alongshoreAverage']
alongshoreAverage = np.stack(alongshoreAverage,axis=0)
# alongshoreStd = inputCusps['alongshoreStd']
# cont_elev_alongshore = inputCusps['cont_elev']
# cont_ts_alongshore = inputCusps['cont_ts']
# cmean_alongshore = inputCusps['cmean']
# cstd_alongshore = inputCusps['cstd']

del profileTimeAlongshore


cuspsPickle = 'cuspTimes.pickle'
with open(cuspsPickle, "rb") as input_file:
    inputTime = pickle.load(input_file)
timeCusps = inputTime['timeCusps']

cuspsTimes = [dt.datetime.utcfromtimestamp(ts) for ts in timeCusps]

del timeCusps





# s = 25000
# plt.figure()
# for hh in range(100):
#     plt.plot(lidar_xFRF,lidarProfiles[s+hh])
#



def rmse(predictions, targets):
    differences = predictions - targets                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val                                           #get the ^





import numba
import numpy as np

@numba.njit()
def interpolate_with_max_gap(orig_x,
                             orig_y,
                             target_x,
                             max_gap=np.inf,
                             orig_x_is_sorted=False,
                             target_x_is_sorted=False):
    """
    Interpolate data linearly with maximum gap. If there is
    larger gap in data than `max_gap`, the gap will be filled
    with np.nan.

    The input values should not contain NaNs.

    Parameters
    ---------
    orig_x: np.array
        The input x-data
    orig_y: np.array
        The input y-data
    target_x: np.array
        The output x-data; the data points in x-axis that
        you want the interpolation results from.
    max_gap: float
        The maximum allowable gap in `orig_x` inside which
        interpolation is still performed. Gaps larger than
        this will be filled with np.nan in the output `target_y`.
    orig_x_is_sorted: boolean, default: False
        If True, the input data `orig_x` is assumed to be monotonically
        increasing. Some performance gain if you supply sorted input data.
    target_x_is_sorted: boolean, default: False
        If True, the input data `target_x` is assumed to be
        monotonically increasing. Some performance gain if you supply
        sorted input data.

    Returns
    ------
    target_y: np.array
        The interpolation results.
    """
    if not orig_x_is_sorted:
        # Sort to be monotonous wrt. input x-variable.
        idx = orig_x.argsort()
        orig_x = orig_x[idx]
        orig_y = orig_y[idx]

    if not target_x_is_sorted:
        target_idx = target_x.argsort()
        # Needed for sorting back the data.
        target_idx_for_reverse = target_idx.argsort()
        target_x = target_x[target_idx]

    target_y = np.empty(target_x.size)
    idx_orig = 0
    orig_gone_through = False

    for idx_target, x_new in enumerate(target_x):

        # Grow idx_orig if needed.
        while not orig_gone_through:

            if idx_orig + 1 >= len(orig_x):
                # Already consumed the orig_x; no more data
                # so we would need to extrapolate
                orig_gone_through = True
            elif x_new > orig_x[idx_orig + 1]:
                idx_orig += 1
            else:
                # x_new <= x2
                break

        if orig_gone_through:
            target_y[idx_target] = np.nan
            continue

        x1 = orig_x[idx_orig]
        y1 = orig_y[idx_orig]
        x2 = orig_x[idx_orig + 1]
        y2 = orig_y[idx_orig + 1]

        if x_new < x1:
            # would need to extrapolate to left
            target_y[idx_target] = np.nan
            continue

        delta_x = x2 - x1

        if delta_x > max_gap:
            target_y[idx_target] = np.nan
            continue

        delta_y = y2 - y1

        if delta_x == 0:
            target_y[idx_target] = np.nan
            continue

        k = delta_y / delta_x

        delta_x_new = x_new - x1
        delta_y_new = k * delta_x_new
        y_new = y1 + delta_y_new

        target_y[idx_target] = y_new

    if not target_x_is_sorted:
        return target_y[target_idx_for_reverse]
    return target_y



from scipy.io.matlab.mio5_params import mat_struct
import scipy.io as sio

# def ReadMatfile(p_mfile):
#     'Parse .mat file to nested python dictionaries'
#
#     def RecursiveMatExplorer(mstruct_data):
#         # Recursive function to extrat mat_struct nested contents
#
#         if isinstance(mstruct_data, mat_struct):
#             # mstruct_data is a matlab structure object, go deeper
#             d_rc = {}
#             for fn in mstruct_data._fieldnames:
#                 d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
#             return d_rc
#
#         else:
#             # mstruct_data is a numpy.ndarray, return value
#             return mstruct_data
#
#     # base matlab data will be in a dict
#     mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
#     mdata_keys = [x for x in mdata.keys() if x not in
#                   ['__header__','__version__','__globals__']]
#
#     # use recursive function
#     dout = {}
#     for k in mdata_keys:
#         dout[k] = RecursiveMatExplorer(mdata[k])
#     return dout
#
# wls = ReadMatfile('/Users/dylananderson/Documents/data/noaaWaterLevels/duck/noaa8651370.mat')
# tide = wls['dailyData']['tide']
# wl = wls['dailyData']['wl']
# seasonal = wls['dailyData']['seasonal']
# msl = wls['dailyData']['msl']
# mmsla = wls['dailyData']['mmsla']
# dsla = wls['dailyData']['dsla']
# ss = wls['dailyData']['ss']
# timeHourly = wls['dailyData']['hourlyDateVec']
# timeMonthly = wls['dailyData']['monthDateVec']
# mmslaMonth = wls['dailyData']['mmsla_month']


import numpy as np
import pandas as pd
import requests
from datetime import datetime


def download_noaa_tides_dylanWithPred(gauge, datum, start_year, end_year):
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



gauge = '8651370'
datum = 'MSL'
start_year = 2016#1978
end_year = 2024
tideout = download_noaa_tides_dylanWithPred(gauge, datum, start_year, end_year)

dat = tideout['wl']
time = tideout['wltime']
time_tide_predUTC = np.asarray([(dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's') for dt64 in tideout['predtimeDateTime'][::5]])
time_tide_pred = np.asarray([datetime.utcfromtimestamp(utc) for utc in time_tide_predUTC])

tide_pred = tideout['pred'][::5]

from scipy.signal import find_peaks
peaks = find_peaks(tide_pred)

highTideTimes = time_tide_pred[peaks[0]]
highTideUTCtime = time_tide_predUTC[peaks[0]]
highTides = tide_pred[peaks[0]]


del tideout


import datetime as dt
from dateutil.relativedelta import relativedelta

st = dt.datetime(2016,1,1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2024,10,1)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)#.strftime('%Y-%m-%d'))
    st += step

# st = dt.datetime(2016,1,1)
# # end = dt.datetime(2021,12,31)
# end = dt.datetime(2024,10,1)
# step = relativedelta(days=2)
# twoDayTime = []
# while st < end:
#     twoDayTime.append(st)#.strftime('%Y-%m-%d'))
#     st += step

def find_files_local(floc,ext_in):
    full_path = floc
    ids = []
    for file in os.listdir(full_path):
        if file.endswith(ext_in):
            if not file.startswith('.'):
                ids.append(file)
    return ids
from datetime import datetime
#
# #start with NOAA water level files
# floc = noaawlfloc
# ext = noaawlext
# fname_in_range = find_files_local(floc,ext)#find_files_in_range(floc,ext,epoch_beg,epoch_end, tzinfo)
# wltime_noaa = []
# wltime_datetime = []
# wl_noaa = []
# for fname_ii in fname_in_range:
#     print('reading... ' + fname_ii)
#     full_path = floc + fname_ii
#     waterlevel_noaa, time_noaa = getlocal_waterlevels(full_path)
#     convertTime = np.asarray([datetime.utcfromtimestamp(st) for st in time_noaa])
#     wltime_datetime = np.append(wltime_datetime,convertTime)
#     wltime_noaa = np.append(wltime_noaa, time_noaa)
#     wl_noaa = np.append(wl_noaa, waterlevel_noaa)
#

#
# from dateutil.relativedelta import relativedelta
# st = dt.datetime(2016, 1, 1,5,30,0)
# end = dt.datetime(2024,10,1)
# step = relativedelta(hours=12.41667)
# wlTimes = []
# wlDateTimes = []
# while st < end:
#     wlDateTimes.append(st) #.strftime('%Y-%m-%d')
#     wlTimes.append((st - datetime(1970,1,1)).total_seconds())
#     st += step
#
# highTideIndices = []
# highTideTimes = []
# highTides = []
# highTideUTCtime = []
# morphTime = []
# morphUTCtime = []
# for qq in range(len(wlDateTimes)-1):
#     # inder = np.where((wlTimes[qq] > wltime_noaa) & (wlTimes[qq] < wltime_noaa))
#     # inder = np.where((wltime_datetime >= wlDateTimes[qq]) & (wltime_datetime <= wlDateTimes[qq+1]))
#     inder = np.where((time_tide_pred >= wlDateTimes[qq]) & (time_tide_pred <= wlDateTimes[qq+1]))
#
#     if len(inder[0]) == 0:
#         print('gap in the record for {}'.format(wlDateTimes[qq]))
#     else:
#         temp = tide_pred[inder]
#         temp = np.delete(temp,np.where(np.isnan(temp)))
#         if len(temp) > 0:
#             subsetWLind = np.nanargmax(tide_pred[inder])
#             subsetTime = time_tide_pred[inder]
#             subsetUTC = time_tide_predUTC[inder]#np.asarray([dt.datetime.utcfromtimestamp(ts) for ts in subsetTime])
#             highTideUTCtime.append(subsetUTC[subsetWLind])
#             highTideTimes.append(subsetTime[subsetWLind])
#             highTides.append(np.nanmax(tide_pred[inder]))
#             morphTime.append(wlDateTimes[qq])
#             morphUTCtime.append(wlTimes[qq])




# dailyAverage = np.nan * np.ones((len(dayTime),len(lidar_xFRF)))
# dailyStd = np.nan * np.ones((len(dayTime),len(lidar_xFRF)))
# dailyAverageWithData = []
# dailyStdWithData = []
# dailyTimeWithData = []
# for qq in range(len(dayTime)-1):
#     inder = np.where((np.asarray(dts)>=np.asarray(dayTime)[qq]) & (np.asarray(dts) <=np.asarray(dayTime)[qq+1]))
#     if len(inder[0])>0:
#         dailyAverage[qq,:] = np.nanmean(lidarProfiles[inder[0],:],axis=0)
#         dailyStd[qq,:] = np.nanstd(lidarProfiles[inder[0],:],axis=0)
#         dailyAverageWithData.append(np.nanmean(lidarProfiles[inder[0],:],axis=0))
#         dailyStdWithData.append(np.nanstd(lidarProfiles[inder[0],:],axis=0))
#         dailyTimeWithData.append(dayTime[qq])
# dailyTimeWithData = np.asarray(dailyTimeWithData)
# dailyAverageWithData = np.asarray(dailyAverageWithData)
# dailyStdWithData = np.asarray(dailyStdWithData)



tidalAverage = np.nan * np.ones((len(highTideTimes),len(lidar_xFRF)))
tidalStd = np.nan * np.ones((len(highTideTimes),len(lidar_xFRF)))
tidalAverageWithData = []
tidalStdWithData = []
tidalTimeWithData = []
tidalTimeUTCWithData = []
tidalTime = []
tidalTimeUTC = []
for qq in range(len(highTideTimes)-1):
    # if np.remainder(qq,100):
    print('done with {} of {}: {}'.format(qq,len(highTideTimes),highTideTimes[qq]))

    # first step - do we have cusps present?
    indexCusps = np.where((np.asarray(cuspsTimes) >= np.asarray(highTideTimes)[qq]) & (np.asarray(cuspsTimes) <= np.asarray(highTideTimes)[qq+1]))
    if len(indexCusps[0]) > 0:
        print('we found a cusp')
        inder = np.where((np.asarray(alongshoreAverageTime) >= np.asarray(highTideTimes)[qq]) & (np.asarray(alongshoreAverageTime) <= np.asarray(highTideTimes)[qq+1]))
        profsOfInterest = alongshoreAverage[inder[0],:]
        m, n = np.shape(profsOfInterest)
        for pp in range(m):
            singleProfOfInterest = profsOfInterest[pp, :]

            nanIndex = np.where(np.isnan(singleProfOfInterest))
            closeNans = np.where(nanIndex[0] > 10)

            profsOfInterest[pp, nanIndex[0][closeNans[0][0]]:] = np.nan * profsOfInterest[pp,
                                                                          nanIndex[0][closeNans[0][0]]:]


        nonNanIndices = ~np.isnan(profsOfInterest)
        goodDataNumbers = np.sum(nonNanIndices, axis=0)
        goodDataInds = np.where(goodDataNumbers < 1)
        profileOfInterestMean = np.nanmean(alongshoreAverage[inder[0], :], axis=0)
        profileOfInterestStd = np.nanstd(alongshoreAverage[inder[0], :], axis=0)
        profileOfInterestMean[goodDataInds] = profileOfInterestMean[goodDataInds] * np.nan
        profileOfInterestStd[goodDataInds] = profileOfInterestStd[goodDataInds] * np.nan
        UNDEF = np.nan

        interpAlongMean = np.interp(lidar_xFRF, lidar_xFRFAlongshore, profileOfInterestMean, left=UNDEF)
        interpAlongStd = np.interp(lidar_xFRF, lidar_xFRFAlongshore, profileOfInterestStd, left=UNDEF)

        tidalAverage[qq, :] = interpAlongMean
        tidalStd[qq, :] = interpAlongStd
        tidalAverageWithData.append(interpAlongMean)
        tidalStdWithData.append(interpAlongStd)
        tidalTimeWithData.append(highTideTimes[qq])
        tidalTimeUTCWithData.append(highTideUTCtime[qq])

    else:

        inder = np.where((np.asarray(dts) >= np.asarray(highTideTimes)[qq]) & (np.asarray(dts) <= np.asarray(highTideTimes)[qq+1]))
        if len(inder[0])>0:
            profsOfInterest = lidarProfiles[inder[0],:]
            m, n = np.shape(profsOfInterest)
            for pp in range(m):
                singleProfOfInterest = profsOfInterest[pp,:]
                nanIndex = np.where(np.isnan(singleProfOfInterest))
                closeNans = np.where(nanIndex[0]>100)
                profsOfInterest[pp,nanIndex[0][closeNans[0][0]]:] = np.nan*profsOfInterest[pp,nanIndex[0][closeNans[0][0]]:]
            nonNanIndices = ~np.isnan(profsOfInterest)
            goodDataNumbers = np.sum(nonNanIndices, axis=0)
            goodDataInds = np.where(goodDataNumbers < 2)
            profileOfInterestMean = np.nanmean(lidarProfiles[inder[0],:],axis=0)
            profileOfInterestStd = np.nanstd(lidarProfiles[inder[0],:],axis=0)
            profileOfInterestMean[goodDataInds] = profileOfInterestMean[goodDataInds]*np.nan
            profileOfInterestStd[goodDataInds] = profileOfInterestStd[goodDataInds]*np.nan
            tidalAverage[qq,:] = profileOfInterestMean
            tidalStd[qq,:] = profileOfInterestStd
            tidalAverageWithData.append(profileOfInterestMean)
            tidalStdWithData.append(profileOfInterestStd)
            tidalTimeWithData.append(highTideTimes[qq])
            tidalTimeUTCWithData.append(highTideUTCtime[qq])

tidalTimeWithData = np.asarray(tidalTimeWithData)
tidalAverageWithData = np.asarray(tidalAverageWithData)
tidalStdWithData = np.asarray(tidalStdWithData)
tidalTimeUTCWithData = np.asarray(tidalTimeUTCWithData)

cont_elev2 = np.arange(-1.,1.00,0.1)    # <<< MUST BE POSITIVELY INCREASING

cont_Tidal, cmean_Tidal, cstd_Tidal = create_contours(tidalAverageWithData,tidalTimeUTCWithData,lidar_xFRF,cont_elev2)

#
# # # plt.figure()
# # # plt.pcolor(dayTime,lidar_xFRF,dailyAverage.T)
plt.figure()
plt.pcolor(highTideTimes,lidar_xFRF,tidalAverage.T)
plt.ylabel('xFRF (m)')
plt.title('Tidal Average Profile densities')
plt.ylim([50,170])
# plt.figure()
# plt.plot(dts,cont_ts[-2,:])


howManyObs = []
# howManyDays = []
howManyTidal= []

for qq in range(len(lidar_xFRF)):
    finder = np.where(np.isnan(elev_input[:,qq]))
    howManyObs.append(len(finder[0]))
    # finder2 = np.where(np.isnan(dailyAverage[:,qq]))
    # howManyDays.append(len(finder2[0]))
    finder3 = np.where(np.isnan(tidalAverage[:,qq]))
    howManyTidal.append(len(finder3[0]))
numOfObs = np.abs(np.asarray(howManyObs)-np.nanmax(np.asarray(howManyObs)))
# numOfDays = np.abs(np.asarray(howManyDays)-np.nanmax(np.asarray(howManyDays)))
numOfTidal = np.abs(np.asarray(howManyTidal)-np.nanmax(np.asarray(howManyTidal)))

# plt.figure()
# p1 = plt.subplot2grid((1,2),(0,0))
# # p2 = plt.subplot2grid((1,2),(0,1))
# p3 = plt.subplot2grid((1,2),(0,1))
# p1.plot(lidar_xFRF,numOfObs)
# # p2.plot(lidar_xFRF,numOfDays)
# p3.plot(lidar_xFRF,numOfTidal)
# p1.set_xlabel('xFRF (m)')
# # p2.set_xlabel('xFRF (m)')
# p3.set_xlabel('xFRF (m)')
# p1.set_ylabel('# of Profiles with data')
# # p2.set_ylabel('# of Days with data')
# p3.set_ylabel('# of Tidal Windows')
# p1.plot([100,100],[-10,50000],'--',color='k')
# # p2.plot([100,100],[-10,2500],'--',color='k')
# p3.plot([100,100],[-10,4200],'--',color='k')
# p1.set_ylim([0,50000])
# # p2.set_ylim([0,2300])
# p3.set_ylim([0,4200])
# p1.set_title('All Observations')
# # p2.set_title('Daily Averaged Profiles')
# p3.set_title('Tidal Averaged Windows')

# asdfg

# d1 = datetime(2017,1,1,0,0,0)
# d2 = datetime(2018,1,1,0,0,0)
# dt = 3600. #seconds has to be a float and not an integer
# profile_num = 960
# dx = 1
# t = np.arange(d1, d2, timedelta(hours=1)).astype(datetime)
# dataloc = ("https://chldata.erdc.dren.mil/thredds/dodsC/frf/geomorphology/elevationTransects/survey/surveyTransects.ncml")
# ncfile = nc.Dataset(dataloc)
# bathy_date= ncfile["date"][:]
# bathy_y = ncfile["profileNumber"][:]
# ifind = np.where((bathy_date>=[tepoch[0]-45*24*60*60*1000]) & (bathy_date<=[tepoch[-1]+45*24*60*60*1000]) & (bathy_y == profile_num))
# bathy_elevation_all= ncfile["elevation"][:]
# bathy_x_all = ncfile["xFRF"][:]
# bathy_elevation= bathy_elevation_all[ifind]
# bathy_x = bathy_x_all[ifind]
# bathy_times = bathy_date[ifind]
# bathy_dates_unique = np.unique(bathy_times)
# # INTERPOLATE BATHY DATA
# offshore_x = 900
# xinterp = np.arange(75, offshore_x + 1, 1)
# zInterpSurvey = np.zeros([len(bathy_dates_unique), len(xinterp)]) * np.nan
# for i in range(len(bathy_dates_unique)):
#     ifind_data = np.where((bathy_times == bathy_dates_unique[i]))
#     if np.size(ifind_data) > 1:
#         z_data_temp = np.array(bathy_elevation[ifind_data])
#         x_data_temp = np.array(bathy_x[ifind_data])
#         isort = np.argsort(x_data_temp)
#         zInterp = interpolate_with_max_gap(x_data_temp[isort], z_data_temp[isort], xinterp, max_gap=10,
#                                            orig_x_is_sorted=False, target_x_is_sorted=False)
#         zInterpSurvey[i, :] = zInterp
# zInterpAll = np.zeros([np.size(t), np.size(xinterp)]) * np.nan
#
# for ix in range(len(xinterp)):
#         tempz = zInterpSurvey[:,ix]
#         inonan = np.where(np.isnan(tempz) == False)
#         if np.size(inonan)>2:
#             zInterpTemp = np.interp(tepoch, np.array(bathy_dates_unique[inonan]), np.array(tempz[inonan]))
#             zInterpAll[:,ix] = zInterpTemp



# c = 1
# for hh in range(150):
#     tempProfileBefore = tidalAverage[c-1,:]
#     tempProfile = tidalAverage[c,:]
#     tempProfileAfter = tidalAverage[c+1,:]
#     tempNanInds = np.where(np.isnan(tempProfile))

#
# plt.figure()
# plt.plot(lidar_xFRF,tempProfileBefore)
# plt.plot(lidar_xFRF,tempProfile,color='k',linewidth=2)
# plt.plot(lidar_xFRF,tempProfileAfter)

# from loess.loess_2d import loess_2d

## LETS EXTEND OUR LIDAR PROFILES WITH DATA WE HAVE FROM EITHER SIDE IN TIME
# AND ALSO CLEAN UP THE PROFILES THAT HAVE SOME WEIRD END EFFECTS
from loess.loess_1d import loess_1d




#
# xIn,yIn = np.meshgrid(np.asarray(highTideUTCtime)[35:45], lidar_xFRF[0:700])
# xOut = np.copy(xIn.flatten())
# yOut = np.copy(yIn.flatten())
# zIn = tidalAverage[35:45,0:700].T
# zIn = zIn.flatten()
# xIn = xIn.flatten()
# yIn = yIn.flatten()
# indNan = np.where(np.isnan(zIn))
# xIn = np.delete(xIn,indNan)
# yIn = np.delete(yIn,indNan)
# zIn = np.delete(zIn,indNan)
#
# zout, wout = loess_2d(xIn,yIn,zIn, xnew=xOut, ynew=yOut, degree=2, frac=0.1,npoints=None, rescale=False, sigz=None)
#
# plt.figure()
# p1 = plt.subplot2grid((2,2),(0,0))
# p1.pcolor(np.asarray(highTideTimes)[35:45],lidar_xFRF,tidalAverage[35:45].T,vmax=6,vmin=-1)
# p1.set_ylabel('xFRF (m)')
# p1.set_ylim([40,110])
# p1.set_title('tidally-averaged profiles')
# p1.tick_params(axis='x', labelrotation=45)
# p2 = plt.subplot2grid((2,2),(0,1))
# p2.pcolor(np.asarray(highTideTimes)[35:46],lidar_xFRF[0:701],zout.reshape((700,10)),vmax=6,vmin=-1)
# p2.set_ylabel('xFRF (m)')
# p2.set_ylim([40,110])
# p2.set_title('loess 2d profiles')
# p2.tick_params(axis='x', labelrotation=45)
# p3 = plt.subplot2grid((2,2),(1,0))
# p3.plot(lidar_xFRF,tidalAverage[40])
# p3.set_xlim([40,110])
# p4 = plt.subplot2grid((2,2),(1,1))
# p4.plot(lidar_xFRF[0:700],zout.reshape((700,10))[:,5])
# p4.set_xlim([40,110])




geomorphdir = '/volumes/anderson/FRF_Data/surveys/'


files = os.listdir(geomorphdir)

files.sort()

subset = files.copy()
subset = subset[1039:]

files_path = [os.path.abspath(geomorphdir) for x in os.listdir(geomorphdir)]

def getBathy(file, lower, upper):
    bathy = Dataset(file)

    xs_bathy = bathy.variables['xFRF'][:]
    ys_bathy = bathy.variables['yFRF'][:]
    zs_bathy = bathy.variables['elevation'][:]
    ts_bathy = bathy.variables['time'][:]
    pr_bathy = bathy.variables['profileNumber'][:]

    zs_bathy = np.ma.masked_where((pr_bathy > upper), zs_bathy)
    ys_bathy = np.ma.masked_where((pr_bathy > upper), ys_bathy)
    xs_bathy = np.ma.masked_where((pr_bathy > upper), xs_bathy)
    pr_bathy = np.ma.masked_where((pr_bathy > upper), pr_bathy)
    ts_bathy = np.ma.masked_where((pr_bathy > upper), ts_bathy)

    zs_bathy = np.ma.masked_where((pr_bathy < lower), zs_bathy)
    ys_bathy = np.ma.masked_where((pr_bathy < lower), ys_bathy)
    xs_bathy = np.ma.masked_where((pr_bathy < lower), xs_bathy)
    pr_bathy = np.ma.masked_where((pr_bathy < lower), pr_bathy)
    ts_bathy = np.ma.masked_where((pr_bathy < lower), ts_bathy)

    output = dict()
    output['x'] = xs_bathy
    output['y'] = ys_bathy
    output['z'] = zs_bathy
    output['pr'] = pr_bathy
    output['t'] = ts_bathy

    return output






extendedTidalAverages = np.copy(tidalAverage)
# extendedTidalAveragesWithData = np.copy(tidalAverageWithData)
# extendedTidalAverages = np.copy(tidalAverage)
# extendedTidalAveragesWithData = np.copy(tidalAverageWithData)
# smoothedTidalAverages = []
bathyX = np.nan*np.copy(tidalAverage)
bathyZ = np.nan*np.copy(tidalAverage)
whichTransect = np.nan*np.ones((len(tidalAverage),))
for i in range(len(subset)):

    file_params = subset[i].split('_')

    ## ### Southern Lines
    data1 = getBathy(os.path.join(geomorphdir, subset[i]), lower=900, upper=920)
    data2 = getBathy(os.path.join(geomorphdir, subset[i]), lower=920, upper=940)
    data3 = getBathy(os.path.join(geomorphdir, subset[i]), lower=950, upper=955)
    data4 = getBathy(os.path.join(geomorphdir, subset[i]), lower=956, upper=965)
    data5 = getBathy(os.path.join(geomorphdir, subset[i]), lower=1000, upper=1015)
    data6 = getBathy(os.path.join(geomorphdir, subset[i]), lower=860, upper=875)

    temp = subset[i].split('_')
    if temp[1] == 'geomorphology':
        temp2 = temp[-1].split('.')
        surveydate = dt.datetime.strptime(temp2[0], '%Y%m%d')
    else:
        surveydate = dt.datetime.strptime(temp[1], '%Y%m%d')

    print('working on {}'.format(surveydate))
    elevs = data1['z']
    cross = data1['x']
    crossind = np.argsort(data1['x'])
    crossS = cross[crossind]
    elevsS = elevs[crossind]

    elevs2 = data2['z']
    cross2 = data2['x']
    crossind2 = np.argsort(data2['x'])
    crossS2 = cross2[crossind2]
    elevsS2 = elevs2[crossind2]

    elevs3 = data3['z']
    cross3 = data3['x']
    crossind3 = np.argsort(data3['x'])
    crossS3 = cross3[crossind3]
    elevsS3 = elevs3[crossind3]

    elevs4 = data4['z']
    cross4 = data4['x']
    crossind4 = np.argsort(data4['x'])
    crossS4 = cross4[crossind4]
    elevsS4 = elevs4[crossind4]

    elevs5 = data5['z']
    cross5 = data5['x']
    crossind5 = np.argsort(data5['x'])
    crossS5 = cross5[crossind5]
    elevsS5 = elevs5[crossind5]

    elevs6 = data6['z']
    cross6 = data6['x']
    crossind6 = np.argsort(data6['x'])
    crossS6 = cross5[crossind6]
    elevsS6 = elevs5[crossind6]

    xSub = np.ma.MaskedArray.filled(crossS, np.nan)
    zSub = np.ma.MaskedArray.filled(elevsS, np.nan)
    xSub2 = np.ma.MaskedArray.filled(crossS2, np.nan)
    zSub2 = np.ma.MaskedArray.filled(elevsS2, np.nan)
    xSub3 = np.ma.MaskedArray.filled(crossS3, np.nan)
    zSub3 = np.ma.MaskedArray.filled(elevsS3, np.nan)
    xSub4 = np.ma.MaskedArray.filled(crossS4, np.nan)
    zSub4 = np.ma.MaskedArray.filled(elevsS4, np.nan)
    xSub5 = np.ma.MaskedArray.filled(crossS5, np.nan)
    zSub5 = np.ma.MaskedArray.filled(elevsS5, np.nan)
    xSub6 = np.ma.MaskedArray.filled(crossS6, np.nan)
    zSub6 = np.ma.MaskedArray.filled(elevsS6, np.nan)

    realValues = ~np.isnan(xSub)
    xSubNew = xSub[~np.isnan(xSub)]
    zSubNew = zSub[~np.isnan(xSub)]

    realValues2 = ~np.isnan(xSub2)
    xSubNew2 = xSub2[~np.isnan(xSub2)]
    zSubNew2 = zSub2[~np.isnan(xSub2)]

    realValues3 = ~np.isnan(xSub3)
    xSubNew3 = xSub3[~np.isnan(xSub3)]
    zSubNew3 = zSub3[~np.isnan(xSub3)]

    realValues4 = ~np.isnan(xSub4)
    xSubNew4 = xSub4[~np.isnan(xSub4)]
    zSubNew4 = zSub4[~np.isnan(xSub4)]

    realValues5 = ~np.isnan(xSub5)
    xSubNew5 = xSub5[~np.isnan(xSub5)]
    zSubNew5 = zSub5[~np.isnan(xSub5)]

    realValues6 = ~np.isnan(xSub6)
    xSubNew6 = xSub6[~np.isnan(xSub6)]
    zSubNew6 = zSub6[~np.isnan(xSub6)]

    temp = np.hstack((np.hstack((np.hstack((realValues,realValues2)),realValues3)),realValues4))
    tempRealValues = np.hstack((temp,realValues5))
    tempRealValues2 = np.hstack((tempRealValues,realValues6))

    tempXSub = np.hstack((np.hstack((np.hstack((xSubNew,xSubNew2)),xSubNew3)),xSubNew4))
    tempXSubNew = np.hstack((tempXSub,xSubNew5))
    tempXSubNew2 = np.hstack((tempXSubNew,xSubNew6))

    ## lets find the closest lidar profile
    surveydateUTC = surveydate.timestamp()
    # timeInd = np.abs(np.asarray(tidalTimeUTCWithData)-surveydateUTC)
    timeInd = np.abs(np.asarray(highTideUTCtime)-surveydateUTC)

    profInd = np.where((np.min(timeInd) == timeInd))

    lidarProf = extendedTidalAverages[profInd[0][0],:]
    # lidarProf = tidalAverageWithData[profInd[0][0],:]
    lidarProfCopy = np.copy(lidarProf)
    lidarProfCopy = np.delete(lidarProfCopy,np.where(np.isnan(lidarProfCopy)))

    if len(lidarProfCopy) == 0:
        print('no profile in this tidally averaged time window')
        timeInd = np.abs(np.asarray(highTideUTCtime) - surveydateUTC)
        profInd = np.where((np.min(timeInd) == timeInd))

        if len(zSubNew4) > 2:
            UNDEF = np.nan
            interpExtendedLidarZ = np.interp(lidar_xFRF, xSubNew4, zSubNew4, left=UNDEF)

            extendedTidalAverages[profInd[0][0] - 1, :] = interpExtendedLidarZ
            extendedTidalAverages[profInd[0][0] + 1, :] = interpExtendedLidarZ

            extendedTidalAverages[profInd[0][0], :] = interpExtendedLidarZ
            # extendedTidalAveragesWithData[profInd[0][0], :] = transect1Avg
            # extendedTidalAverages[matchingFullTidalInd[0][0], interTidalLidarInd[0][0]:] = interpExtendedLidarZ
            # extendedTidalAveragesWithData[profInd[0][0], interTidalLidarInd[0][0]:] = interpExtendedLidarZ
            # bathyX.append(chosenX)
            bathyZ[profInd[0][0], :] = interpExtendedLidarZ
            whichTransect[qq] = 960
            print('but we have added a profile up to {} at profInd = {} and time = {}'.format(np.nanmax(zSubNew4),profInd[0][0],surveydate))


    else:

        interTidalLidarInd = np.where((lidarProf<1.5) & (lidarProf>-1.5))
        interTidalLidarZ = lidarProf[interTidalLidarInd]
        interTidalLidarX = lidar_xFRF[interTidalLidarInd]
        # print('we are trying to fuse to {}'.format(tidalTimeWithData[profInd]))
        print('we are trying to fuse to {}'.format(highTideTimes[profInd[0][0]]))

        # lidarZextrapolated
        # matchingFullTidalInd = np.where(tidalTimeWithData[profInd] == highTideTimes)
        matchingFullTidalInd = profInd  #np.where(tidalTimeWithData[profInd] == highTideTimes)


        if len(interTidalLidarZ) < 2:
            print('needing to look higher on the profile to find lidar data')
            interTidalLidarInd = np.where((lidarProf < 3.5) & (lidarProf > -1.5))
            interTidalLidarZ = lidarProf[interTidalLidarInd]
            interTidalLidarX = lidar_xFRF[interTidalLidarInd]

        if len(interTidalLidarZ) < 2:
            print('We should probably just skip this one....')
        else:

            if len(zSubNew) > 2:
                zSub1Intertp = np.interp(interTidalLidarX,xSubNew,zSubNew)
                r1 = np.corrcoef(interTidalLidarZ, zSub1Intertp)[0, 1]
                rmse1 = rmse(interTidalLidarZ, zSub1Intertp)
            else:
                r1 = 0
                rmse1 = 100

            if len(zSubNew2) > 2:
                zSub2Intertp = np.interp(interTidalLidarX,xSubNew2,zSubNew2)
                r2 = np.corrcoef(interTidalLidarZ, zSub2Intertp)[0, 1]
                rmse2 = rmse(interTidalLidarZ, zSub2Intertp)
            else:
                r2 = 0
                rmse2 = 100

            if len(zSubNew3) > 2:
                zSub3Intertp = np.interp(interTidalLidarX,xSubNew3,zSubNew3)
                r3 = np.corrcoef(interTidalLidarZ, zSub3Intertp)[0, 1]
                rmse3 = rmse(interTidalLidarZ, zSub3Intertp)
            else:
                r3 = 0
                rmse3 = 100

            if len(zSubNew4) > 2:
                zSub4Intertp = np.interp(interTidalLidarX,xSubNew4,zSubNew4)
                r4 = np.corrcoef(interTidalLidarZ, zSub4Intertp)[0, 1]
                rmse4 = rmse(interTidalLidarZ, zSub4Intertp)
            else:
                r4 = 0
                rmse4 = 100

            if len(zSubNew5) > 2:
                zSub5Intertp = np.interp(interTidalLidarX,xSubNew5,zSubNew5)
                r5 = np.corrcoef(interTidalLidarZ, zSub5Intertp)[0,1]
                rmse5 = rmse(interTidalLidarZ, zSub5Intertp)
            else:
                r5 = 0
                rmse5 = 100

            if len(zSubNew6) > 2:
                zSub6Intertp = np.interp(interTidalLidarX,xSubNew6,zSubNew6)
                r6 = np.corrcoef(interTidalLidarZ, zSub6Intertp)[0,1]
                rmse6 = rmse(interTidalLidarZ, zSub6Intertp)
            else:
                r6 = 0
                rmse6 = 100

            allRMSE = [rmse1,rmse2,rmse3,rmse4,rmse5,rmse6]
            bestCaseInd = np.argmin([rmse1,rmse2,rmse3,rmse4,rmse5,rmse6])

            interpBathyOnToX = lidar_xFRF[interTidalLidarInd[0][0]:]
            if bestCaseInd == 0:
                interpExtendedLidarZ = np.interp(interpBathyOnToX,xSubNew,zSubNew)
                chosenX = xSubNew
                chosenZ = zSubNew
                lineNumber= 914
            elif bestCaseInd == 1:
                interpExtendedLidarZ = np.interp(interpBathyOnToX,xSubNew2,zSubNew2)
                chosenX = xSubNew2
                chosenZ = zSubNew2
                lineNumber= 927

            elif bestCaseInd == 2:
                interpExtendedLidarZ = np.interp(interpBathyOnToX,xSubNew3,zSubNew3)
                chosenX = xSubNew3
                chosenZ = zSubNew3
                lineNumber= 951

            elif bestCaseInd == 3:
                interpExtendedLidarZ = np.interp(interpBathyOnToX,xSubNew4,zSubNew4)
                chosenX = xSubNew4
                chosenZ = zSubNew4
                lineNumber= 960

            elif bestCaseInd == 4:
                interpExtendedLidarZ = np.interp(interpBathyOnToX,xSubNew5,zSubNew5)
                chosenX = xSubNew5
                chosenZ = zSubNew5
                lineNumber= 1006

            elif bestCaseInd == 5:
                interpExtendedLidarZ = np.interp(interpBathyOnToX,xSubNew6,zSubNew6)
                chosenX = xSubNew6
                chosenZ = zSubNew6
                lineNumber= 869

                print('HEY WE CHOSE THE 869 LINE')

            if allRMSE[bestCaseInd] < 0.4:
                transect1 = np.nan*extendedTidalAverages[matchingFullTidalInd[0][0],:]
                transect1[interTidalLidarInd[0][0]:] = interpExtendedLidarZ
                transect1Temp = np.vstack((extendedTidalAverages[matchingFullTidalInd[0][0],:],transect1))
                transect1Avg = np.nanmean(transect1Temp,axis=0)

                transect2 = np.nan*extendedTidalAverages[matchingFullTidalInd[0][0]-1,:]
                transect2[interTidalLidarInd[0][0]:] = interpExtendedLidarZ
                transect2Temp = np.vstack((extendedTidalAverages[matchingFullTidalInd[0][0],:],transect2))
                transect2Avg = np.nanmean(transect2Temp,axis=0)
                transect3 = np.nan*extendedTidalAverages[matchingFullTidalInd[0][0]-1,:]
                transect3[interTidalLidarInd[0][0]:] = interpExtendedLidarZ
                transect3Temp = np.vstack((extendedTidalAverages[matchingFullTidalInd[0][0],:],transect3))
                transect3Avg = np.nanmean(transect3Temp,axis=0)
                extendedTidalAverages[matchingFullTidalInd[0][0]-1, :] = transect2Avg
                extendedTidalAverages[matchingFullTidalInd[0][0]+1, :] = transect3Avg

                extendedTidalAverages[matchingFullTidalInd[0][0], :] = transect1Avg
                # extendedTidalAveragesWithData[profInd[0][0], :] = transect1Avg
                # extendedTidalAverages[matchingFullTidalInd[0][0], interTidalLidarInd[0][0]:] = interpExtendedLidarZ
                # extendedTidalAveragesWithData[profInd[0][0], interTidalLidarInd[0][0]:] = interpExtendedLidarZ
                # bathyX.append(chosenX)
                bathyZ[matchingFullTidalInd[0][0], interTidalLidarInd[0][0]:] = interpExtendedLidarZ
                whichTransect[qq] = lineNumber

            else:
                print('but RMSE was {}'.format(allRMSE[bestCaseInd]))
                # bathyX.append(np.nan)
                # bathyZ.append(np.nan)



# plt.figure()
# plt.pcolor(highTideTimes,lidar_xFRF,extendedTidalAverages.T)
# plt.ylabel('xFRF (m)')
# plt.title('Tidal Average Profiles With Bathy')
# plt.ylim([50,170])



howManyObs = []
howManyTidalExtended = []
howManyTidal= []

for qq in range(len(lidar_xFRF)):
    finder = np.where(np.isnan(elev_input[:,qq]))
    howManyObs.append(len(finder[0]))
    finder2 = np.where(np.isnan(extendedTidalAverages[:,qq]))
    howManyTidalExtended.append(len(finder2[0]))
    finder3 = np.where(np.isnan(tidalAverage[:,qq]))
    howManyTidal.append(len(finder3[0]))
numOfObs = np.abs(np.asarray(howManyObs)-np.nanmax(np.asarray(howManyObs)))
numOfTidalExtended = np.abs(np.asarray(howManyTidalExtended)-np.nanmax(np.asarray(howManyTidalExtended)))
numOfTidal = np.abs(np.asarray(howManyTidal)-np.nanmax(np.asarray(howManyTidal)))
#
# plt.figure()
# p1 = plt.subplot2grid((1,3),(0,0))
# p2 = plt.subplot2grid((1,3),(0,2))
# p3 = plt.subplot2grid((1,3),(0,1))
# p1.plot(lidar_xFRF,numOfObs)
# p2.plot(lidar_xFRF,numOfTidalExtended)
# p3.plot(lidar_xFRF,numOfTidal)
# p1.set_xlabel('xFRF (m)')
# p2.set_xlabel('xFRF (m)')
# p3.set_xlabel('xFRF (m)')
# p1.set_ylabel('# of Profiles with data')
# p2.set_ylabel('# of Tidal with Bathy')
# p3.set_ylabel('# of Tidal Windows')
# p1.plot([100,100],[-10,50000],'--',color='k')
# p2.plot([100,100],[-10,4200],'--',color='k')
# p3.plot([100,100],[-10,4200],'--',color='k')
# p1.set_ylim([0,50000])
# p2.set_ylim([0,4200])
# p3.set_ylim([0,4200])
# p1.set_title('All Observations')
# p2.set_title('Tidal with Bathy')
# p3.set_title('Tidal Averaged Windows')


testSample = np.copy(extendedTidalAverages)
testTime = np.asarray(highTideUTCtime)#[35:105]
smoothUpperTidalAverage = np.nan * np.ones((np.shape(testSample)))
for hh in range(len(lidar_xFRF[0:220])):
    ogTime = np.copy(testTime)
    ogSample = np.copy(testSample[:,hh])
    badSpots = np.where(np.isnan(ogSample))
    ogSample = np.delete(ogSample,badSpots)
    ogTime = np.delete(ogTime,badSpots)
    if len(ogTime)>0:
        zInterp = interpolate_with_max_gap(ogTime, ogSample, testTime, max_gap=86400*150)#,orig_x_is_sorted=False, target_x_is_sorted=False)
        smoothUpperTidalAverage[:,hh] = zInterp
    print('finished x = {}'.format(lidar_xFRF[hh]))

smoothUpperTidalAverage[:,220:] = extendedTidalAverages[:,220:]

plt.figure()
plt.pcolor(highTideTimes,lidar_xFRF,smoothUpperTidalAverage.T,cmap='viridis')
plt.ylabel('xFRF (m)')
plt.title('Tidal Average Profiles With Bathy')
plt.ylim([50,170])


minus00Time = []
minus00X = []
minus00Z = []
minus01Time = []
minus01X = []
minus01Z = []
minus02Time = []
minus02X = []
minus02Z = []
minus03Time = []
minus03X = []
minus03Z = []
minus04Time = []
minus04X = []
minus04Z = []
minus05Time = []
minus05X = []
minus05Z = []

for i in range(len(subset)):

    file_params = subset[i].split('_')

    ## ### Northern Lines
    data1 = getBathy(os.path.join(geomorphdir, subset[i]), lower=900, upper=920)
    data2 = getBathy(os.path.join(geomorphdir, subset[i]), lower=920, upper=940)
    data3 = getBathy(os.path.join(geomorphdir, subset[i]), lower=950, upper=955)
    data4 = getBathy(os.path.join(geomorphdir, subset[i]), lower=956, upper=965)
    data5 = getBathy(os.path.join(geomorphdir, subset[i]), lower=1000, upper=1015)
    data6 = getBathy(os.path.join(geomorphdir, subset[i]), lower=860, upper=875)

    temp = subset[i].split('_')
    if temp[1] == 'geomorphology':
        temp2 = temp[-1].split('.')
        surveydate = dt.datetime.strptime(temp2[0], '%Y%m%d')
    else:
        surveydate = dt.datetime.strptime(temp[1], '%Y%m%d')

    print('working on {}'.format(surveydate))
    elevs = data1['z']
    cross = data1['x']
    crossind = np.argsort(data1['x'])
    crossS = cross[crossind]
    elevsS = elevs[crossind]

    elevs2 = data2['z']
    cross2 = data2['x']
    crossind2 = np.argsort(data2['x'])
    crossS2 = cross2[crossind2]
    elevsS2 = elevs2[crossind2]

    elevs3 = data3['z']
    cross3 = data3['x']
    crossind3 = np.argsort(data3['x'])
    crossS3 = cross3[crossind3]
    elevsS3 = elevs3[crossind3]

    elevs4 = data4['z']
    cross4 = data4['x']
    crossind4 = np.argsort(data4['x'])
    crossS4 = cross4[crossind4]
    elevsS4 = elevs4[crossind4]

    elevs5 = data5['z']
    cross5 = data5['x']
    crossind5 = np.argsort(data5['x'])
    crossS5 = cross5[crossind5]
    elevsS5 = elevs5[crossind5]

    elevs6 = data6['z']
    cross6 = data6['x']
    crossind6 = np.argsort(data6['x'])
    crossS6 = cross5[crossind6]
    elevsS6 = elevs5[crossind6]

    xSub = np.ma.MaskedArray.filled(crossS, np.nan)
    zSub = np.ma.MaskedArray.filled(elevsS, np.nan)
    xSub2 = np.ma.MaskedArray.filled(crossS2, np.nan)
    zSub2 = np.ma.MaskedArray.filled(elevsS2, np.nan)
    xSub3 = np.ma.MaskedArray.filled(crossS3, np.nan)
    zSub3 = np.ma.MaskedArray.filled(elevsS3, np.nan)
    xSub4 = np.ma.MaskedArray.filled(crossS4, np.nan)
    zSub4 = np.ma.MaskedArray.filled(elevsS4, np.nan)
    xSub5 = np.ma.MaskedArray.filled(crossS5, np.nan)
    zSub5 = np.ma.MaskedArray.filled(elevsS5, np.nan)
    xSub6 = np.ma.MaskedArray.filled(crossS6, np.nan)
    zSub6 = np.ma.MaskedArray.filled(elevsS6, np.nan)

    realValues = ~np.isnan(xSub)
    xSubNew = xSub[~np.isnan(xSub)]
    zSubNew = zSub[~np.isnan(xSub)]

    realValues2 = ~np.isnan(xSub2)
    xSubNew2 = xSub2[~np.isnan(xSub2)]
    zSubNew2 = zSub2[~np.isnan(xSub2)]

    realValues3 = ~np.isnan(xSub3)
    xSubNew3 = xSub3[~np.isnan(xSub3)]
    zSubNew3 = zSub3[~np.isnan(xSub3)]

    realValues4 = ~np.isnan(xSub4)
    xSubNew4 = xSub4[~np.isnan(xSub4)]
    zSubNew4 = zSub4[~np.isnan(xSub4)]

    realValues5 = ~np.isnan(xSub5)
    xSubNew5 = xSub5[~np.isnan(xSub5)]
    zSubNew5 = zSub5[~np.isnan(xSub5)]

    realValues6 = ~np.isnan(xSub6)
    xSubNew6 = xSub6[~np.isnan(xSub6)]
    zSubNew6 = zSub6[~np.isnan(xSub6)]

    findMinus3at4 = np.where((zSubNew4 > -1.25) & (zSubNew4 < -0))
    findMinus3at3 = np.where((zSubNew3 > -1.25) & (zSubNew3 < -0))
    findMinus3at5 = np.where((zSubNew5 > -1.25) & (zSubNew5 < -0))
    findMinus3at2 = np.where((zSubNew2 > -1.25) & (zSubNew2 < -0))

    if len(findMinus3at4[0]) > 2:
        print('found data at 960 profile on {}'.format(surveydate))
        tempFinder = findMinus3at4
        tempX = xSubNew4[tempFinder]
        tempZ = zSubNew4[tempFinder]
        xInterped = np.linspace(tempX[0],tempX[-1],450)
        zInterpForMinus = np.interp(xInterped,tempX,tempZ)


    elif len(findMinus3at3[0]) > 2:
        print('found data at 950 profile on {}'.format(surveydate))
        tempFinder = findMinus3at3
        tempX = xSubNew3[tempFinder]
        tempZ = zSubNew3[tempFinder]
        xInterped = np.linspace(tempX[0],tempX[-1],450)
        zInterpForMinus = np.interp(xInterped,tempX,tempZ)

    elif len(findMinus3at5[0]) > 2:
        print('found data at 1006 profile on {}'.format(surveydate))
        tempFinder = findMinus3at5
        tempX = xSubNew5[tempFinder]
        tempZ = zSubNew5[tempFinder]
        xInterped = np.linspace(tempX[0], tempX[-1], 450)
        zInterpForMinus = np.interp(xInterped, tempX, tempZ)


    # xFoundMinus3 = tempX[tempFinder[0]]
    # zFoundMinus3 = tempZ[tempFinder[0]]
    zOffsetMinus00 = np.abs(zInterpForMinus+0.0)
    closestZindMinus00 = np.nanargmin(zOffsetMinus00)
    closestXMinus00 = xInterped[closestZindMinus00]
    minus00Time.append(surveydate)
    minus00X.append(closestXMinus00)
    minus00Z.append(zInterpForMinus[closestZindMinus00])

    zOffsetMinus01 = np.abs(zInterpForMinus+0.1)
    closestZindMinus01 = np.nanargmin(zOffsetMinus01)
    closestXMinus01 = xInterped[closestZindMinus01]
    minus01Time.append(surveydate)
    minus01X.append(closestXMinus01)
    minus01Z.append(zInterpForMinus[closestZindMinus01])

    zOffsetMinus02 = np.abs(zInterpForMinus+0.2)
    closestZindMinus02 = np.nanargmin(zOffsetMinus02)
    closestXMinus02 = xInterped[closestZindMinus02]
    minus02Time.append(surveydate)
    minus02X.append(closestXMinus02)
    minus02Z.append(zInterpForMinus[closestZindMinus02])

    zOffsetMinus03 = np.abs(zInterpForMinus+0.3)
    closestZindMinus03 = np.nanargmin(zOffsetMinus03)
    closestXMinus03 = xInterped[closestZindMinus03]
    minus03Time.append(surveydate)
    minus03X.append(closestXMinus03)
    minus03Z.append(zInterpForMinus[closestZindMinus03])

    zOffsetMinus04 = np.abs(zInterpForMinus+0.4)
    closestZindMinus04 = np.nanargmin(zOffsetMinus04)
    closestXMinus04 = xInterped[closestZindMinus04]
    minus04Time.append(surveydate)
    minus04X.append(closestXMinus04)
    minus04Z.append(zInterpForMinus[closestZindMinus04])

    zOffsetMinus05 = np.abs(zInterpForMinus+0.5)
    closestZindMinus05 = np.nanargmin(zOffsetMinus05)
    closestXMinus05 = xInterped[closestZindMinus05]
    minus05Time.append(surveydate)
    minus05X.append(closestXMinus05)
    minus05Z.append(zInterpForMinus[closestZindMinus05])



#
# inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus2Time])
# outX = highTideUTCtime
# minus1Interp = np.interp(outX,inX,inY)
#
# inY = np.asarray(minus3X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus3Time])
# outX = highTideUTCtime
# minus15Interp = np.interp(outX,inX,inY)
#
# plt.figure()
# plt.plot(minus2Time,minus2X)
# plt.plot(minus3Time,minus3X)
#
# plt.plot([dt.datetime.fromtimestamp(ts) for ts in highTideUTCtime],minus1Interp)
# plt.plot([dt.datetime.fromtimestamp(ts) for ts in highTideUTCtime],minus15Interp)
#
#
#
#
# plt.figure()
# plt.pcolor(highTideTimes,lidar_xFRF,smoothUpperTidalAverage.T)
# plt.plot(np.asarray(minus00Time),minus00X,color='black',label='0.0')
# plt.plot(np.asarray(minus01Time),minus01X,color='red',label='-0.1')
# plt.plot(np.asarray(minus02Time),minus02X,color='orange',label='-0.2')
# plt.plot(np.asarray(minus03Time),minus03X,color='purple',label='-0.3')
# plt.plot(np.asarray(minus04Time),minus04X,color='pink',label='-0.4')



## What variables do we want to get out of the profiles?
# minimum elevation?
# location of the dune toe (3 m)
# location of MHHW (0.457 m)
# location of MHW (0.36 m)
# MSL is -0.128



# interpX = np.arange(0,50)
# interpX = np.linspace(0,75,151)
interpX = np.linspace(0,75,750)

subaerial = np.nan * np.ones((len(highTideTimes),len(interpX)))
mhhwContour = np.nan*(np.ones((len(highTideTimes),)))
mhwContour = np.nan*(np.ones((len(highTideTimes),)))
mslContour = np.nan*(np.ones((len(highTideTimes),)))
duneContour = np.nan*(np.ones((len(highTideTimes),)))
mslVolume = np.nan*(np.ones((len(highTideTimes),)))
mhhwVolume = np.nan*(np.ones((len(highTideTimes),)))

interpTime = []
c = 0
for tt in range(len(highTideTimes)):

    findDune = np.where((smoothUpperTidalAverage[tt,:] > 2.75) & (smoothUpperTidalAverage[tt,:] < 4.25))


    findMSL = np.where((smoothUpperTidalAverage[tt,:] > -0.75) & (smoothUpperTidalAverage[tt,:] < .75))
    findMHHW = np.where((smoothUpperTidalAverage[tt,:] > 0) & (smoothUpperTidalAverage[tt,:] < 1))
    findMHW = np.where((smoothUpperTidalAverage[tt,:] > -0.2) & (smoothUpperTidalAverage[tt,:] < 0.9))

    if len(findDune[0]) == 0:
        print('no profile on {}'.format(highTideTimes[tt]))
    else:


        xFoundDune = lidar_xFRF[findDune[0]]
        zFoundDune = smoothUpperTidalAverage[tt,findDune[0]]
        zOffsetDune = np.abs(zFoundDune-3.75)
        closestZindDune = np.where(zOffsetDune == np.nanmin(zOffsetDune))
        closestXDune = xFoundDune[closestZindDune[0]]
        duneContour[tt] = closestXDune

        if len(findMHHW[0]) == 0:
            print('found a dune but no MHHW on {}'.format(highTideTimes[tt]))
        else:
            xFoundMHHW = lidar_xFRF[findMHHW[0]]
            zFoundMHHW = smoothUpperTidalAverage[tt, findMHHW[0]]
            zOffsetMHHW = np.abs(zFoundMHHW - 0.457)
            closestZindMHHW = np.where(zOffsetMHHW == np.nanmin(zOffsetMHHW))
            closestXMHHW = xFoundMHHW[closestZindMHHW[0]]
            mhhwContour[tt] = closestXMHHW


        if len(findMHW[0]) == 0:
            print('found a dune but no MHHW on {}'.format(highTideTimes[tt]))
        else:
            xFoundMHW = lidar_xFRF[findMHW[0]]
            zFoundMHW = smoothUpperTidalAverage[tt, findMHW[0]]
            zOffsetMHW = np.abs(zFoundMHW - 0.36)
            closestZindMHW = np.where(zOffsetMHW == np.nanmin(zOffsetMHW))
            closestXMHW = xFoundMHW[closestZindMHW[0]]
            mhwContour[tt] = closestXMHW


        if len(findMSL[0]) == 0:
            print('found a dune but no MSL on {}'.format(highTideTimes[tt]))
        else:
            xFoundMSL = lidar_xFRF[findMSL[0]]
            zFoundMSL = smoothUpperTidalAverage[tt, findMSL[0]]
            zOffsetMSL = np.abs(zFoundMSL + 0.128)
            closestZindMSL = np.where(zOffsetMSL == np.nanmin(zOffsetMSL))
            closestXMSL = xFoundMSL[closestZindMSL[0]]
            mslContour[tt] = closestXMSL


        xOffsetDune = lidar_xFRF-closestXDune
        zInterp = np.interp(interpX,xOffsetDune,smoothUpperTidalAverage[tt,:])
        if c == 0:
            interpProfile = zInterp
        else:
            interpProfile = np.vstack((interpProfile,zInterp))

        findMSLinterped = np.where((zInterp > -0.75) & (zInterp < .75))
        if len(findMSLinterped[0]) == 0:
            print('skipping volume calculation on {}'.format(highTideTimes[tt]))
        else:
            xFoundMSL = interpX[findMSLinterped[0]]
            zFoundMSL = zInterp[findMSLinterped[0]]
            zOffsetMSL = np.abs(zFoundMSL + 0.128)
            closestZindMSL = np.where(zOffsetMSL == np.nanmin(zOffsetMSL))
            closestXMSL = xFoundMSL[closestZindMSL[0]]
            tempAbove = zInterp[:closestZindMSL[0][0]]
            mslVolume[tt] = np.trapz(y=tempAbove-(-0.128*np.ones(len(tempAbove))),x=interpX[:closestZindMSL[0][0]])

        findMHHWinterped = np.where((zInterp > 0) & (zInterp < 1))
        if len(findMHHWinterped[0]) == 0:
            print('skipping volume calculation on {}'.format(highTideTimes[tt]))
        else:
            xFoundMHHW = interpX[findMHHWinterped[0]]
            zFoundMHHW = zInterp[findMHHWinterped[0]]
            zOffsetMHHW = np.abs(zFoundMHHW - 0.457)
            closestZindMHHW = np.where(zOffsetMHHW == np.nanmin(zOffsetMHHW))
            closestXMHHW = xFoundMHHW[closestZindMHHW[0]]
            tempAbove = zInterp[:closestZindMHHW[0][0]]
            mhhwVolume[tt] = np.trapz(y=tempAbove-(0.457*np.ones(len(tempAbove))),x=interpX[:closestZindMHHW[0][0]])


        c = c + 1
        subaerial[tt,:] = zInterp
        interpTime.append(highTideTimes[tt])





plt.figure()
p10 = plt.subplot2grid((4,1),(0,0))
p10.plot(highTideTimes,duneContour)
p10.set_ylabel('Dune position (m, xFRF)')
p11 = plt.subplot2grid((4,1),(1,0))
p11.plot(highTideTimes,mhhwContour)
p11.set_ylabel('MHHW (m, xFRF)')
p12 = plt.subplot2grid((4,1),(2,0))
p12.plot(highTideTimes,mhwContour)
p12.set_ylabel('MHW (m, xFRF)')
p12 = plt.subplot2grid((4,1),(3,0))
p12.plot(highTideTimes,mslContour)
p12.set_ylabel('MSL (m, xFRF)')

# p12.plot(highTideTimes,mslVolume,label='MSL')
# p12.plot(highTideTimes,mhhwVolume,label='MHHW')
# p12.set_ylabel('Volumes ($m^{3}/m$)')
# p12.legend()





clusterPickle = 'tidalAveragedMetrics.pickle'
output = {}
output['interpTimeS'] = interpTime
output['eofSubaerial'] = subaerial
output['lidar_xFRF'] = lidar_xFRF
output['interpX'] = interpX
output['mslContour'] = mslContour
output['highTideTimes'] = highTideTimes
output['highTideUTCtime'] = highTideUTCtime
output['mhwContour'] = mhwContour
output['tidalAverageWithData'] = tidalAverageWithData
output['tidalAverage'] = tidalAverage
output['duneContour'] = duneContour
output['mhhwContour'] = mhhwContour
output['smoothUpperTidalAverage'] = smoothUpperTidalAverage
output['mhhwVolume'] = mhhwVolume
output['mslVolume'] = mslVolume
import pickle
with open(clusterPickle,'wb') as f:
    pickle.dump(output, f)





asdf



stormPickle = 'stormHs90Over24Hours.pickle'
with open(stormPickle, "rb") as input_file:
    inputStorms = pickle.load(input_file)
timeStormList = inputStorms['timeStormList']
hsStormList = inputStorms['hsStormList']
hsMaxStormList = inputStorms['hsMaxStormList']
tpStormList = inputStorms['tpStormList']
dmStormList = inputStorms['dmStormList']
hourStormList = inputStorms['hourStormList']
indStormList = inputStorms['indStormList']
durationStormList = inputStorms['durationStormList']
wavePowerStormList = inputStorms['wavePowerStormList']
longshorePowerStormList = inputStorms['longshorePowerStormList']
startTimeStormList = inputStorms['startTimeStormList']
endTimeStormList = inputStorms['endTimeStormList']
cHs = inputStorms['cHs']
cTp = inputStorms['cTp']
cDp = inputStorms['cDp']
waveNorm = inputStorms['waveNorm']
cTime = inputStorms['cTime']


fig1 = plt.figure()
p100 = plt.subplot2grid((1,1),(0,0))
p100.plot(highTideTimes,mhwContour)
p100.set_ylabel('MHW (m, xFRF)')
for ff in range(len(timeStormList)):
    p100.plot([timeStormList[ff][0],timeStormList[ff][0]],[75,140],color='orange')



# asdfg
#
# # plt.scatter([dt.datetime.fromtimestamp(ts) for ts in lidarTime],cont_ts[4,:],10,color='orange')
# s1 = plt.scatter([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData],cont_Tidal[2,:],10,color='pink',label='0.8 m')
# s2 = plt.scatter([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData],cont_Tidal[3,:],10,color='red',label='0.7 m')
# s3 = plt.scatter([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData],cont_Tidal[4,:],10,color='orange',label='0.6 m')
# s4 = plt.scatter([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData],cont_Tidal[5,:],10,color='purple',label='0.5 m')
# plt.legend()
# plt.title('-0.4 contour extracted from both bathy (red) and lidar (orange)')
# #
# lidarMinus05 = cont_Tidal[5,:]
# lidarMinus05Time = np.copy([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData])
# missingLidarMinus05Inds = np.where(np.isnan(lidarMinus05))
# lidarMinus05 = np.delete(lidarMinus05,missingLidarMinus05Inds)
# lidarMinus05Time = np.delete(lidarMinus05Time,missingLidarMinus05Inds)
# minus05combined = np.hstack((lidarMinus05,np.asarray(minus05X)))
# minus05combinedTime = np.hstack((lidarMinus05Time,np.asarray(minus05Time)))
# sorted05Inds = np.argsort(minus05combinedTime)
# minus05sortedZ = minus05combined[sorted05Inds]
# minus05sortedZTime = minus05combinedTime[sorted05Inds]
# # inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus05sortedZTime])
# outX = highTideUTCtime #tidalTimeUTCWithData
# minus05Interp = np.interp(outX,inX,minus05sortedZ)
#
#
# lidarMinus00 = cont_Tidal[10,:]
# lidarMinus00Time = np.copy([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData])
# missingLidarMinus00Inds = np.where(np.isnan(lidarMinus00))
# lidarMinus00 = np.delete(lidarMinus00,missingLidarMinus00Inds)
# lidarMinus00Time = np.delete(lidarMinus00Time,missingLidarMinus00Inds)
# minus00combined = np.hstack((lidarMinus00,np.asarray(minus00X)))
# minus00combinedTime = np.hstack((lidarMinus00Time,np.asarray(minus00Time)))
# sorted00Inds = np.argsort(minus00combinedTime)
# minus00sortedZ = minus00combined[sorted00Inds]
# minus00sortedZTime = minus00combinedTime[sorted00Inds]
# # inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus00sortedZTime])
# outX = highTideUTCtime #tidalTimeUTCWithData
# minus00Interp = np.interp(outX,inX,minus00sortedZ)
#
#
# lidarMinus01 = cont_Tidal[9,:]
# lidarMinus01Time = np.copy([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData])
# missingLidarMinus01Inds = np.where(np.isnan(lidarMinus01))
# lidarMinus01 = np.delete(lidarMinus01,missingLidarMinus01Inds)
# lidarMinus01Time = np.delete(lidarMinus01Time,missingLidarMinus01Inds)
# minus01combined = np.hstack((lidarMinus01,np.asarray(minus01X)))
# minus01combinedTime = np.hstack((lidarMinus01Time,np.asarray(minus01Time)))
# sorted01Inds = np.argsort(minus01combinedTime)
# minus01sortedZ = minus01combined[sorted01Inds]
# minus01sortedZTime = minus01combinedTime[sorted01Inds]
# # inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus01sortedZTime])
# outX = highTideUTCtime #tidalTimeUTCWithData
# minus01Interp = np.interp(outX,inX,minus01sortedZ)
#
#
#
# lidarMinus02 = cont_Tidal[8,:]
# lidarMinus02Time = np.copy([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData])
# missingLidarMinus02Inds = np.where(np.isnan(lidarMinus02))
# lidarMinus02 = np.delete(lidarMinus02,missingLidarMinus02Inds)
# lidarMinus02Time = np.delete(lidarMinus02Time,missingLidarMinus02Inds)
# minus02combined = np.hstack((lidarMinus02,np.asarray(minus02X)))
# minus02combinedTime = np.hstack((lidarMinus02Time,np.asarray(minus02Time)))
# sorted02Inds = np.argsort(minus02combinedTime)
# minus02sortedZ = minus02combined[sorted02Inds]
# minus02sortedZTime = minus02combinedTime[sorted02Inds]
# # inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus02sortedZTime])
# outX = highTideUTCtime #tidalTimeUTCWithData
# minus02Interp = np.interp(outX,inX,minus02sortedZ)
#
# lidarMinus03 = cont_Tidal[7,:]
# lidarMinus03Time = np.copy([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData])
# missingLidarMinus03Inds = np.where(np.isnan(lidarMinus03))
# lidarMinus03 = np.delete(lidarMinus03,missingLidarMinus03Inds)
# lidarMinus03Time = np.delete(lidarMinus03Time,missingLidarMinus03Inds)
# minus03combined = np.hstack((lidarMinus03,np.asarray(minus03X)))
# minus03combinedTime = np.hstack((lidarMinus03Time,np.asarray(minus03Time)))
# sorted03Inds = np.argsort(minus03combinedTime)
# minus03sortedZ = minus03combined[sorted03Inds]
# minus03sortedZTime = minus03combinedTime[sorted03Inds]
# # inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus03sortedZTime])
# outX = highTideUTCtime #tidalTimeUTCWithData
# minus03Interp = np.interp(outX,inX,minus03sortedZ)
#
# lidarMinus04 = cont_Tidal[6,:]
# lidarMinus04Time = np.copy([dt.datetime.fromtimestamp(ts) for ts in tidalTimeUTCWithData])
# missingLidarMinus04Inds = np.where(np.isnan(lidarMinus04))
# lidarMinus04 = np.delete(lidarMinus04,missingLidarMinus04Inds)
# lidarMinus04Time = np.delete(lidarMinus04Time,missingLidarMinus04Inds)
# minus04combined = np.hstack((lidarMinus04,np.asarray(minus04X)))
# minus04combinedTime = np.hstack((lidarMinus04Time,np.asarray(minus04Time)))
# sorted04Inds = np.argsort(minus04combinedTime)
# minus04sortedZ = minus04combined[sorted04Inds]
# minus04sortedZTime = minus04combinedTime[sorted04Inds]
# # inY = np.asarray(minus2X).flatten()
# inX = np.asarray([dt.datetime.timestamp(ts) for ts in minus04sortedZTime])
# outX = highTideUTCtime #tidalTimeUTCWithData
# minus04Interp = np.interp(outX,inX,minus04sortedZ)

#
# plt.figure()
# plt.pcolor(highTideTimes,lidar_xFRF,tidalAverage.T)
# s1 = plt.plot(tidalTimeWithData,minus08Interp,color='pink',label='0.8 m')
# s2 = plt.plot(tidalTimeWithData,minus07Interp,color='red',label='0.7 m')
# s3 = plt.plot(tidalTimeWithData,minus06Interp,color='orange',label='0.6 m')
# s4 = plt.plot(tidalTimeWithData,minus05Interp,color='purple',label='0.5 m')
# plt.legend()
#
#


#
# from scipy.interpolate import CubicSpline
# from scipy.interpolate import interp1d
# counter = 1
# extrapolatedX = []
# extrapolatedZ = []
# lidarZextrapolated = np.copy(tidalAverage)
# # lidarZextrapolated = np.copy(tidalAverageWithData)
#
#
# for qq in range(len(tidalAverage)):
#     if np.remainder(qq,100):
#         print('done with {} of {}: {}'.format(qq,len(highTideTimes),highTideTimes[qq]))
#
#     temporaryProfile = tidalAverage[qq,:]
#     copyForCheck = np.copy(temporaryProfile)
#     indsOfNan = np.where(np.isnan(temporaryProfile))
#     copyForCheck = np.delete(copyForCheck,indsOfNan)
#
#     if len(copyForCheck) > 0:
#
#         indsBelow0 = np.where((temporaryProfile<0))
#
#         if len(indsBelow0[0]) > 25:
#             print('no need to extend we have a decent amount of data below 0.0 = {} points'.format(len(indsBelow0[0])))
#         else:
#             print('ideal line to extend on {}'.format(highTideTimes[qq]))
#             indsBelow0pt5 = np.where((temporaryProfile < 0.5))
#             if len(indsBelow0pt5[0]) < 70:
#                 print('woah, need to look at lower than 0.75')
#                 indsBelow0pt5 = np.where((temporaryProfile < 0.75))
#                 if len(indsBelow0pt5[0]) < 70:
#                     print('woah, need to look at lower than 1.00')
#                     indsBelow0pt5 = np.where((temporaryProfile < 1.00))
#                     if len(indsBelow0pt5[0]) < 70:
#                         print('woah, need to look at lower than 1.25')
#                         indsBelow0pt5 = np.where((temporaryProfile < 1.25))
#                         if len(indsBelow0pt5[0]) < 70:
#                             print('woah, need to look at lower than 1.50')
#                             indsBelow0pt5 = np.where((temporaryProfile < 1.50))
#                             if len(indsBelow0pt5[0]) < 70:
#                                 print('woah, need to look at lower than 1.75')
#                                 indsBelow0pt5 = np.where((temporaryProfile < 1.75))
#                                 if len(indsBelow0pt5[0]) < 70:
#                                     print('woah, need to look at lower than 2.00')
#                                     indsBelow0pt5 = np.where((temporaryProfile < 2.00))
#             if len(indsBelow0pt5[0]) == 0:
#                 print('its bad, lets skip this one altogether')
#
#             else:
#                 if minus06Interp[qq] < minus05Interp[qq]:
#                     print('our -0.5 contour is further offshore than -0.6 at profile {}:{}'.format(qq, highTideTimes[qq]))
#                     extX = np.hstack((lidar_xFRF[indsBelow0pt5].flatten(),minus05Interp[qq]))
#                     extZ = np.hstack((temporaryProfile[indsBelow0pt5],-0.5))
#                     sortedInds = np.argsort(extX)
#                     extX = extX[sortedInds]
#                     extZ = extZ[sortedInds]
#                     uniqueInds = np.unique(extX,return_index=True)
#                     extX = extX[uniqueInds[1]]
#                     extZ = extZ[uniqueInds[1]]
#                 elif (minus06Interp[qq]-minus05Interp[qq]) < 4 and (minus06Interp[qq]-minus05Interp[qq]) > 2:
#                     print('our -0.5 contour is onshore of -0.6 and within 5 m at profile {}:{}'.format(qq, highTideTimes[qq]))
#                     # extX = np.hstack((lidar_xFRF[indsBelow0pt5].flatten(),minus05Interp[qq],minus06Interp[qq]))
#                     # extZ = np.hstack((temporaryProfile[indsBelow0pt5],-0.5,-0.6))
#                     extX = np.hstack((lidar_xFRF[indsBelow0pt5].flatten(),minus05Interp[qq]))
#                     extZ = np.hstack((temporaryProfile[indsBelow0pt5],-0.5))
#                     sortedInds = np.argsort(extX)
#                     extX = extX[sortedInds]
#                     extZ = extZ[sortedInds]
#                     uniqueInds = np.unique(extX,return_index=True)
#                     extX = extX[uniqueInds[1]]
#                     extZ = extZ[uniqueInds[1]]
#                 else:
#                     #
#                     # extX = np.asarray([lidar_xFRF[indsBelow0pt5][0],minus05Interp[qq],minus06Interp[qq]]).flatten()
#                     # extZ = np.asarray([temporaryProfile[indsBelow0pt5][0],-0.5,-0.6])
#                     extX = np.hstack((lidar_xFRF[indsBelow0pt5].flatten(),minus05Interp[qq]))
#                     extZ = np.hstack((temporaryProfile[indsBelow0pt5],-0.5))
#                     sortedInds = np.argsort(extX)
#                     extX = extX[sortedInds]
#                     extZ = extZ[sortedInds]
#                     uniqueInds = np.unique(extX,return_index=True)
#                     extX = extX[uniqueInds[1]]
#                     extZ = extZ[uniqueInds[1]]
#
#
#                 if lidar_xFRF[indsBelow0pt5][-1] >= minus05Interp[qq]:
#                     print('we should skip this one ...')
#
#                 else:
#
#                     offshoreXInd = np.where(lidar_xFRF<minus05Interp[qq])[0][-1]
#                     # offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                     offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#
#                     if minus05Interp[qq] -lidar_xFRF[indsBelow0pt5[0][-1]] > 12:
#
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1] + 15
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = CubicSpline(extX, extZ, bc_type='natural')
#                         # f = interp1d(extX,extZ,kind = 'quadratic', fill_value = 'extrapolate')
#                         z_new = f(offshoreX)
#
#
#                     else:
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#
#                     if np.max(z_new) - extZ[-2] > 0.15:
#                         print('we are too parabolic upwards')
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#                     if z_new[-1] - np.min(z_new) > 0.1:
#                         print('we are too parabolic downwards')
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#
#                     if np.max(z_new) > 3:
#                         print('we went crazy upwards!')
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#                     elif np.min(z_new) < -3:
#                         print('we went crazy downwards!')
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#                     elif np.max(z_new) - z_new[0] > 0.5:
#                         print('we are too parabolic upwards')
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#                     elif z_new[-1] - np.min(z_new) > 0.2:
#                         print('we are too parabolic downwards')
#                         offshoreXInd = np.where(lidar_xFRF < minus05Interp[qq])[0][-1]
#                         offshoreX = lidar_xFRF[indsBelow0pt5[0][0]:offshoreXInd]
#                         f = interp1d(extX,extZ)
#                         z_new = f(offshoreX)
#                     # elif (np.max(z_new) - np.min(z_new)) > 0.2:
#                     #     print('we are too parabolic downwards')
#                     else:
#                         difInd = indsBelow0pt5[0][-1]-indsBelow0pt5[0][0]
#                         lidarZextrapolated[qq,indsBelow0pt5[0][-1]:offshoreXInd] = z_new[difInd:]
#
#
#                         # plt.figure()
#                         # plt.plot(offshoreX,z_new,'--',color='black')
#                         # plt.plot(lidar_xFRF, temporaryProfile)
#                         # plt.scatter(minus05Interp[qq], -0.5, 10, color='red')
#                         # plt.scatter(minus06Interp[qq], -0.6, 10, color='orange')
#                         # plt.scatter(minus07Interp[qq], -0.7, 10, color='purple')
#                         # plt.savefig('/volumes/anderson/extendedProfiles/prof{}'.format(counter))
#                         # plt.close()
#                         counter = counter + 1
#                         extrapolatedX.append(offshoreX)
#                         extrapolatedZ.append(z_new)
#
#
#





#
# testSample = np.copy(lidarelev)
# testTime = np.asarray(lidarTime)#[35:105]
# smoothLidar = np.nan * np.ones((np.shape(testSample)))
# for hh in range(len(lidarTime)):
#
#     ogX = np.copy(lidar_xFRF)
#     ogSample = np.copy(testSample[hh,:])
#     badSpots = np.where(np.isnan(ogSample))
#     ogSample = np.delete(ogSample,badSpots)
#     ogX = np.delete(ogX,badSpots)
#     if len(ogX)>0:
#         zInterp = interpolate_with_max_gap(ogX, ogSample, lidar_xFRF, max_gap=20)#,orig_x_is_sorted=False, target_x_is_sorted=False)
#
#
#         smoothLidar[:,hh] = zInterp
#     print('finished time = {}'.format(dt.datetime.fromtimestamp(lidarTime[hh])))
#





#
# plt.figure()
# plt.pcolor(highTideTimes,lidar_xFRF,extendedTidalAverages.T)
# plt.ylabel('xFRF (m)')
# plt.title('Extended Profile densities')
#
#
# counter = 1
# for qq in range(len(highTideTimes)):
#
#     tempProfile = extendedTidalAverages[qq,:]
#     nanFinder = np.where(np.isnan(tempProfile))
#     prof = np.delete(tempProfile,nanFinder)
#     if len(prof) > 3:
#
#         if np.nanmin(prof) < -1:
#             plt.figure(figsize=(6,6))
#             p1 = plt.subplot2grid((2,2),(0,0))
#             p2 = plt.subplot2grid((2,2),(0,1))
#             p3 = plt.subplot2grid((2,2),(1,0))
#             p4 = plt.subplot2grid((2,2),(1,1))
#
#             p1.plot(lidar_xFRF,extendedTidalAverages[qq,:])
#             p1.plot(lidar_xFRF,tidalAverage[qq,:],color='k',linewidth=2)
#             p1.plot(lidar_xFRF,bathyZ[qq],color=[0.5,0.5,0.5])
#             p1.plot()
#             p1.set_title(highTideTimes[qq])
#             p1.set_ylim([-2.5,3])
#             p1.set_xlim([50,150])
#             for hh in range(5):
#                 p2.plot(lidar_xFRF,extendedTidalAverages[qq-hh-1,:])
#             p2.set_title('5 profiles before')
#             p2.plot(lidar_xFRF,extendedTidalAverages[qq,:],'k')
#             p2.set_ylim([-2.5,3])
#             p2.set_xlim([50,150])
#
#             for hh in range(5):
#                 p3.plot(lidar_xFRF,extendedTidalAverages[qq+hh+1,:])
#             p3.set_title('5 profiles after')
#             p3.plot(lidar_xFRF,extendedTidalAverages[qq,:],'k')
#             p3.set_ylim([-2.5,3])
#             p3.set_xlim([50,150])
#
#             p4.pcolor(np.asarray(highTideTimes)[qq-5:qq+5],lidar_xFRF,extendedTidalAverages[qq-5:qq+5,:].T,vmin=-2,vmax=2)
#             p4.set_ylim([50,150])
#             p4.set_xlim([highTideTimes[qq-6],highTideTimes[qq+6]])
#             plt.savefig('/volumes/anderson/recoveryProfiles/prof{}'.format(counter))
#             plt.close()
#             counter = counter + 1




# plt.figure()
# p1 = plt.subplot2grid((1,2),(0,0))
# p2 = plt.subplot2grid((1,2),(0,1))
# p1.plot(xSubNew,zSubNew)
# p1.plot(xSubNew2,zSubNew2)
# p1.plot(xSubNew3,zSubNew3)
# p1.plot(xSubNew4,zSubNew4)
# p1.plot(lidar_xFRF,lidarProf,color='k',linewidth=2,label='Lidar')
# p1.set_xlim([47,255])
# p1.set_ylim([-3.1,5])
# p2.scatter(interTidalLidarZ,zSub1Intertp,label='914 = {} RMSE'.format(np.round(rmse1*100)/100))
# p2.scatter(interTidalLidarZ,zSub2Intertp,label='927 = {} RMSE'.format(np.round(rmse2*100)/100))
# p2.scatter(interTidalLidarZ,zSub3Intertp,label='951 = {} RMSE'.format(np.round(rmse3*100)/100))
# p2.scatter(interTidalLidarZ,zSub4Intertp,label='960 = {} RMSE'.format(np.round(rmse4*100)/100))
# p2.plot([0,2],[0,2],'k--')
# plt.legend()
# p2.set_xlabel('Lidar Profile (m)')
# p2.set_ylabel('Bathy Profiles (m)')
# p2.set_title('1-to-1 Intertidal Profiles')
# p1.set_xlabel('xFRF (m)')
# p1.set_ylabel('Elevation (m)')
# p1.set_title('Profile Comparisons')
# asdfg
#
# testSample = np.copy(extendedTidalAverages)
# testTime = np.asarray(highTideUTCtime)#[35:105]
# smoothTidalAverage = np.nan * np.ones((np.shape(testSample)))
# for hh in range(len(lidar_xFRF)):
#     ogTime = np.copy(testTime)
#     ogSample = np.copy(testSample[:,hh])
#     badSpots = np.where(np.isnan(ogSample))
#     ogSample = np.delete(ogSample,badSpots)
#     ogTime = np.delete(ogTime,badSpots)
#     if len(ogTime)>0:
#         zInterp = interpolate_with_max_gap(ogTime, ogSample, testTime, max_gap=86400*7)#,orig_x_is_sorted=False, target_x_is_sorted=False)
#         smoothTidalAverage[:,hh] = zInterp
#     print('finished x = {}'.format(lidar_xFRF[hh]))
#
# #
# # #
# # plt.figure()
# # p1 = plt.subplot2grid((1,2),(0,0))
# # p1.pcolor(np.asarray(highTideTimes)[35:105],lidar_xFRF[0:1100],tidalAverage[35:105,0:1100].T)
# # p1.set_ylabel('xFRF (m)')
# # # p1.set_ylim([40,140])
# # p1.set_title('tidally-averaged profiles')
# # p1.tick_params(axis='x', labelrotation=45)
# # p2 = plt.subplot2grid((1,2),(0,1))
# # p2.pcolor(np.asarray(highTideTimes)[35:105],lidar_xFRF[0:1100],smoothTidalAverage.T)
# # p2.set_ylabel('xFRF (m)')
# # # p1.set_ylim([40,140])
# # p2.set_title('tidally-averaged profiles')
# # p2.tick_params(axis='x', labelrotation=45)
#
# #
# plt.figure()
# p1 = plt.subplot2grid((2,1),(0,0))
# p1.pcolor(np.asarray(highTideTimes),lidar_xFRF,extendedTidalAverages.T)
# p1.set_ylabel('xFRF (m)')
# # p1.set_ylim([40,140])
# p1.set_title('tidally-averaged extended profiles')
# p1.tick_params(axis='x', labelrotation=45)
# p2 = plt.subplot2grid((2,1),(1,0))
# p2.pcolor(np.asarray(highTideTimes),lidar_xFRF,smoothTidalAverage.T)
# p2.set_ylabel('xFRF (m)')
# # p1.set_ylim([40,140])
# p2.set_title('interpolated profiles')
# p2.tick_params(axis='x', labelrotation=45)

#
# asdfg
#
#
#
#
#
#
#
# howManyObs = []
# howManyDays = []
# howManyTidal= []
# howManyTidalExt= []
#
# for qq in range(len(lidar_xFRF)):
#     finder = np.where(np.isnan(elev_input[:,qq]))
#     howManyObs.append(len(finder[0]))
#     finder2 = np.where(np.isnan(dailyAverage[:,qq]))
#     howManyDays.append(len(finder2[0]))
#     finder3 = np.where(np.isnan(tidalAverage[:,qq]))
#     howManyTidal.append(len(finder3[0]))
#     finder4 = np.where(np.isnan(smoothTidalAverage[:,qq]))
#     howManyTidalExt.append(len(finder4[0]))
# numOfObs = np.abs(np.asarray(howManyObs)-np.nanmax(np.asarray(howManyObs)))
# numOfDays = np.abs(np.asarray(howManyDays)-np.nanmax(np.asarray(howManyDays)))
# numOfTidal = np.abs(np.asarray(howManyTidal)-np.nanmax(np.asarray(howManyTidal)))
# numOfTidalExt = np.abs(np.asarray(howManyTidalExt)-np.nanmax(np.asarray(howManyTidalExt)))
#
# plt.figure()
# p1 = plt.subplot2grid((1,3),(0,0))
# p2 = plt.subplot2grid((1,3),(0,1))
# p3 = plt.subplot2grid((1,3),(0,2))
# p1.plot(lidar_xFRF,numOfObs)
# p2.plot(lidar_xFRF,numOfTidal)
# p3.plot(lidar_xFRF,numOfTidalExt)
# p1.set_xlabel('xFRF (m)')
# p2.set_xlabel('xFRF (m)')
# p3.set_xlabel('xFRF (m)')
# p1.set_ylabel('# of Profiles with data')
# p2.set_ylabel('# of Tidal Windows')
# p3.set_ylabel('# of Tidal Smoothed')
# p1.plot([100,100],[-10,50000],'--',color='k')
# p2.plot([100,100],[-10,4200],'--',color='k')
# p3.plot([100,100],[-10,4500],'--',color='k')
# p1.set_ylim([0,50000])
# p2.set_ylim([0,4200])
# p3.set_ylim([0,4500])
# p1.set_title('All Observations')
# p2.set_title('Tidal Averaged Windows')
# p3.set_title('Tidal Extended and Smoothed')
#
#
#
#
#
#
#
#
#
#
#
#
# asdfg
# ## What variables do we want to get out of the profiles?
# # minimum elevation?
# # location of the dune toe (3 m)
# # location of MHHW (0.457 m)
# # location of MHW (0.36 m)
# # MSL is -0.128
#
#
#
# # interpX = np.arange(0,50)
# # interpX = np.linspace(0,75,151)
# interpX = np.linspace(0,75,750)
#
# subaerial = np.nan * np.ones((len(highTideTimes),len(interpX)))
# mhhwContour = np.nan*(np.ones((len(highTideTimes),)))
# mslContour = np.nan*(np.ones((len(highTideTimes),)))
# duneContour = np.nan*(np.ones((len(highTideTimes),)))
# mslVolume = np.nan*(np.ones((len(highTideTimes),)))
# mhhwVolume = np.nan*(np.ones((len(highTideTimes),)))
#
# interpTime = []
# c = 0
# for tt in range(len(highTideTimes)):
#
#     findMHHW = np.where((smoothTidalAverage[tt,:] > 0) & (smoothTidalAverage[tt,:] < 1))
#     findDune = np.where((smoothTidalAverage[tt,:] > 2.75) & (smoothTidalAverage[tt,:] < 4.25))
#     findMSL = np.where((smoothTidalAverage[tt,:] > -0.75) & (smoothTidalAverage[tt,:] < .75))
#
#     if len(findDune[0]) == 0:
#         print('no profile on {}'.format(highTideTimes[tt]))
#     else:
#
#
#         xFoundDune = lidar_xFRF[findDune[0]]
#         zFoundDune = smoothTidalAverage[tt,findDune[0]]
#         zOffsetDune = np.abs(zFoundDune-3.75)
#         closestZindDune = np.where(zOffsetDune == np.nanmin(zOffsetDune))
#         closestXDune = xFoundDune[closestZindDune[0]]
#         duneContour[tt] = closestXDune
#         if len(findMHHW[0]) == 0:
#             print('found a dune but no MHHW on {}'.format(highTideTimes[tt]))
#         else:
#             xFoundMHHW = lidar_xFRF[findMHHW[0]]
#             zFoundMHHW = smoothTidalAverage[tt, findMHHW[0]]
#             zOffsetMHHW = np.abs(zFoundMHHW - 0.457)
#             closestZindMHHW = np.where(zOffsetMHHW == np.nanmin(zOffsetMHHW))
#             closestXMHHW = xFoundMHHW[closestZindMHHW[0]]
#             mhhwContour[tt] = closestXMHHW
#
#         if len(findMSL[0]) == 0:
#             print('found a dune but no MSL on {}'.format(highTideTimes[tt]))
#         else:
#             xFoundMSL = lidar_xFRF[findMSL[0]]
#             zFoundMSL = smoothTidalAverage[tt, findMSL[0]]
#             zOffsetMSL = np.abs(zFoundMSL + 0.128)
#             closestZindMSL = np.where(zOffsetMSL == np.nanmin(zOffsetMSL))
#             closestXMSL = xFoundMSL[closestZindMSL[0]]
#             mslContour[tt] = closestXMSL
#
#
#         xOffsetDune = lidar_xFRF-closestXDune
#         zInterp = np.interp(interpX,xOffsetDune,smoothTidalAverage[tt,:])
#         if c == 0:
#             interpProfile = zInterp
#         else:
#             interpProfile = np.vstack((interpProfile,zInterp))
#
#         findMSLinterped = np.where((zInterp > -0.75) & (zInterp < .75))
#         if len(findMSLinterped[0]) == 0:
#             print('skipping volume calculation on {}'.format(highTideTimes[tt]))
#         else:
#             xFoundMSL = interpX[findMSLinterped[0]]
#             zFoundMSL = zInterp[findMSLinterped[0]]
#             zOffsetMSL = np.abs(zFoundMSL + 0.128)
#             closestZindMSL = np.where(zOffsetMSL == np.nanmin(zOffsetMSL))
#             closestXMSL = xFoundMSL[closestZindMSL[0]]
#             tempAbove = zInterp[:closestZindMSL[0][0]]
#             mslVolume[tt] = np.trapz(y=tempAbove-(-0.128*np.ones(len(tempAbove))),x=interpX[:closestZindMSL[0][0]])
#
#         findMHHWinterped = np.where((zInterp > 0) & (zInterp < 1))
#         if len(findMHHWinterped[0]) == 0:
#             print('skipping volume calculation on {}'.format(highTideTimes[tt]))
#         else:
#             xFoundMHHW = interpX[findMHHWinterped[0]]
#             zFoundMHHW = zInterp[findMHHWinterped[0]]
#             zOffsetMHHW = np.abs(zFoundMHHW - 0.457)
#             closestZindMHHW = np.where(zOffsetMHHW == np.nanmin(zOffsetMHHW))
#             closestXMHHW = xFoundMHHW[closestZindMHHW[0]]
#             tempAbove = zInterp[:closestZindMHHW[0][0]]
#             mhhwVolume[tt] = np.trapz(y=tempAbove-(0.457*np.ones(len(tempAbove))),x=interpX[:closestZindMHHW[0][0]])
#
#
#         c = c + 1
#         subaerial[tt,:] = zInterp
#         interpTime.append(highTideTimes[tt])
#
#
#
#
#
# plt.figure()
# p10 = plt.subplot2grid((3,1),(0,0))
# p10.plot(highTideTimes,duneContour)
# p10.set_ylabel('Dune position (m, xFRF)')
# p11 = plt.subplot2grid((3,1),(1,0))
# p11.plot(highTideTimes,mhhwContour)
# p11.set_ylabel('MHHW position (m, xFRF)')
# p12 = plt.subplot2grid((3,1),(2,0))
# p12.plot(highTideTimes,mslVolume,label='MSL')
# p12.plot(highTideTimes,mhhwVolume,label='MHHW')
# p12.set_ylabel('Volumes ($m^{3}/m$)')
# p12.legend()
#
#



eofSubaerial = np.copy(interpProfile[:,0:400])
badRows = np.unique(np.where(np.isnan(eofSubaerial))[0])
eofSubaerial = np.delete(eofSubaerial,badRows,0)

interpTimeS = np.copy(interpTime)
interpTimeS = np.delete(interpTimeS,badRows)


#
# clusterPickle = 'tidalAveragedSmoothed.pickle'
# output = {}
# output['interpTimeS'] = interpTime
# output['eofSubaerial'] = subaerial
# output['lidar_xFRF'] = lidar_xFRF
# output['interpX'] = interpX
# output['smoothTidalAverage'] = smoothTidalAverage
# output['highTideTimes'] = highTideTimes
# output['highTideUTCtime'] = highTideUTCtime
# output['extendedTidalAverages'] = extendedTidalAverages
# output['tidalAverageWithData'] = tidalAverageWithData
# output['tidalAverage'] = tidalAverage
#
#
# import pickle
# with open(clusterPickle,'wb') as f:
#     pickle.dump(output, f)
#






from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
data = eofSubaerial # where shorelines is matrix with dimensions of: time (rows) x alongshore transect number (columns)
dataMean = np.mean(data,axis=0) # this will give you an average for each cross-shore transect
dataStd = np.std(data,axis=0)
dataNorm = (data[:,:] - dataMean) / dataStd
# dataNorm[np.isnan(dataNorm)] = 0 # I think this is how you would handle the occasional nan, for now hopefully you don't have any...

# principal components analysis
ipca = PCA(n_components=min(dataNorm.shape[0], dataNorm.shape[1]))
PCs = ipca.fit_transform(dataNorm)  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

plt.figure()
ax1 = plt.subplot2grid((2,3),(0,0))
ax1.plot(interpTimeS,PCs[:,0])
ax1.set_title('Mode 1')
ax2 = plt.subplot2grid((2,3),(1,0))
ax2.plot(interpX[0:400],EOFs[0,:])
ax3 = plt.subplot2grid((2,3),(0,1))
ax3.plot(interpTimeS,PCs[:,1])
ax3.set_title('Mode 2')
ax4 = plt.subplot2grid((2,3),(1,1))
ax4.plot(interpX[0:400],EOFs[1,:])
ax5 = plt.subplot2grid((2,3),(0,2))
ax5.plot(interpTimeS,PCs[:,2])
ax5.set_title('Mode 3')
ax6 = plt.subplot2grid((2,3),(1,2))
ax6.plot(interpX[0:400],EOFs[2,:])







import scipy.linalg as la
import numpy.linalg as npla
from scipy.signal import hilbert

nanmean = np.nanmean(eofSubaerial,axis=0)
finder = np.where(np.isnan(eofSubaerial))
eofSubaerial[finder] = nanmean[finder[1]]
demean = eofSubaerial - np.mean(eofSubaerial,axis=0)
data = (hilbert(demean.T))
data = data.T
c = np.matmul(np.conj(data).T,data)/np.shape(data)[0]

lamda, loadings = la.eigh(c)
lamda2, loadings2 = npla.eig(c)
ind = np.argsort(lamda[::-1])
lamda[::-1].sort()
loadings = loadings[:,ind]


pcs = np.dot(data, loadings)# / np.sqrt(lamda)
loadings = loadings# * np.sqrt(lamda)
pcsreal = np.real(pcs[:,0:200])
pcsimag = np.imag(pcs[:,0:200])
eofreal = np.real(loadings[:,0:200])
eofimag = np.imag(loadings[:,0:200])
S = np.power(loadings*np.conj(loadings),0.5) * np.sqrt(lamda)

theta = np.arctan2(eofimag,eofreal)
theta2 = theta*180/np.pi

Rt = np.power(pcs*np.conj(pcs),0.5) / np.sqrt(lamda)

phit = np.arctan2(pcsimag,pcsreal)
phit2 = phit*180/np.pi


mode = 1

fig, ax = plt.subplots(2,2)

ax[0,0].plot(interpX[0:400], S[:,mode],'o')
ax[0,0].set_ylabel('Spatial Magnitude (m)')
ax[0,0].set_xlabel('Cross-shore (m)')
ax[1,0].plot(interpX[0:400], theta2[:,mode],'o')
ax[1,0].set_ylabel('Spatial Phase (deg)')
ax[1,0].set_xlabel('Cross-shore (m)')
ax[0,1].plot(interpTimeS,Rt[:,mode],'o')
ax[0,1].set_ylabel('Temporal Magnitude (m)')
ax[0,1].set_xlabel('Time')
ax[1,1].plot(interpTimeS,phit2[:,mode],'o')
ax[1,1].set_ylabel('Temporal Phase (deg)')
ax[1,1].set_xlabel('Time')





PC1 = Rt[:, mode]*np.sin(phit[:, mode]) + Rt[:, mode]*np.cos(phit[:, mode])


totalV = np.sum(lamda)
percentV = lamda / totalV

ztemp = 0*np.ones(len(interpX[0:400]),)
timestep = 200
for mode in range(2):
    # ztemp = ztemp + Rt[timestep,mode]*np.sin(phit[timestep,mode]) * S[:,mode]*np.sin(theta[:,mode]) + Rt[timestep,mode]*np.cos(phit[timestep,mode]) * S[:,mode]*np.cos(theta[:,mode])
    ztemp = ztemp + Rt[timestep,mode]*S[:,mode]*np.cos(phit[timestep,mode] - theta[:,mode])


def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)



timeind = np.arange(0,len(interpTimeS))
#timeind = np.arange(3, 50)
RtSubset = Rt[timeind, :]
phitSubset = phit[timeind, :]
phit2Subset = phit2[timeind, :]
timeSubset = interpTimeS[timeind]
alllinesSubset = eofSubaerial[timeind, :]

eofPred = np.nan * np.ones((np.shape(alllinesSubset)))
eofPred2 = np.nan * np.ones((np.shape(alllinesSubset)))
eofPred3 = np.nan * np.ones((np.shape(alllinesSubset)))
eofPred4 = np.nan * np.ones((np.shape(alllinesSubset)))

for timestep in range(len(timeind)):
    mode = 0
    eofPred[timestep, :] = RtSubset[timestep, mode]* S[:, mode] * np.cos(phitSubset[timestep, mode] - theta[:, mode])
    mode = 1
    eofPred2[timestep, :] = RtSubset[timestep, mode]* S[:, mode] * np.cos(phitSubset[timestep, mode] - theta[:, mode])

    mode = 2
    eofPred3[timestep, :] = RtSubset[timestep, mode]* S[:, mode] * np.cos(phitSubset[timestep, mode] - theta[:, mode])

    mode = 3
    eofPred4[timestep, :] = RtSubset[timestep, mode]* S[:, mode] * np.cos(phitSubset[timestep, mode] - theta[:, mode])



t1 = 0
t2 = -1
fig, ax = plt.subplots(1,5)
plt.set_cmap('RdBu_r')

tg, xg = np.meshgrid(interpTimeS, interpX[0:400])
plt0 = ax[0].pcolor(xg,tg,(eofSubaerial-np.mean(eofSubaerial, axis=0)).T, vmin=-1.8, vmax=1.8)
fig.colorbar(plt0, ax=ax[0], orientation='horizontal')
ax[0].set_ylim([interpTimeS[t1], interpTimeS[t2]])
ax[0].set_title('Surveys (dev.)')

plt1 = ax[1].pcolor(xg,tg,eofPred.T, vmin=-.75, vmax=0.75)
ax[1].set_ylim([interpTimeS[t1], interpTimeS[t2]])
fig.colorbar(plt1, ax=ax[1], orientation='horizontal')
ax[1].set_title('CEOF1 {:.2f}'.format(percentV[0]))
ax[1].get_yaxis().set_ticks([])

plt2 = ax[2].pcolor(xg,tg,eofPred2.T, vmin=-.85, vmax=.85)
ax[2].set_ylim([interpTimeS[t1], interpTimeS[t2]])
fig.colorbar(plt2, ax=ax[2], orientation='horizontal')
ax[2].set_title('CEOF2 {:.2f}'.format(percentV[1]))
ax[2].get_yaxis().set_ticks([])

plt3 = ax[3].pcolor(xg,tg,eofPred3.T, vmin=-.65, vmax=.65)
ax[3].set_ylim([interpTimeS[t1], interpTimeS[t2]])
ax[3].set_title('CEOF3 {:.2f}'.format(percentV[2]))
ax[3].get_yaxis().set_ticks([])

fig.colorbar(plt3, ax=ax[3], orientation='horizontal')

plt4 = ax[4].pcolor(xg,tg,eofPred4.T, vmin=-.65, vmax=.65)
ax[4].set_ylim([interpTimeS[t1], interpTimeS[t2]])
ax[4].set_title('CEOF4 {:.2f}'.format(percentV[3]))
ax[4].get_yaxis().set_ticks([])
fig.colorbar(plt4, ax=ax[4], orientation='horizontal')

plt.tight_layout(pad=0.5)

plt.show()
# plt.figure()
# plt.plot(interpX,np.nanmean(interpProfile,axis=0))
# plt.plot(interpX,np.nanpercentile(interpProfile,5,axis=0))
# plt.plot(interpX,np.nanpercentile(interpProfile,95,axis=0))
#
# plt.figure()
# # plt.pcolor(interpTime,interpX,interpProfile.T)
# plt.pcolor(highTideTimes,interpX,subaerial.T)








import math
def rotatePoint(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotateLine(origin, x, y, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    # qx = [(ox + math.cos(angle) * (x[qq] - ox) - math.sin(angle) * (y[qq] - oy)) for qq in range(len(x))]
    # qy = [(oy + math.sin(angle) * (x[qq] - ox) + math.cos(angle) * (y[qq] - oy)) for qq in range(len(x))]
    qx = [(math.cos(angle) * (x[qq] - ox) - math.sin(angle) * (y[qq] - oy)) for qq in range(len(x))]
    qy = [(math.sin(angle) * (x[qq] - ox) + math.cos(angle) * (y[qq] - oy)) for qq in range(len(x))]
    return qx, qy


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


origin = [20,-20]
angle = math.radians(5)
x = interpX
y = np.nanmean(interpProfile,axis=0)
rx, ry = rotateLine(origin,x,y,angle)

plt.figure()
p1 = plt.subplot2grid((1,2),(0,0))
p2 = plt.subplot2grid((1,2),(0,1))
p1.plot(interpX,np.nanmean(interpProfile,axis=0))
p2.plot(rx,ry)

plt.figure()
s1 = plt.subplot2grid((2,3),(0,0),)
for qq in range(len(interpTime)):
    s1.plot(interpX,interpProfile[qq,:])
nanTracker = []
for ff in range(len(interpX)):
    findnans = np.where(np.isnan(interpProfile[:,ff]))
    nanTracker.append(len(findnans[0])/len(interpTime))
s3 = plt.subplot2grid((2,3),(1,0))
s3.plot(interpX,nanTracker)
s3.set_ylim([0,1])
s2 = plt.subplot2grid((2,3),(0,1))
c = 0
for qq in range(len(interpTime)):
    origin = [0, 0]
    angle = math.radians(-2.5)
    x = interpX
    y = interpProfile[qq,:]
    rx, ry = rotateLine(origin, x, y, angle)
    if c == 0:
        rotatedYs = ry
    else:
        rotatedYs = np.vstack((rotatedYs,ry))
    c = c+1
    s2.plot(rx,ry)
nanTracker2 = []
for ff in range(len(rx)):
    findnans = np.where(np.isnan(rotatedYs[:,ff]))
    nanTracker2.append(len(findnans[0])/len(interpTime))
s4 = plt.subplot2grid((2,3),(1,1))
s4.plot(rx,nanTracker2)
s4.set_ylim([0,1])
s5 = plt.subplot2grid((2,3),(0,2))
c = 0
for qq in range(len(interpTime)):
    origin = [0, 0]
    angle = math.radians(-10)
    x = interpX
    y = interpProfile[qq,:]
    rx, ry = rotateLine(origin, x, y, angle)
    if c == 0:
        rotatedYs = ry
    else:
        rotatedYs = np.vstack((rotatedYs,ry))
    c = c+1
    s5.plot(rx,ry)
nanTracker3 = []
for ff in range(len(rx)):
    findnans = np.where(np.isnan(rotatedYs[:,ff]))
    nanTracker3.append(len(findnans[0])/len(interpTime))
s6 = plt.subplot2grid((2,3),(1,2))
s6.plot(rx,nanTracker3)
s6.set_ylim([0,1])











import datetime as dt
import numpy as np
from netCDF4 import Dataset
import os
import more_itertools as mit
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

#
# def round_to_nearest_half_hour(t):
#     if t.minute < 15:
#         return t.replace(minute=0, second=0, microsecond=0)
#     elif t.minute < 45:
#         return t.replace(minute=30, second=0, microsecond=0)
#     else:
#         return (t.replace(hour=t.hour + 1, minute=0, second=0, microsecond=0)
#                 if t.hour < 23 else t.replace(day=t.day + 1, hour=0, minute=0, second=0, microsecond=0))


#
from datetime import timedelta
def round_to_nearest_half_hour(t):

    t.replace(second=0, microsecond=0)
    delta = timedelta(minutes=int((30 - int(t.minute) % 30) % 30))

    if t.minute < 15:
        newt = t-timedelta(minutes=t.minute)
    elif t.minute > 45:
        newt = t+delta
    else:
        newt = t.replace(minute=30)
    return newt.replace(second=0, microsecond=0)

# from math import floor, ceil
# def round_to_nearest_half_hour(t):
#
#     hours = t.hour
#     minutes = t.minute
#     half_hours = hours + minutes / 60.0
#     print(half_hours)
#     rounded = (floor(half_hours * 2) if half_hours % 0.5 < 0.25 else ceil(half_hours * 2)) / 2
#     new_hour = int(rounded)
#     new_minute = int((rounded - new_hour) * 60)
#     if new_hour == 24:
#         t.replace(day= t.day+1, hour=0, minute=new_minute, second=0, microsecond=0)
#     else:
#         t.replace(hour=new_hour, minute=new_minute, second=0, microsecond=0)
#     return t

local_base = '/volumes/anderson/FRF_data/'
# local_base = 'D:/FRF_data/'

wave_base17 = 'waverider-17m/'
files17 = sorted((f for f in os.listdir(local_base+wave_base17) if not f.startswith(".")), key=str.lower) #os.listdir(local_base+wave_base17)
# files17.sort()
files_path17 = [os.path.join(os.path.abspath(local_base+wave_base17), x) for x in files17]

wave_base26 = 'waverider-26m/'
files26 = sorted((f for f in os.listdir(local_base+wave_base26) if not f.startswith(".")), key=str.lower)#os.listdir(local_base+wave_base26)
# files26.sort()
files_path26 = [os.path.join(os.path.abspath(local_base+wave_base26), x) for x in files26]

wave_base45 = 'awac-4.5m/'
files45 = sorted((f for f in os.listdir(local_base+wave_base45) if not f.startswith(".")), key=str.lower)#os.listdir(local_base+wave_base26)
# files26.sort()
files_path45 = [os.path.join(os.path.abspath(local_base+wave_base45), x) for x in files45]

def getWaves(file):
    waveData = Dataset(file)
    wave_peakdir = waveData.variables["wavePrincipleDirection"][:]
    wave_Tp = waveData.variables["waveTp"][:]
    wave_Hs = waveData.variables["waveHs"][:]
    wave_time = waveData.variables["time"][:]
    output = dict()
    output['dp'] = wave_peakdir
    output['tp'] = wave_Tp
    output['hs'] = wave_Hs
    output['time'] = wave_time
    return output

time17 = []
hs17 = []
tp17 = []
dir17 = []
for i in files_path17:
    waves = getWaves(i)
    dir17 = np.append(dir17,waves['dp'])
    hs17 = np.append(hs17,waves['hs'])
    tp17 = np.append(tp17,waves['tp'])
    time17 = np.append(time17,waves['time'].flatten())


goodInds = np.where(hs17>0)
hs17 = hs17[goodInds]
tp17 = tp17[goodInds]
dir17 = dir17[goodInds]
time17 = time17[goodInds]
import datetime as DT
tWave17 = [DT.datetime.fromtimestamp(x) for x in time17]

recentInds = np.where(np.asarray(tWave17) > DT.datetime(2016,1,1))

hs17 = hs17[recentInds]
tp17 = tp17[recentInds]
dir17 = dir17[recentInds]
tWave17 = np.asarray(tWave17)[recentInds]


timeWave17 = [round_to_nearest_half_hour(tt) for tt in tWave17]

time26 = []
hs26 = []
tp26 = []
dir26 = []
for i in files_path26:
    waves = getWaves(i)
    dir26 = np.append(dir26,waves['dp'])
    hs26 = np.append(hs26,waves['hs'])
    tp26 = np.append(tp26,waves['tp'])
    time26 = np.append(time26,waves['time'].flatten())

goodInds26 = np.where(hs26>0)
hs26 = hs26[goodInds26]
tp26 = tp26[goodInds26]
dir26 = dir26[goodInds26]
time26 = time26[goodInds26]
tWave26 = [DT.datetime.fromtimestamp(x) for x in time26]

recentInds26 = np.where(np.asarray(tWave26) > DT.datetime(2016,1,1))

hs26 = hs26[recentInds26]
tp26 = tp26[recentInds26]
dir26 = dir26[recentInds26]
tWave26 = np.asarray(tWave26)[recentInds26]

timeWave26 = [round_to_nearest_half_hour(tt) for tt in tWave26]





time45 = []
hs45 = []
tp45 = []
dir45 = []
for i in files_path26:
    waves = getWaves(i)
    dir45 = np.append(dir45,waves['dp'])
    hs45 = np.append(hs45,waves['hs'])
    tp45 = np.append(tp45,waves['tp'])
    time45 = np.append(time45,waves['time'].flatten())

goodInds45 = np.where(hs45>0)
hs45 = hs45[goodInds45]
tp45 = tp45[goodInds45]
dir45 = dir45[goodInds45]
time45 = time45[goodInds45]
tWave45 = [DT.datetime.fromtimestamp(x) for x in time45]

recentInds45 = np.where(np.asarray(tWave45) > DT.datetime(2016,1,1))

hs45 = hs45[recentInds45]
tp45 = tp45[recentInds45]
dir45 = dir45[recentInds45]
tWave45 = np.asarray(tWave45)[recentInds45]

timeWave45 = [round_to_nearest_half_hour(tt) for tt in tWave45]




from dateutil.relativedelta import relativedelta
st = dt.datetime(2016, 1, 1)
end = dt.datetime(2024,7,1)
step = relativedelta(minutes=30)
waveTimes = []
while st < end:
    waveTimes.append(st)#.strftime('%Y-%m-%d'))
    st += step



timesWithNo17 = [x for x in waveTimes if x not in timeWave17]
ind_dict17 = dict((k,i) for i,k in enumerate(waveTimes))
inter17 = set(timesWithNo17).intersection(waveTimes)
indices17 = [ind_dict17[x] for x in inter17]
indices17.sort()

timeWithout17 = np.asarray(waveTimes)[indices17]


timesWithNo17But26 = [x for x in timeWave26 if x in timeWithout17]
ind_dict = dict((k,i) for i,k in enumerate(timeWave26))
inter = set(timesWithNo17But26).intersection(timeWave26)
indices = [ind_dict[x] for x in inter]
indices.sort()





# plt.figure()
# plt.plot(timeWave17,hs17,'.')
# plt.plot(np.asarray(timeWave26)[indices],hs26[indices],'.')

combinedTime = np.hstack((np.asarray(timeWave17),np.asarray(timeWave26)[indices]))
combinedHs = np.hstack((hs17,hs26[indices]))
combinedTp = np.hstack((tp17,tp26[indices]))
combinedDp = np.hstack((dir17,dir26[indices]))

reindex = np.argsort(combinedTime)
cTime = combinedTime[reindex]
cHs = combinedHs[reindex]
cTp = combinedTp[reindex]
cDp = combinedDp[reindex]

cutOff = np.where(cTime < DT.datetime(2024,7,1))
cTime = cTime[cutOff]
cHs = cHs[cutOff]
cTp = cTp[cutOff]
cDp = cDp[cutOff]

timesWithNoC = [x for x in waveTimes if x not in cTime]
ind_dictC = dict((k,i) for i,k in enumerate(waveTimes))
interC = set(timesWithNoC).intersection(waveTimes)
indicesC = [ind_dictC[x] for x in interC]
indicesC.sort()

timeWithoutC = np.asarray(waveTimes)[indicesC]

timesWithNoCBut45 = [x for x in timeWave45 if x in timeWithoutC]
ind_dict = dict((k,i) for i,k in enumerate(timeWave45))
inter = set(timesWithNoCBut45).intersection(timeWave45)
indices2 = [ind_dict[x] for x in inter]
indices2.sort()



combinedTime45 = np.hstack((np.asarray(cTime),np.asarray(timeWave45)[indices2]))
combinedHs45 = np.hstack((cHs,hs45[indices2]))
combinedTp45 = np.hstack((cTp,tp45[indices2]))
combinedDp45 = np.hstack((cDp,dir45[indices2]))

reindex2 = np.argsort(combinedTime45)
cTime = combinedTime45[reindex2]
cHs = combinedHs45[reindex2]
cTp = combinedTp45[reindex2]
cDp = combinedDp45[reindex2]




waveNorm = cDp - 72
neg = np.where((waveNorm > 180))
waveNorm[neg[0]] = waveNorm[neg[0]]-360
offpos = np.where((waveNorm>90))
offneg = np.where((waveNorm<-90))
waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
waveNorm[offneg[0]] = waveNorm[offneg[0]]*0

# def wavetransform_point(H0, theta0, H1, theta1, T, h2, h1, g, breakcrit):



def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

avgHs = np.nanmean(cHs)
hs98 = np.nanpercentile(cHs,98)
hs95 = np.nanpercentile(cHs,95)
hs90 = np.nanpercentile(cHs,90)
hs85 = np.nanpercentile(cHs,85)
hsSmooth = moving_average(cHs,5)#np.asarray([avgHs,moving_average(hs,3),avgHs])
# stormHsInd = np.where((hsSmooth > 1.5))
stormHsInd = np.where((hsSmooth > hs85))
stormHsList = [list(group) for group in mit.consecutive_groups(stormHsInd[0])]




hsStormList = []
hsMaxStormList = []
tpStormList = []
dmStormList = []
timeStormList = []
hourStormList = []
indStormList = []
durationStormList = []
wavePowerStormList = []
longshorePowerStormList = []
startTimeStormList = []
endTimeStormList = []
averageDm = []
averageTp = []
maxStormWL = []
stormWL = []
gapBetweenStorms = 4
c = 0
while c < len(stormHsList)-2:

    i1 = stormHsList[c][0]
    i2 = stormHsList[c][-1]
    t1 = cTime[i1]
    t2 = cTime[i2]
    nexti1 = cTime[stormHsList[c+1][0]]
    diff = nexti1 - t2

    # if c+1 == (len(stormHsList)-1):
    #     print('end of the line')
    #
    # elif diff < timedelta(days=0,hours=gapBetweenStorms,seconds=0):
    #     print('we combined a storm: gap of {} to {}'.format(t2,nexti1))
    #     print('diff = {}'.format(diff))
    #
    #     i2 = stormHsList[c+1][-1]
    #     t2 = cTime[i2]
    #     nexti1 = cTime[stormHsList[c+2][0]]
    #     diff2 = nexti1-t2
    #     c = c + 1
    #     if c+2 == (len(stormHsList) - 1):
    #         print('end of the line')
    #     elif diff2 < timedelta(days=0,hours=gapBetweenStorms,seconds=0):
    #         print('we stacked 3 of them together: gap of {} to {}'.format(t2,nexti1))
    #         print('diff = {}'.format(diff2))
    #
    #         i2 = stormHsList[c + 2][-1]
    #         t2 = cTime[i2]
    #         nexti1 = cTime[stormHsList[c + 3][0]]
    #         diff3 = nexti1 - t2
    #         c = c + 1
    #
    #         if c+3 == (len(stormHsList) - 1):
    #             print('end of the line')
    #         elif diff3 < timedelta(days=0,hours=gapBetweenStorms,seconds=0):
    #             print('we stacked 4 of them together: gap of {} to {}'.format(t2,nexti1))
    #             print('diff = {}'.format(diff3))
    #
    #             i2 = stormHsList[c + 3][-1]
    #             t2 = cTime[i2]
    #             nexti1 = cTime[stormHsList[c + 4][0]]
    #             diff4 = nexti1 - t2
    #             c = c + 1
    #
    #             if c+4 == (len(stormHsList) - 1):
    #                 print('end of the line')
    #             elif diff4 < timedelta(days=0,hours=gapBetweenStorms,seconds=0):
    #                 print('we stacked 5 of them together???: gap of {} to {}'.format(t2,nexti1))
    #                 print('diff = {}'.format(diff4))
    #
    #                 i2 = stormHsList[c + 4][-1]
    #                 t2 = cTime[i2]
    #                 nexti1 = cTime[stormHsList[c + 5][0]]
    #                 diff5 = nexti1 - t2
    #                 c = c + 1
    #
    #                 if c+5 == (len(stormHsList) - 1):
    #                     print('end of the line')
    #                 elif diff5 < timedelta(days=0,hours=gapBetweenStorms,seconds=0):
    #                     print('we stacked 6 of them together?????: gap of {} to {}'.format(t2,nexti1))
    #                     print('diff = {}'.format(diff5))
    #
    #                     i2 = stormHsList[c + 5][-1]
    #                     t2 = cTime[i2]
    #                     nexti1 = cTime[stormHsList[c + 6][0]]
    #                     diff6 = nexti1 - t2
    #                     c = c + 1
    #
    #                     if c+6 == (len(stormHsList) - 1):
    #                         print('end of the line')
    #                     elif diff6 < timedelta(days=0,hours=gapBetweenStorms,seconds=0):
    #                         print('bummer, need another loop')
    c = c + 1
    tempWave = np.where((cTime <= t2) & (cTime >= t1))
    newDiff = t2-t1
    if newDiff >= timedelta(hours=12):
        # print(newDiff)
        tempWave = np.where((cTime <= t2) & (cTime >= t1))
        # tempWL = np.where((np.asarray(wltime_datetime) < t2) & (np.asarray(wltime_datetime) >= t1))
        #
        # maxStormWL.append(np.nanmax(wl_noaa[tempWL]))
        # stormWL.append(wl_noaa[tempWL])
        indices = np.arange(i1,i2)

        lwpC = 1025 * np.square(cHs[tempWave]) * cTp[tempWave] * (9.81 / (64 * np.pi)) * np.cos(
            waveNorm[tempWave] * (np.pi / 180)) * np.sin(waveNorm[tempWave] * (np.pi / 180))
        weC = np.square(cHs[tempWave]) * cTp[tempWave]

        averageDm.append(np.nanmean(cDp[tempWave]))
        averageTp.append(np.nanmean(cTp[tempWave]))

        wavePowerStormList.append(np.nansum(weC))
        longshorePowerStormList.append(np.nansum(lwpC))
        hsStormList.append(cHs[tempWave])
        hsMaxStormList.append(np.nanmax(cHs[tempWave]))
        tpStormList.append(cTp[tempWave])
        dmStormList.append(cDp[tempWave])
        timeStormList.append(cTime[tempWave])


        # duration = (int(t2.strftime('%s'))-int(t1.strftime('%H')))/60/60

        duration = (t2.timestamp()-t1.timestamp())/60/60
        durationStormList.append(duration)
        indStormList.append(indices)
        # startTimeStormList.append(int(t1.strftime('%S')))
        # endTimeStormList.append(int(t2.strftime('%S')))
        startTimeStormList.append(t1.timestamp())
        endTimeStormList.append(t2.timestamp())



# Let's check to see storm proximity...

separation = []
for qq in range(len(endTimeStormList)-1):
    diff = startTimeStormList[qq+1] - endTimeStormList[qq]
    separation.append(diff)



startLidarProfile = []
startLidarTime = []
startLidarIndex = []
startLidarContour = []
endLidarProfile = []
endLidarTime = []
endLidarIndex = []
endLidarContour = []
for qq in range(len(timeStormList)):
    tempStartDate = timeStormList[qq][0]
    # get all differences with date as values
    cloz_dictStart = {
        abs(tempStartDate.timestamp() - date.timestamp()): date
        for date in dts}
    # extracting minimum key using min()
    resStart = cloz_dictStart[min(cloz_dictStart.keys())]

    startInd = np.where(min(cloz_dictStart.keys())== np.array(list(cloz_dictStart.keys())))[0][0]

    tempEndDate = timeStormList[qq][-1]
    # get all differences with date as values
    cloz_dictEnd = {
        abs(tempEndDate.timestamp() - date.timestamp()): date
        for date in dts}
    # extracting minimum key using min()
    resEnd = cloz_dictEnd[min(cloz_dictEnd.keys())]

    endInd = np.where(min(cloz_dictEnd.keys())== np.array(list(cloz_dictEnd.keys())))[0][0]

    startLidarTime.append(resStart)
    startLidarProfile.append(lidarProfiles[startInd,:])
    startLidarContour.append(lidarContours[2,startInd])
    startLidarIndex.append(startInd)
    endLidarTime.append(resEnd)
    endLidarProfile.append(lidarProfiles[endInd,:])
    endLidarContour.append(lidarContours[2,endInd])
    endLidarIndex.append(endInd)





from sklearn.cluster import KMeans
num_clusters = 15

PCsub = np.asarray([np.asarray(wavePowerStormList),np.asarray(durationStormList),np.asarray(hsMaxStormList),np.asarray(averageDm)]).T
badInds = np.where(np.isnan(PCsub))
PCsub[badInds] = 0

kma = KMeans(n_clusters=num_clusters, n_init=500).fit(PCsub)
# groupsize
_, group_size = np.unique(kma.labels_, return_counts=True)
# groups
d_groups = {}
for k in range(num_clusters):
    d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)

bmus=kma.labels_





import matplotlib.cm as cm
profileColors = cm.rainbow(np.linspace(0, 1,num_clusters))
plt.figure()
plot1 = plt.subplot2grid((2,2),(0,0),colspan=2)
plot1.plot(cTime,cHs,color='k')
# plot1.plot(cTime[2:-2],hsSmooth)
for qq in range(len(hsStormList)):
    print('bmus = {}'.format(bmus[qq]))
    plot1.plot(timeStormList[qq],hsStormList[qq],'-',color=profileColors[bmus[qq],:])
plot1.set_ylabel('Hs (m)')

# plot2 = plt.subplot2grid((2,2),(1,0),colspan=1)
# sc1 = plot2.scatter(durationStormList,hsMaxStormList,10,wavePowerStormList,vmin=np.min(wavePowerStormList),vmax=25000)
# plot2.set_xlim([8,150])
# plot2.set_ylabel('Max Storm $H_{s}$ (m)')
# plot2.set_xlabel('Storm Duration (hr)')
# c1 = plt.colorbar(sc1,ax=plot2)
# c1.set_label('Cumulative Storm Wave Power')
plot2 = plt.subplot2grid((2,2),(1,0),colspan=1)
sc1 = plot2.scatter(durationStormList,hsMaxStormList,10,startLidarContour,vmin=np.nanmin(startLidarContour),vmax=np.nanmax(startLidarContour))
plot2.set_xlim([8,150])
plot2.set_ylabel('Max Storm $H_{s}$ (m)')
plot2.set_xlabel('Storm Duration (hr)')
c1 = plt.colorbar(sc1,ax=plot2)
c1.set_label('Start Storm Shoreline')

plot3 = plt.subplot2grid((2,2),(1,1),colspan=1)
sc2 = plot3.scatter(durationStormList,hsMaxStormList,10,bmus,cmap='rainbow')
plot3.set_xlim([8,150])
plot3.set_ylabel('Max Storm $H_{s}$ (m)')
plot3.set_xlabel('Storm Duration (hr)')
c2 = plt.colorbar(sc2,ax=plot3)
c2.set_label('Cluster')



#
# from hexalattice.hexalattice import *
# hex_centers, _ = create_hex_grid(nx=5,
#                                  ny=5,
#                                  do_plot=True)
# tile_centers_x = hex_centers[:, 0]
# tile_centers_y = hex_centers[:, 1]


#
# plt.figure()
# plot1 = plt.subplot2grid((2,1),(0,0))
# plot1.plot(cTime,cHs,'k')
#
# for qq in range(len(hsStormList)):
#     plot1.plot(timeStormList[qq],hsStormList[qq],'.',color='orange')
# plot1.set_ylabel('Hs (m)')
# plot2 = plt.subplot2grid((2,1),(1,0))
# # plot2.plot(cTime,cHs)
# # for qq in range(len(hsStormList)):
# plot2.scatter(startTimeStormList,wavePowerStormList)
# # plot2.ylabel('Hs (m)')

import matplotlib.cm as cm

#
# plt.figure()
# p2 = plt.subplot2grid((1,1),(0,0))
# p2.pcolor(dayTime,lidar_xFRF,dailyAverage.T)
# # for qq in range(len(hsStormList)):
# #     p2.plot([timeStormList[qq][0],timeStormList[qq][0]],[50,200],'--',color='black')
# sc = p2.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList]),np.asarray(200*np.ones((len(startTimeStormList),))),s=np.asarray(wavePowerStormList)/100,c=np.asarray(startLidarContour)-np.asarray(endLidarContour),cmap=cm.plasma,vmin=-10,vmax=10)












plt.figure()
p1 = plt.subplot2grid((3,2),(0,1))
# p1.fill_between(lidar_xFRF,np.nanmean(dailyAverage,axis=0)-np.nanstd(dailyAverage,axis=0),np.nanmean(dailyAverage,axis=0)+np.nanstd(dailyAverage,axis=0))
# p1.fill_between(lidar_xFRF,np.nanpercentile(dailyAverage,1,axis=0),np.nanpercentile(dailyAverage,99,axis=0),color=[0.65,0.65,0.65])
# p1.plot(lidar_xFRF,np.nanmean(dailyAverage,axis=0),color='k')
p1.fill_between(lidar_xFRF,np.nanpercentile(lidarProfiles,0.5,axis=0),np.nanpercentile(lidarProfiles,99.5,axis=0),color=[0.65,0.65,0.65])
p1.plot(lidar_xFRF,np.nanmean(lidarProfiles,axis=0),color='k')

p1.set_xlim([50,150])
p1.set_xlabel('xFRF (m)')
p1.set_ylabel('Elev. (NAVD88, m)')
# p2.pcolor(dayTime,lidar_xFRF,dailyAverage.T)

p2 = plt.subplot2grid((3,2),(1,0),colspan=2)
p2.plot(cTime,cHs,'k')
p2.set_ylabel('Hs (m)')
p2.set_xlim([cTime[0],cTime[-1]])
# p3 = plt.subplot2grid((3,2),(2,0),colspan=2)
# # for qq in range(len(hsStormList)):
# #     p2.plot([timeStormList[qq][0],timeStormList[qq][0]],[50,200],'--',color='black')
# # sc = p3.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList]),np.asarray(200*np.ones((len(startTimeStormList),))),s=np.asarray(wavePowerStormList)/100,c=np.asarray(startLidarContour)-np.asarray(endLidarContour),cmap=cm.plasma,vmin=-10,vmax=10)
# sc = p3.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList]),np.asarray(startLidarContour),s=np.asarray(wavePowerStormList)/100,c=(np.asarray(startLidarContour)-np.asarray(endLidarContour)),cmap=cm.plasma,vmin=-2,vmax=12)
#
# p3.set_ylabel('Pre-storm MHW (xFRF, m)')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Wave Power')
# cbaxes = inset_axes(p3, width="30%", height="3%", loc=1)
# cb = plt.colorbar(sc,cax=cbaxes,orientation='horizontal')
# cb.set_ticks([-2,12])
# cb.set_ticklabels([-2,12])
# cb.set_label('Cross-shore Change During Storm (m)')
p3 = plt.subplot2grid((3,2),(2,0),colspan=2)
sc2 = p3.pcolor(dayTime,lidar_xFRF,dailyAverage.T)
p3.set_xlabel('xFRF (m)')
p3.set_ylabel('yFRF (m)')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Wave Power')
cbaxes = inset_axes(p3, width="30%", height="3%", loc=1)
cb = plt.colorbar(sc2,cax=cbaxes,orientation='horizontal')
cb.set_ticks([0,5])
cb.set_ticklabels([0,5])
cb.set_label('Elevation (NAVD88, m)')





fig2 = plt.figure()
p2a = plt.subplot2grid((2,1),(0,0))
sc2 = p2a.pcolor(dayTime,lidar_xFRF,dailyAverage.T)
p2a.set_xlabel('xFRF (m)')
p2a.set_ylabel('yFRF (m)')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Wave Power')
cbaxes = inset_axes(p2a, width="30%", height="3%", loc=1)
cb = plt.colorbar(sc2,cax=cbaxes,orientation='horizontal')
cb.set_ticks([0,5])
cb.set_ticklabels([0,5])
cb.set_label('Elevation (NAVD88, m)')

p3a = plt.subplot2grid((2,1),(1,0))
sc3 = p3a.pcolor(dayTime,lidar_xFRF,dailyStd.T)
p3a.set_xlabel('xFRF (m)')
p3a.set_ylabel('yFRF (m)')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Wave Power')
cbaxes = inset_axes(p3a, width="30%", height="3%", loc=1)
cb = plt.colorbar(sc3,cax=cbaxes,orientation='horizontal')
# cb.set_ticks([0,5])
# cb.set_ticklabels([0,5])
# cb.set_label('std (m)')


