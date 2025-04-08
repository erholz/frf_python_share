import pickle
import numpy as np


years = np.arange(2016,2025)

timeCusps = []
for hh in years:
    filename = "cuspsOverTime{}.pickle".format(hh)
    with open(filename, "rb") as input_file:
        inputCusps = pickle.load(input_file)
    maxSjFinal = inputCusps['maxSjFinal']
    thresholdAtMaxSj = inputCusps['thresholdAtMaxSj']
    contourTime = inputCusps['contourTime']
    for qq in range(len(maxSjFinal)):
        if maxSjFinal[qq] > thresholdAtMaxSj[qq]:
            timeCusps.append(contourTime[qq])


#
# import matplotlib.pyplot as plt
# import numpy as np
# import datetime as dt
# from run_lidarcollect import *
# from run_hydrocollect import *
# from funcs.create_contours import *
# from funcs.lidar_check import *
# from funcs.calculate_beachvol import *
# from funcs.lidar_fillgaps import *
# from run_makeplots import *
# import pickle
# import os
# # DEFINE WHERE FRF DATA FILES ARE LOCATED
# # local_base = 'D:/FRF_data/'
# local_base = '/volumes/anderson/FRF_data/'
#
# # DEFINE TIME PERIOD OF INTEREST
# time_beg = '2016-01-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
# time_end = '2024-10-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
# tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4
#
# # DEFINE CONTOUR ELEVATIONS OF INTEREST
# cont_elev = np.arange(-0.25,3.25,0.25)    # <<< MUST BE POSITIVELY INCREASING
#
# # DEFINE NUMBER OF PROFILES TO PLOT
# num_profs_plot = 15
#
# # DEFINE SUBDIR WITH LIDAR FILES
# lidarfloc = local_base + 'dune_lidar/lidar_transect/'
# lidarext = 'nc'  # << change not recommended; defines file type to look for
#
# # DEFINE SUBDIR WITH NOAA WATERLEVEL FILES
# noaawlfloc = local_base + 'waterlevel/'
# noaawlext = 'nc'  # << change not recommended; defines file type to look for
#
# # DEFINE SUBDIR WITH LIDAR HYDRO FILES
# lidarhydrofloc = local_base + 'waves_lidar/lidar_hydro/'
# lidarhydroext = 'nc'  # << change not recommended; defines file type to look for
#
#
#
#
# # -------------------- BEGIN RUN_CODE.PY --------------------
#
# # convert period of interest to datenum
# time_format = '%Y-%m-%dT%H:%M:%S'
# epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
# epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
# TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)
# # Save timing variables
# with open('timeinfo.pickle','wb') as file:
#     pickle.dump([tzinfo,time_format,time_beg,time_end,epoch_beg,epoch_end,TOI_duration], file)
#
# # run file run_lidarcollect.py
# lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext)
#
# # Remove weird data (first order filtering)
# stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
# pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
# tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
# lidarelev[tmpii] = np.nan
#
# # run file create_contours.py
# elev_input = lidarelev
# cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)
#
#
# lidarTime = lidartime
# lidarProfiles = lidarelev
# lidarContours = cont_ts
# # dts = [dt.datetime.utcfromtimestamp(ts) for ts in lidarTime]
# dts = [dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in lidarTime]
# dates = [dt.datetime.utcfromtimestamp(ts) for ts in lidarTime]
# cuspsDates = [dt.datetime.utcfromtimestamp(ts) for ts in timeCusps]
#
#
# import datetime as dt
# from dateutil.relativedelta import relativedelta
#
# st = dt.datetime(2016,1,1)
# # end = dt.datetime(2021,12,31)
# end = dt.datetime(2024,7,1)
# step = relativedelta(days=1)
# dayTime = []
# while st < end:
#     dayTime.append(st)#.strftime('%Y-%m-%d'))
#     st += step
# dailyAverage = np.nan * np.ones((len(dayTime),len(lidar_xFRF)))
# dailyStd = np.nan * np.ones((len(dayTime),len(lidar_xFRF)))
#
# for qq in range(len(dayTime)-1):
#     inder = np.where((np.asarray(dates)>=np.asarray(dayTime)[qq]) & (np.asarray(dates) <=np.asarray(dayTime)[qq+1]))
#     if len(inder[0])>0:
#         dailyAverage[qq,:] = np.nanmean(lidarProfiles[inder[0],:],axis=0)
#         dailyStd[qq,:] = np.nanstd(lidarProfiles[inder[0],:],axis=0)
#
#
#
# plt.figure()
# # plt.pcolor(dates,lidar_xFRF,lidarProfiles.T)
# plt.pcolor(dayTime,lidar_xFRF,dailyAverage.T)
#
# plt.plot(dates,lidarContours[5,:],'.',color='red')
#
# plt.plot(cuspsDates,190*np.ones((len(np.asarray(timeCusps)))),'.',color='orange')
# plt.xlabel('Time')
# plt.ylabel('xFRF (m)')
# plt.colorbar()
# # plt.plot(dts,lidarContours[6,:],'.')
# # plt.plot(dts,lidarContours[8,:],'.')
#
#


clusterPickle = 'cuspTimes.pickle'
output = {}
# output['cuspsDates'] = cuspsDates
# output['contourDates'] = dates
# output['lidarContours'] = lidarContours
# output['dayTime'] = dayTime
# output['lidar_xFRF'] = lidar_xFRF
# output['dailyAverage'] = dailyAverage
# output['lidarProfiles'] = lidarProfiles
# output['dailyStd'] = dailyStd
# output['lidarTime'] = lidarTime
# output['lidarProfiles'] = lidarProfiles
# output['cont_elev'] = cont_elev
output['timeCusps'] = timeCusps

import pickle
with open(clusterPickle,'wb') as f:
    pickle.dump(output, f)
