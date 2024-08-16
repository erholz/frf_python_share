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
local_base = 'D:/FRF_data/'
# local_base = '/volumes/macDrive/FRF_data/'

# DEFINE TIME PERIOD OF INTEREST
time_beg = '2016-01-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2024-07-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4

# DEFINE CONTOUR ELEVATIONS OF INTEREST
cont_elev = np.arange(-0.25,4.25,0.5)    # <<< MUST BE POSITIVELY INCREASING

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
# plt.figure()
# plt.pcolor(lidarTime,lidar_xFRF,lidarProfiles.T)
# dts = [dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in lidarTime]
dts = [dt.datetime.utcfromtimestamp(ts) for ts in lidarTime]
# plt.figure()
# plt.plot(dts,lidarContours[4,:],'.')
# plt.plot(dts,lidarContours[6,:],'.')
# plt.plot(dts,lidarContours[8,:],'.')

import datetime as dt
from dateutil.relativedelta import relativedelta

st = dt.datetime(2016,1,1)
# end = dt.datetime(2021,12,31)
end = dt.datetime(2024,7,1)
step = relativedelta(days=1)
dayTime = []
while st < end:
    dayTime.append(st)#.strftime('%Y-%m-%d'))
    st += step


def find_files_local(floc,ext_in):
    full_path = floc
    ids = []
    for file in os.listdir(full_path):
        if file.endswith(ext_in):
            ids.append(file)
    return ids

#start with NOAA water level files
floc = noaawlfloc
ext = noaawlext
fname_in_range = find_files_local(floc,ext)#find_files_in_range(floc,ext,epoch_beg,epoch_end, tzinfo)
wltime_noaa = []
wl_noaa = []
for fname_ii in fname_in_range:
    print('reading... ' + fname_ii)
    full_path = floc + fname_ii
    waterlevel_noaa, time_noaa = getlocal_waterlevels(full_path)
    wltime_noaa = np.append(wltime_noaa, time_noaa)
    wl_noaa = np.append(wl_noaa, waterlevel_noaa)


from dateutil.relativedelta import relativedelta
st = dt.datetime(2016, 1, 1)
end = dt.datetime(2024,7,1)
step = relativedelta(hours=12.5)
wlTimes = []
while st < end:
    wlTimes.append(st)#.strftime('%Y-%m-%d'))
    st += step

import datetime as DT
tWL = np.asarray([DT.datetime.fromtimestamp(x) for x in wltime_noaa])

highTideIndices = []
highTideTimes = []
highTides = []
for qq in range(len(wlTimes)-1):
    inder = np.where((tWL > wlTimes[qq]) & (tWL < wlTimes[qq+1]))
    if len(inder[0]) > 0:
        if np.isnan(np.nanmax(wl_noaa[inder])):
            print('nan water levels on {}'.format(wlTimes[qq]))
        else:
            subsetWLind = np.nanargmax(wl_noaa[inder])
            subsetTime = wltime_noaa[inder]
            highTideTimes.append(subsetTime[subsetWLind])
            highTides.append(np.nanmax(wl_noaa[inder]))
    else:
        print('no water levels on {}'.format(wlTimes[qq]))

highTideTimes = np.asarray([DT.datetime.fromtimestamp(x) for x in highTideTimes])


import matplotlib.pyplot as plt
plt.figure()
plt.plot(tWL,wl_noaa)
plt.plot(highTideTimes,highTides,'r*')



tidalAverage = np.nan * np.ones((len(highTideTimes),len(lidar_xFRF)))
tidalStd = np.nan * np.ones((len(highTideTimes),len(lidar_xFRF)))

for qq in range(len(highTideTimes)-1):
    inder = np.where((np.asarray(dts)>=np.asarray(highTideTimes)[qq]) & (np.asarray(dts) <=np.asarray(highTideTimes)[qq+1]))
    if len(inder[0])>0:
        tidalAverage[qq,:] = np.nanmean(lidarProfiles[inder[0],:],axis=0)
        tidalStd[qq,:] = np.nanstd(lidarProfiles[inder[0],:],axis=0)



# DEFINE CONTOUR ELEVATIONS OF INTEREST
cont_elev = np.arange(-0.25,4.25,0.5)    # <<< MUST BE POSITIVELY INCREASING
# run file create_contours.py
tidalContours, tidalCmean, tidalCstd = create_contours(tidalAverage,highTideTimes,lidar_xFRF,cont_elev)



clusterPickle = 'tidalLidarAverages.pickle'
output = {}
output['tidalAverage'] = tidalAverage
output['tidalStd'] = tidalStd
output['highTideTimes'] = highTideTimes
output['highTides'] = highTides
output['tWL'] = tWL
output['wl_noaa'] = wl_noaa
output['wlTimes'] = wlTimes
output['st'] = dts
output['lidarTime'] = lidarTime
output['lidarProfiles'] = lidarProfiles
output['lidarContours'] = lidarContours
output['cont_elev'] = cont_elev
output['time_beg'] = time_beg
output['time_end'] = time_end
output['wlTimes'] = wlTimes
output['cmean'] = cmean
output['cstd'] = cstd
output['dayTime'] = dayTime
output['lidar_xFRF'] = lidar_xFRF
output['cont_elev'] = cont_elev
output['tidalContours'] = tidalContours
output['tidalCmean'] = tidalCmean
output['tidalCstd'] = tidalCstd

import pickle
with open(clusterPickle,'wb') as f:
    pickle.dump(output, f)

plt.figure()
p3 = plt.subplot2grid((1,1),(0,0))
sc2 = p3.pcolor(highTideTimes,lidar_xFRF,tidalAverage.T)
p3.set_xlabel('xFRF (m)')
p3.set_ylabel('yFRF (m)')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Wave Power')
cbaxes = inset_axes(p3, width="30%", height="3%", loc=1)
cb = plt.colorbar(sc2,cax=cbaxes,orientation='horizontal')
cb.set_ticks([0,5])
cb.set_ticklabels([0,5])
cb.set_label('Elevation (NAVD88, m)')

# ideas...
# extract contours of the averaged profiles
# find the dune contour and subtract everything to match that
# Require a minimum width and find all profiles that are within that
#
plt.figure()
plt.plot(highTideTimes,tidalContours[2,:])

