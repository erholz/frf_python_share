import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mat73
#
#
# local_base = 'D:/FRF_data/'
local_base = '/volumes/anderson/FRF_data/'
# DEFINE SUBDIR WITH NOAA WATERLEVEL FILES
demfloc = local_base + 'dune_lidar/lidar_dems/'
demext = 'nc'  # << change not recommended; defines file type to look for


import os
def find_files_local(floc,ext_in):
    full_path = floc
    ids = []
    allFiles = os.listdir(full_path)
    allFiles.sort()
    for file in allFiles:
        if file.startswith('._'):
            print('skipping a hidden file')
        elif file.endswith(ext_in):
            ids.append(file)
    return ids

fname_in_range = find_files_local(demfloc,demext)


from netCDF4 import Dataset

import datetime as DT

alongshoreAverage = []
alongshoreStd = []
profileTime = []

for fname_ii in fname_in_range:
    print('reading... ' + fname_ii)
    full_path = demfloc + fname_ii
    ## Lidar dataset
    ds = Dataset(full_path, "r")
    # qaqc_fac = ds.variables["beachProfileQCFlag"][:]
    # lidar_pmissing = ds.variables["percentTimeSeriesMissing"][:, :]
    lidar_elev = ds.variables["elevation"][:,:,:]
    # lidar_elevstd = ds.variables["elevationSigma"][:]
    lidar_time = ds.variables["time"][:]
    lidar_xFRF = ds.variables["xFRF"][:].filled(fill_value=np.NaN)
    xs = lidar_xFRF
    lidar_yFRF = ds.variables["yFRF"][:].filled(fill_value=np.NaN)
    # Use only 200m of 500m DEM to focus on area with best data coverage
    idx_y = np.arange(150, 251)  # MATLAB is 1-indexed, Python is 0-indexed
    ysC = lidar_yFRF[idx_y]
    for hh in range(len(lidar_time)):
        temp = np.nanmean(lidar_elev[hh,idx_y, :].filled(fill_value=np.NaN),axis=0)
        badVals = np.where(np.isnan(temp))
        temp = np.delete(temp,badVals)
        if len(temp) > 7:
            alongshoreAverage.append(np.nanmean(lidar_elev[hh,idx_y, :].filled(fill_value=np.NaN),axis=0))
            alongshoreStd.append(np.nanstd(lidar_elev[hh,idx_y, :].filled(fill_value=np.NaN),axis=0))
            profileTime.append(lidar_time[hh])
        else:
            print('found a profile that was only {}'.format(len(temp)))


cont_elev = np.arange(-0.25,2.25,0.25)    # <<< MUST BE POSITIVELY INCREASING


from funcs.create_contours import *
# run file create_contours.py
elev_input = np.asarray(alongshoreAverage)[:,0:]
lidartime = np.asarray(profileTime)
lidar_xFRF = lidar_xFRF[0:]
cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)


clusterPickle = 'alongshoreAverages.pickle'
output = {}
output['profileTime'] = profileTime
output['lidar_yFRF'] = lidar_yFRF
output['lidar_xFRF'] = lidar_xFRF
output['ysC'] = ysC
output['alongshoreAverage'] = alongshoreAverage
output['alongshoreStd'] = alongshoreStd
output['cont_elev'] = cont_elev
output['cont_ts'] = cont_ts
output['cmean'] = cmean
output['cstd'] = cstd

import pickle
with open(clusterPickle,'wb') as f:
    pickle.dump(output, f)


# plt.figure()
# plt.pcolor(np.asarray(profileTime),lidar_xFRF,np.asarray(alongshoreAverage)[:-1,:-1].T)



