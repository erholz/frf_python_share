import pickle
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load the dataframe
import os
from datetime import datetime
from funcs.align_data_time import align_data_fullspan





# Load temporally aligned data - need to add lidarelev_fullspan
picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
# picklefile_dir = './'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)

# Load and process wave data
Hs_8m_fullspan = data_wave8m[:,0]
Tp_8m_fullspan = data_wave8m[:,1]
dir_8m_fullspan = data_wave8m[:,2]
Hs_17m_fullspan = data_wave8m[:,0]
Tp_17m_fullspan = data_wave8m[:,1]
dir_17m_fullspan = data_wave8m[:,2]

fig, ax = plt.subplots()
ax.plot(time_fullspan,)


cuspfile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/cusp_presence/'
with open(cuspfile_dir+'cuspTimes.pickle', 'rb') as file:
    datload = pickle.load(file)
cusp_time = np.array(datload['timeCusps'])
cusp_presence = np.ones(shape=cusp_time.shape)

cusp_fullspan = np.empty(shape=time_fullspan.shape)
cusp_fullspan[:] = 0
for tt in np.arange(cusp_time.size):
    iiclose = np.where(abs(cusp_time[tt] - time_fullspan) == np.nanmin(abs(cusp_time[tt] - time_fullspan)))[0]
    if iiclose.size > 1:
        iiclose = iiclose[0]
    cusp_fullspan[iiclose] = 1



fig, ax = plt.subplots()
tplot = pd.to_datetime(cusp_time, unit='s', origin='unix')
ax.plot(cusp_time,cusp_presence,'x')
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
ax.plot(time_fullspan,cusp_fullspan,'+')

