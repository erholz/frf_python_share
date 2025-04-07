import pickle
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd  # to load the dataframe
import os
from datetime import datetime
from funcs.align_data_time import align_data_fullspan
from funcs.create_contours import *
from funcs.wavefuncs import *


## Load temporally aligned data - need to add lidarelev_fullspan
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
with open(picklefile_dir+'set_id_tokeep_14Dec2024.pickle', 'rb') as file:
    set_id_tokeep, plot_start_iikeep = pickle.load(file)
# picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
# picklefile_dir = './'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,_,_,_,_,_,_,_,_,_,_,_,_ = pickle.load(file)
# with open(picklefile_dir+'bathylidar_combo.pickle','rb') as file:
#     lidar_xFRF,bathylidar_combo = pickle.load(file)
with open(picklefile_dir+'waves_8m&17m_2015_2024.pickle','rb') as file:
    [data_wave8m,data_wave17m,data_wave8m_filled] = pickle.load(file)
with open(picklefile_dir+'stormHs95_Over12Hours.pickle','rb') as f:
    output = pickle.load(f)
# list(output)
storm_start = np.array(output['startTimeStormList'])
storm_end = np.array(output['endTimeStormList'])
storm_startWIS = np.array(output['startTimeStormListWIS'])
storm_endWIS = np.array(output['endTimeStormListWIS'])
all_Hs = np.array(output['cHs'])
all_time = np.empty(shape=all_Hs.shape)
all_time[:] = np.nan
for tt in np.arange(all_Hs.size):
    all_time[tt] = output['cTime'][tt].timestamp()


# # find all the times where we are going to subset ML model data from...
# Nlook = 4*24
# time_dummy = np.empty(shape=time_fullspan.shape)
# time_dummy[:] = -1
# for jj in plot_start_iikeep[set_id_tokeep]:
#     iigrab = np.arange(jj,jj+Nlook).astype(int)
#     time_dummy[iigrab] = 1
# # find breaks in time_dummy
# [H,T,crossUP] = upcross(time_dummy,time_fullspan)
# [H,T,crossDOWN] = upcross(-1*time_dummy,time_fullspan)
# fig, ax = plt.subplots()
# ax.plot(time_fullspan,time_dummy,'.')
# ax.plot(time_fullspan[crossUP],time_dummy[crossUP],'^')
# ax.plot(time_fullspan[crossDOWN],time_dummy[crossDOWN],'v')

# Actually, what we want is the total times available (i think...?)

## Remove storms outside of general time of interest
storm_start = storm_start[(storm_start >= time_fullspan[0]) & (storm_start < time_fullspan[-1])]
storm_end = storm_end[(storm_end > time_fullspan[0]) & (storm_end <= time_fullspan[-1])]
storm_startWIS = storm_startWIS[(storm_startWIS >= time_fullspan[0]) & (storm_startWIS < time_fullspan[-1])]
storm_endWIS = storm_endWIS[(storm_endWIS > time_fullspan[0]) & (storm_endWIS <= time_fullspan[-1])]

## Start by identifing times when FRF or WIS data says stormy
storm_flag = np.empty(shape=time_fullspan.shape)        # BINARY - stormy == 1, calm/non-stormy = nan
storm_flag[:] = 0
for jj in np.arange(len(storm_start)):
    tt_during_storm = (time_fullspan >= storm_start[jj]) & (time_fullspan <= storm_end[jj])
    storm_flag[tt_during_storm] = 1
for jj in np.arange(len(storm_startWIS)):
    tt_during_storm = (time_fullspan >= storm_startWIS[jj]) & (time_fullspan <= storm_endWIS[jj])
    storm_flag[tt_during_storm] = 1
storm_timeend_all = []
storm_timestart_all = []
storm_iiend_all = []
storm_iistart_all = []
storm_flag[storm_flag == 0] = -1
iicross = np.where(storm_flag[1:]*storm_flag[0:-1] < 0)[0]
for jj in np.arange(iicross.size):
    if (storm_flag[iicross[jj]] == -1) & (storm_flag[iicross[jj]+1] == 1):
        storm_timestart_all = np.append(storm_timestart_all,time_fullspan[iicross[jj]])
        storm_iistart_all = np.append(storm_iistart_all,int(iicross[jj]+1))
    elif (storm_flag[iicross[jj]] == 1) & (storm_flag[iicross[jj]+1] == -1):
        storm_timeend_all = np.append(storm_timeend_all, time_fullspan[iicross[jj]])
        storm_iiend_all = np.append(storm_iiend_all,int(iicross[jj]+1))
    else:
        print('help')


# plot both Hs time series and non-storm sample times
h94 = 2.1
h95 = 2.2
tplot = pd.to_datetime(time_fullspan.astype(int),unit='s',origin='unix')
fig, ax = plt.subplots()
ax.plot(all_time,all_Hs,'k',linewidth=0.1)
ax.plot(all_time,np.zeros(all_time.size,)+h94,'r')
yplot = h94 + np.zeros(storm_end.size,)
ax.plot(storm_start,yplot,'^')
ax.plot(storm_end,yplot,'v')
# fig, ax = plt.subplots()
for jj in np.arange(storm_timeend_all.size-1):
    # tmpwidth = time_fullspan[crossDOWN[jj]] - time_fullspan[crossUP[jj]]
    # tmpwidth = time_fullspan[crossDOWN[jj+1]] - time_fullspan[crossUP[jj+1]]
    tmpwidth = storm_timestart_all[jj+1] - storm_timeend_all[jj]
    if tmpwidth < 0:
        print('negative tmpwidth for jj ='+str(jj))
    # left, bottom, width, height = (time_fullspan[crossDOWN[jj]], 0, tmpwidth, 99999999)
    left, bottom, width, height = (storm_timeend_all[jj], 0, tmpwidth, 99999999)
    exec('patch'+str(jj)+' = plt.Rectangle((left, bottom), width, height, alpha=0.1, color="m")')
    exec('ax.add_patch(patch'+str(jj)+')')
ax.set_xlim(time_fullspan[0],time_fullspan[-1])
ax.set_ylim(0,7)

fig, ax = plt.subplots()
ax.plot(time_fullspan[crossUP],np.zeros(crossUP.size,),'+')
ax.plot(time_fullspan[crossDOWN],np.zeros(crossDOWN.size,),'x')





