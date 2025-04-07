import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd  # to load the dataframe
import datetime as dt
from datetime import timezone
import numpy as np
from netCDF4 import Dataset
import os
import more_itertools as mit
import pickle
from datetime import timedelta


################## LOAD DATA ##################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'
with open(picklefile_dir+'waves_WISandFRF.pickle', 'rb') as file:
    combinedHsWIS,combinedTpWIS,combinedDmWIS,combinedTimeWIS = pickle.load(file)
with open(picklefile_dir+'waves_FRF.pickle','rb') as file:
    cHs,cTp,cDp,cTime = pickle.load(file)
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,_,_,_,_,_,_,_,_,_,_,_,_ = pickle.load(file)


################## DEFINE STORM ##################

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

avgHs = np.nanmean(combinedHsWIS)
hs98 = np.nanpercentile(combinedHsWIS,98)
hs95 = np.nanpercentile(combinedHsWIS,95)
hs94 = np.nanpercentile(combinedHsWIS,94)
hs90 = np.nanpercentile(combinedHsWIS,90)
hs85 = np.nanpercentile(combinedHsWIS,85)

hsSmooth = moving_average(cHs,3)#np.asarray([avgHs,moving_average(hs,3),avgHs])
hsSmoothWIS = moving_average(combinedHsWIS,3)#np.asarray([avgHs,moving_average(hs,3),avgHs])
stormHsIndWIS = np.where((hsSmoothWIS > hs94))
stormHsListWIS = [list(group) for group in mit.consecutive_groups(stormHsIndWIS[0])]
stormHsInd = np.where((hsSmooth > hs94))
stormHsList = [list(group) for group in mit.consecutive_groups(stormHsInd[0])]


################## GET STORM STATISTICS FROM FRF DATA ##################

stormLengths = np.asarray([len(tt) for tt in stormHsList])
over12HourStorms = np.where(stormLengths>=23)
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
peakTimeStormList = []
startTimeStormList = []
endTimeStormList = []
c = 0
# for yy in range(len(over12HourStorms[0])):
for yy in np.arange(len(over12HourStorms[0])-1):

    index = over12HourStorms[0][yy]
    i1 = stormHsList[index][0]
    i2 = stormHsList[index][-1]
    t1 = cTime[i1]
    t2 = cTime[i2]

    # should we check if we need to add storm waves onto this?
    nexti1 = cTime[stormHsList[index+1][0]]
    # diff = (int(nexti1.strftime('%s'))-int(t2.strftime('%s')))/60/60#nexti1 - t2
    diff = ((nexti1-dt.datetime(1970,1,1)).total_seconds() - (t2-dt.datetime(1970,1,1)).total_seconds() )/3600

    if diff < 12:#timedelta(hours=12):
        print('Next storm is within 12 hours')
        t2 = cTime[stormHsList[index+1][-1]]

    previousi2 = cTime[stormHsList[index-1][-1]]
    # diff2 = (int(t1.strftime('%s'))-int(previousi2.strftime('%s')))/60/60#t1 - previousi2
    diff2 = ((t1-dt.datetime(1970,1,1)).total_seconds() - (previousi2-dt.datetime(1970,1,1)).total_seconds() )/3600

    if diff2 < 12:#timedelta(hours=12):
        print('Previous storm was within 12 hours')
        t1 = cTime[stormHsList[index-1][0]]

    c = c + 1
    tempWave = np.where((cTime <= t2) & (cTime >= t1))
    newDiff = t2-t1
    if newDiff >= timedelta(hours=12):
        # print(newDiff)
        tempWave = np.where((cTime <= t2) & (cTime >= t1))
        indices = np.arange(i1,i2)

        lwpC = 1025 * np.square(cHs[tempWave]) * cTp[tempWave] * (9.81 / (64 * np.pi)) * np.cos(
            cDp[tempWave] * (np.pi / 180)) * np.sin(cDp[tempWave] * (np.pi / 180))
        weC = np.square(cHs[tempWave]) * cTp[tempWave]
        tempHsWaves = cHs[tempWave]
        wavePowerStormList.append(np.nansum(weC))
        longshorePowerStormList.append(np.nansum(lwpC))
        hsStormList.append(cHs[tempWave])
        hsMaxStormList.append(np.nanmax(cHs[tempWave]))
        peakTimeStormList.append(cTime[np.where(np.nanmax(tempHsWaves)==tempHsWaves)][0])
        tpStormList.append(cTp[tempWave])
        dmStormList.append(cDp[tempWave])
        timeStormList.append(cTime[tempWave])
        # duration = (int(t2.strftime('%s'))-int(t1.strftime('%s')))/60/60
        duration = ((t2 - dt.datetime(1970, 1, 1)).total_seconds() - (
                    t1 - dt.datetime(1970, 1, 1)).total_seconds()) / 3600
        durationStormList.append(duration)
        indStormList.append(indices)
        t1 = t1.replace(tzinfo=timezone.utc)
        t2 = t2.replace(tzinfo=timezone.utc)
        startTimeStormList.append(int(dt.datetime.timestamp(t1)))
        endTimeStormList.append(int(dt.datetime.timestamp(t2)))


################## GET STORM STATISTICS FROM WIS DATA ##################

stormLengthsWIS = np.asarray([len(tt) for tt in stormHsListWIS])
over12HourStormsWIS = np.where(stormLengthsWIS>=23)
hsStormListWIS = []
hsMaxStormListWIS = []
tpStormListWIS = []
dmStormListWIS = []
timeStormListWIS = []
hourStormListWIS = []
indStormListWIS = []
durationStormListWIS = []
wavePowerStormListWIS = []
longshorePowerStormListWIS = []
peakTimeStormListWIS = []
startTimeStormListWIS = []
endTimeStormListWIS = []
c = 0
for yy in range(len(over12HourStormsWIS[0])-1):

    index = over12HourStormsWIS[0][yy]
    i1 = stormHsListWIS[index][0]
    i2 = stormHsListWIS[index][-1]
    t1 = combinedTimeWIS[i1]
    t2 = combinedTimeWIS[i2]

    # should we check if we need to add storm waves onto this?
    nexti1 = combinedTimeWIS[stormHsListWIS[index+1][0]]
    # diff = (int(nexti1.strftime('%s'))-int(t2.strftime('%s')))/60/60#nexti1 - t2
    diff = ((nexti1-dt.datetime(1970,1,1)).total_seconds() - (t2-dt.datetime(1970,1,1)).total_seconds() )/3600

    if diff < 12:#timedelta(hours=12):
        print('Next storm is within 12 hours')
        t2 = combinedTimeWIS[stormHsListWIS[index+1][-1]]

    previousi2 = combinedTimeWIS[stormHsListWIS[index-1][-1]]
    # diff2 = (int(t1.strftime('%s'))-int(previousi2.strftime('%s')))/60/60#t1 - previousi2
    diff2 = ((t1-dt.datetime(1970,1,1)).total_seconds() - (previousi2-dt.datetime(1970,1,1)).total_seconds() )/3600

    if diff2 < 12:#timedelta(hours=12):
        print('Previous storm was within 12 hours')
        t1 = combinedTimeWIS[stormHsListWIS[index-1][0]]

    c = c + 1
    tempWave = np.where((combinedTimeWIS <= t2) & (combinedTimeWIS >= t1))
    newDiff = t2-t1
    if newDiff >= timedelta(hours=12):
        # print(newDiff)
        tempWave = np.where((combinedTimeWIS <= t2) & (combinedTimeWIS >= t1))
        indices = np.arange(i1,i2)

        lwpC = 1025 * np.square(combinedHsWIS[tempWave]) * combinedTpWIS[tempWave] * (9.81 / (64 * np.pi)) * np.cos(
            combinedDmWIS[tempWave] * (np.pi / 180)) * np.sin(combinedDmWIS[tempWave] * (np.pi / 180))
        weC = np.square(combinedHsWIS[tempWave]) * combinedTpWIS[tempWave]
        tempHsWaves = combinedHsWIS[tempWave]
        wavePowerStormListWIS.append(np.nansum(weC))
        longshorePowerStormListWIS.append(np.nansum(lwpC))
        hsStormListWIS.append(combinedHsWIS[tempWave])
        hsMaxStormListWIS.append(np.nanmax(combinedHsWIS[tempWave]))
        peakTimeStormListWIS.append(combinedTimeWIS[np.where(np.nanmax(tempHsWaves)==tempHsWaves)][0])
        tpStormListWIS.append(combinedTpWIS[tempWave])
        dmStormListWIS.append(combinedDmWIS[tempWave])
        timeStormListWIS.append(combinedTimeWIS[tempWave])
        # duration = (int(t2.strftime('%s'))-int(t1.strftime('%s')))/60/60
        duration = ((t2 - dt.datetime(1970, 1, 1)).total_seconds() - (
                t1 - dt.datetime(1970, 1, 1)).total_seconds()) / 3600
        durationStormListWIS.append(duration)
        indStormListWIS.append(indices)
        t1 = t1.replace(tzinfo=timezone.utc)
        t2 = t2.replace(tzinfo=timezone.utc)
        startTimeStormListWIS.append(int(dt.datetime.timestamp(t1)))
        endTimeStormListWIS.append(int(dt.datetime.timestamp(t2)))


################## MAKE STORMY_FULLSPAN ##################

stormy_fullspan = np.empty(shape=time_fullspan.shape)        # BINARY - stormy == 1, calm/non-stormy = nan
stormy_fullspan[:] = 0

## Remove storms outside of general time of interest
storm_start = np.array(startTimeStormList[:])
storm_end = np.array(endTimeStormList[:])
storm_startWIS = np.array(startTimeStormListWIS[:])
storm_endWIS = np.array(endTimeStormListWIS[:])
storm_start = storm_start[(storm_start >= time_fullspan[0]) & (storm_start < time_fullspan[-1])]
storm_end = storm_end[(storm_end > time_fullspan[0]) & (storm_end <= time_fullspan[-1])]
storm_startWIS = storm_startWIS[(storm_startWIS >= time_fullspan[0]) & (storm_startWIS < time_fullspan[-1])]
storm_endWIS = storm_endWIS[(storm_endWIS > time_fullspan[0]) & (storm_endWIS <= time_fullspan[-1])]

## Identify times when FRF or WIS data says stormy
for jj in np.arange(len(storm_start)):
    tt_during_storm = (time_fullspan >= storm_start[jj]) & (time_fullspan <= storm_end[jj])
    stormy_fullspan[tt_during_storm] = 1
for jj in np.arange(len(storm_startWIS)):
    tt_during_storm = (time_fullspan >= storm_startWIS[jj]) & (time_fullspan <= storm_endWIS[jj])
    stormy_fullspan[tt_during_storm] = 1

storm_timeend_all = []
storm_timestart_all = []
storm_iiend_all = []
storm_iistart_all = []
stormy_fullspan[stormy_fullspan == 0] = -1
iicross = np.where(stormy_fullspan[1:]*stormy_fullspan[0:-1] < 0)[0]
for jj in np.arange(iicross.size):
    if (stormy_fullspan[iicross[jj]] == -1) & (stormy_fullspan[iicross[jj]+1] == 1):
        storm_timestart_all = np.append(storm_timestart_all,time_fullspan[iicross[jj]])
        storm_iistart_all = np.append(storm_iistart_all,int(iicross[jj]+1))
    elif (stormy_fullspan[iicross[jj]] == 1) & (stormy_fullspan[iicross[jj]+1] == -1):
        storm_timeend_all = np.append(storm_timeend_all, time_fullspan[iicross[jj]])
        storm_iiend_all = np.append(storm_iiend_all,int(iicross[jj]+1))
    else:
        print('help')


################## SAVE DATA ##################

with open(picklefile_dir+'stormy_times_fullspan.pickle','wb') as file:
    pickle.dump([time_fullspan,stormy_fullspan,storm_timestart_all,storm_timeend_all],file)

