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


# Load processed deep-water waves from get_wavesForStormClimate
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'stormWaves_WISandFRF.pickle', 'rb') as file:
    combinedHsWIS,combinedTpWIS,combinedDmWIS,combinedTimeWIS = pickle.load(file)
with open(picklefile_dir+'stormWaves_FRF.pickle','rb') as file:
    cHs,cTp,cDp,cTime = pickle.load(file)

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
# stormHsInd = np.where((hsSmooth > 1.5))
stormHsIndWIS = np.where((hsSmoothWIS > hs94))
stormHsListWIS = [list(group) for group in mit.consecutive_groups(stormHsIndWIS[0])]
stormHsInd = np.where((hsSmooth > hs94))
stormHsList = [list(group) for group in mit.consecutive_groups(stormHsInd[0])]


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


plt.figure()
plot1 = plt.subplot2grid((2,1),(0,0))
plot1.plot(combinedTimeWIS,combinedHsWIS)
for qq in range(len(hsStormListWIS)):
    plot1.plot(timeStormListWIS[qq],hsStormListWIS[qq],'.',color='orange')
plot1.set_ylabel('Hs (m)')
plot2 = plt.subplot2grid((2,1),(1,0))
# plot2.plot(cTime,cHs)
# for qq in range(len(hsStormList)):
plot2.scatter(np.asarray(peakTimeStormListWIS),np.asarray(wavePowerStormListWIS))
# plot2.ylabel('Hs (m)')

plt.figure()
cp = plt.scatter(np.asarray(durationStormListWIS),np.asarray(hsMaxStormListWIS),c=np.asarray(wavePowerStormListWIS),vmin=0,vmax=30000)
cb = plt.colorbar(cp)
cb.set_label('Cumulative Wave Power')
plt.ylabel('Max Hs (m)')
plt.xlabel('Duration (hrs)')

afterStormWIS = (np.asarray(endTimeStormListWIS)[1:]-np.asarray(startTimeStormListWIS)[0:-1])/60/60/24
plt.figure()
cp = plt.scatter(np.asarray(durationStormListWIS)[0:-1],np.asarray(hsMaxStormListWIS)[0:-1],c=np.asarray(afterStormWIS),vmin=0,vmax=60)
cb = plt.colorbar(cp)
cb.set_label('Recovery Time After (days)')
plt.ylabel('Max Hs (m)')
plt.xlabel('Duration (hrs)')



plt.figure()
plot1 = plt.subplot2grid((2,1),(0,0))
plot1.plot(cTime,cHs)
for qq in range(len(hsStormList)):
    plot1.plot(timeStormList[qq],hsStormList[qq],'.',color='orange')
plot1.set_ylabel('Hs (m)')
plot2 = plt.subplot2grid((2,1),(1,0))
# plot2.plot(cTime,cHs)
# for qq in range(len(hsStormList)):
plot2.scatter(startTimeStormList,wavePowerStormList)
# plot2.ylabel('Hs (m)')




plt.figure()
cp = plt.scatter(np.asarray(durationStormList),np.asarray(hsMaxStormList),c=np.asarray(wavePowerStormList),vmin=0,vmax=30000)
cb = plt.colorbar(cp)
cb.set_label('Cumulative Wave Power')
plt.ylabel('Max Hs (m)')
plt.xlabel('Duration (hrs)')

afterStorm = (np.asarray(endTimeStormList)[1:]-np.asarray(startTimeStormList)[0:-1])/60/60/24
plt.figure()
cp = plt.scatter(np.asarray(durationStormList)[0:-1],np.asarray(hsMaxStormList)[0:-1],c=np.asarray(afterStorm),vmin=0,vmax=60)
cb = plt.colorbar(cp)
cb.set_label('Recovery Time After (days)')
plt.ylabel('Max Hs (m)')
plt.xlabel('Duration (hrs)')



clusterPickle = 'stormHs95_Over12Hours.pickle'
output = {}
output['timeStormList'] = timeStormList
output['hsStormList'] = hsStormList
output['hsMaxStormList'] = hsMaxStormList
output['tpStormList'] = tpStormList
output['dmStormList'] = dmStormList
output['hourStormList'] = hourStormList
output['indStormList'] = indStormList
output['durationStormList'] = durationStormList
output['wavePowerStormList'] = wavePowerStormList
output['longshorePowerStormList'] = longshorePowerStormList
output['startTimeStormList'] = startTimeStormList
output['endTimeStormList'] = endTimeStormList
output['cHs'] = cHs
output['cTp'] = cTp
output['cDp'] = cDp     # waves are normalized to FRF shoreline
output['cTime'] = cTime
output['peakTimeStormList'] = peakTimeStormList
output['timeStormListWIS'] = timeStormListWIS
output['hsStormListWIS'] = hsStormListWIS
output['hsMaxStormListWIS'] = hsMaxStormListWIS
output['tpStormListWIS'] = tpStormListWIS
output['dmStormListWIS'] = dmStormListWIS
output['hourStormListWIS'] = hourStormListWIS
output['indStormListWIS'] = indStormListWIS
output['durationStormListWIS'] = durationStormListWIS
output['wavePowerStormListWIS'] = wavePowerStormListWIS
output['longshorePowerStormListWIS'] = longshorePowerStormListWIS
output['startTimeStormListWIS'] = startTimeStormListWIS
output['endTimeStormListWIS'] = endTimeStormListWIS
output['combinedHsWIS'] = combinedHsWIS
output['combinedTpWIS'] = combinedTpWIS
output['combinedDmWIS'] = combinedDmWIS # waves are normalized to FRF shoreline
output['combinedTimeWIS'] = combinedTimeWIS
output['peakTimeStormListWIS'] = peakTimeStormListWIS


picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
# with open(picklefile_dir+clusterPickle,'wb') as f:
#     pickle.dump(output, f)





