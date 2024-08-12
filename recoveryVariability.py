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


dailyAverage = np.nan * np.ones((len(dayTime),len(lidar_xFRF)))
dailyStd = np.nan * np.ones((len(dayTime),len(lidar_xFRF)))

for qq in range(len(dayTime)-1):
    inder = np.where((np.asarray(dts)>=np.asarray(dayTime)[qq]) & (np.asarray(dts) <=np.asarray(dayTime)[qq+1]))
    if len(inder[0])>0:
        dailyAverage[qq,:] = np.nanmean(lidarProfiles[inder[0],:],axis=0)
        dailyStd[qq,:] = np.nanstd(lidarProfiles[inder[0],:],axis=0)

#
# plt.figure()
# plt.pcolor(dayTime,lidar_xFRF,dailyAverage.T)





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

# local_base = '/volumes/macDrive/FRF_data/'
local_base = 'D:/FRF_data/'

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





plt.figure()
plt.plot(timeWave17,hs17,'.')
plt.plot(np.asarray(timeWave26)[indices],hs26[indices],'.')

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
hsSmooth = moving_average(cHs,3)#np.asarray([avgHs,moving_average(hs,3),avgHs])
# stormHsInd = np.where((hsSmooth > 1.5))
stormHsInd = np.where((hsSmooth > hs90))
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
gapBetweenStorms = 4
c = 0
while c < len(stormHsList)-1:

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
        indices = np.arange(i1,i2)

        lwpC = 1025 * np.square(cHs[tempWave]) * cTp[tempWave] * (9.81 / (64 * np.pi)) * np.cos(
            waveNorm[tempWave] * (np.pi / 180)) * np.sin(waveNorm[tempWave] * (np.pi / 180))
        weC = np.square(cHs[tempWave]) * cTp[tempWave]

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


