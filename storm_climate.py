import datetime as dt
import numpy as np
from netCDF4 import Dataset
import os
import more_itertools as mit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

local_base = '/volumes/macDrive/FRF_data/'

wave_base17 = 'waverider-17m/'
files17 = sorted((f for f in os.listdir(local_base+wave_base17) if not f.startswith(".")), key=str.lower) #os.listdir(local_base+wave_base17)
# files17.sort()
files_path17 = [os.path.join(os.path.abspath(local_base+wave_base17), x) for x in files17]

wave_base26 = 'waverider-26m/'
files26 = sorted((f for f in os.listdir(local_base+wave_base26) if not f.startswith(".")), key=str.lower)#os.listdir(local_base+wave_base26)
# files26.sort()
files_path26 = [os.path.join(os.path.abspath(local_base+wave_base26), x) for x in files26]

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

from dateutil.relativedelta import relativedelta
st = dt.datetime(2016, 1, 1)
end = dt.datetime(2024,10,1)
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
plt.plot(timeWave17,hs17,'.',label='17m')
plt.plot(np.asarray(timeWave26)[indices],hs26[indices],'.',label='26m')

combinedTime = np.hstack((np.asarray(timeWave17),np.asarray(timeWave26)[indices]))
combinedHs = np.hstack((hs17,hs26[indices]))
combinedTp = np.hstack((tp17,tp26[indices]))
combinedDp = np.hstack((dir17,dir26[indices]))

reindex = np.argsort(combinedTime)
cTime = combinedTime[reindex]
cHs = combinedHs[reindex]
cTp = combinedTp[reindex]
cDp = combinedDp[reindex]

cutOff = np.where(cTime < DT.datetime(2024,10,1))
cTime = cTime[cutOff]
cHs = cHs[cutOff]
cTp = cTp[cutOff]
cDp = cDp[cutOff]

waveNorm = cDp - 72
neg = np.where((waveNorm > 180))
waveNorm[neg[0]] = waveNorm[neg[0]]-360
offpos = np.where((waveNorm>90))
offneg = np.where((waveNorm<-90))
waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
waveNorm[offneg[0]] = waveNorm[offneg[0]]*0

# def wavetransform_point(H0, theta0, H1, theta1, T, h2, h1, g, breakcrit):

asdfg

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



stormLengths = np.asarray([len(tt) for tt in stormHsList])

over12HourStorms = np.where(stormLengths>=47)





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
c = 0
for yy in range(len(over12HourStorms[0])):

    index = over12HourStorms[0][yy]
    i1 = stormHsList[index][0]
    i2 = stormHsList[index][-1]
    t1 = cTime[i1]
    t2 = cTime[i2]

    # should we check if we need to add storm waves onto this?
    nexti1 = cTime[stormHsList[index+1][0]]
    diff = (int(nexti1.strftime('%s'))-int(t2.strftime('%s')))/60/60#nexti1 - t2
    if diff < 12:#timedelta(hours=12):
        print('Next storm is within 12 hours')
        t2 = cTime[stormHsList[index+1][-1]]

    previousi2 = cTime[stormHsList[index-1][-1]]
    diff2 = (int(t1.strftime('%s'))-int(previousi2.strftime('%s')))/60/60#t1 - previousi2
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
            waveNorm[tempWave] * (np.pi / 180)) * np.sin(waveNorm[tempWave] * (np.pi / 180))
        weC = np.square(cHs[tempWave]) * cTp[tempWave]

        wavePowerStormList.append(np.nansum(weC))
        longshorePowerStormList.append(np.nansum(lwpC))
        hsStormList.append(cHs[tempWave])
        hsMaxStormList.append(np.nanmax(cHs[tempWave]))
        tpStormList.append(cTp[tempWave])
        dmStormList.append(cDp[tempWave])
        timeStormList.append(cTime[tempWave])
        duration = (int(t2.strftime('%s'))-int(t1.strftime('%s')))/60/60
        durationStormList.append(duration)
        indStormList.append(indices)
        startTimeStormList.append(int(t1.strftime('%s')))
        endTimeStormList.append(int(t2.strftime('%s')))




#
#
# hsStormList = []
# hsMaxStormList = []
# tpStormList = []
# dmStormList = []
# timeStormList = []
# hourStormList = []
# indStormList = []
# durationStormList = []
# wavePowerStormList = []
# longshorePowerStormList = []
# startTimeStormList = []
# endTimeStormList = []
# c = 0
# while c < len(stormHsList)-1:
#
#     i1 = stormHsList[c][0]
#     i2 = stormHsList[c][-1]
#     t1 = cTime[i1]
#     t2 = cTime[i2]
#     nexti1 = cTime[stormHsList[c+1][0]]
#     diff = nexti1 - t2
#
#     # if diff < timedelta(hours=6):
#     #     i2 = stormHsList[c+1][-1]
#     #     t2 = cTime[i2]
#     #     nexti1 = cTime[stormHsList[c+2][0]]
#     #     diff2 = nexti1-t2
#     #     c = c + 1
#     #     print('we combined a storm')
#     #     if diff2 < timedelta(hours=6):
#     #         i2 = stormHsList[c + 2][-1]
#     #         t2 = cTime[i2]
#     #         nexti1 = cTime[stormHsList[c + 3][0]]
#     #         diff3 = nexti1 - t2
#     #         c = c + 1
#     #         print('we stacked 3 of them together')
#     #
#     #         if diff3 < timedelta(hours=6):
#     #             i2 = stormHsList[c + 3][-1]
#     #             t2 = cTime[i2]
#     #             nexti1 = cTime[stormHsList[c + 4][0]]
#     #             diff4 = nexti1 - t2
#     #             c = c + 1
#     #             if diff4 < timedelta(hours=6):
#     #                 print('bummer, need another loop')
#     c = c + 1
#     tempWave = np.where((cTime <= t2) & (cTime >= t1))
#     newDiff = t2-t1
#     if newDiff >= timedelta(hours=12):
#         # print(newDiff)
#         tempWave = np.where((cTime <= t2) & (cTime >= t1))
#         indices = np.arange(i1,i2)
#
#         lwpC = 1025 * np.square(cHs[tempWave]) * cTp[tempWave] * (9.81 / (64 * np.pi)) * np.cos(
#             waveNorm[tempWave] * (np.pi / 180)) * np.sin(waveNorm[tempWave] * (np.pi / 180))
#         weC = np.square(cHs[tempWave]) * cTp[tempWave]
#
#         wavePowerStormList.append(np.nansum(weC))
#         longshorePowerStormList.append(np.nansum(lwpC))
#         hsStormList.append(cHs[tempWave])
#         hsMaxStormList.append(np.nanmax(cHs[tempWave]))
#         tpStormList.append(cTp[tempWave])
#         dmStormList.append(cDp[tempWave])
#         timeStormList.append(cTime[tempWave])
#         duration = (int(t2.strftime('%s'))-int(t1.strftime('%s')))/60/60
#         durationStormList.append(duration)
#         indStormList.append(indices)
#         startTimeStormList.append(int(t1.strftime('%s')))
#         endTimeStormList.append(int(t2.strftime('%s')))



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



clusterPickle = 'stormHs90Over24Hours.pickle'
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
output['cDp'] = cDp
output['waveNorm'] = waveNorm
output['cTime'] = cTime
import pickle
with open(clusterPickle,'wb') as f:
    pickle.dump(output, f)






#
# from run_makeplots import *
# from funcs.create_contours import *
# from run_lidarcollect import *
# # DEFINE SUBDIR WITH LIDAR FILES
# lidarfloc = local_base + 'dune_lidar/lidar_transect/'
# lidarext = 'nc'  # << change not recommended; defines file type to look for
# # DEFINE CONTOUR ELEVATIONS OF INTEREST
# cont_elev = np.arange(-0.25,4.25,0.5)    # <<< MUST BE POSITIVELY INCREASING
#
# preProfile = []
# postProfile = []
# allStormProfiles = []
# allStormProfileTimes = []
# for hhh in range(len(durationStormList)):
#     print('working on storm {} of {}: {}'.format(hhh,len(durationStormList), dt.datetime.fromtimestamp(startTimeStormList[hhh])))
#     # convert period of interest to datenum
#     time_format = '%Y-%m-%dT%H:%M:%S'
#     epoch_beg = startTimeStormList[hhh]#dt.datetime.strptime(time_beg,time_format).timestamp()
#     epoch_end = endTimeStormList[hhh]#dt.datetime.strptime(time_end,time_format).timestamp()
#     TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)
#     tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4
#
#     # run file run_lidarcollect.py
#     lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext, epoch_end, epoch_beg, tzinfo)
#
#     # Remove weird data (first order filtering)
#     stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
#     pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
#     if np.size(lidarelev) == 0:
#         print('skipping lidar data check')
#     else:
#         tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
#         lidarelev[tmpii] = np.nan
#     # # run file create_contours.py
#     # elev_input = lidarelev
#     # cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)
#     #
#     # plot_ProfilesTimestack(elev_input,lidartime,lidar_xFRF)
#     # plot_ProfilesSubset(elev_input,lidartime,lidar_xFRF,len(lidartime),dt.datetime.fromtimestamp(epoch_beg).strftime("%Y-%m-%d %I:%M:%S"),dt.datetime.fromtimestamp(epoch_end).strftime("%Y-%m-%d %I:%M:%S"),tzinfo,TOI_duration)
#     if np.size(lidarelev) == 0:
#         print('no lidar data for this storm?')
#     else:
#         preProfile.append(lidarelev[0,:])
#         postProfile.append(lidarelev[-1,:])
#         allStormProfiles.append(lidarelev)
#         allStormProfileTimes.append(lidartime)
#
#
#
# allRecoveryProfiles = []
# allRecoveryProfileTimes = []
# for hhh in range(len(durationStormList)-1):
#     print('working on recovery {} of {}: {}'.format(hhh,len(durationStormList), dt.datetime.fromtimestamp(endTimeStormList[hhh])))
#     # convert period of interest to datenum
#     time_format = '%Y-%m-%dT%H:%M:%S'
#     epoch_beg = endTimeStormList[hhh]#dt.datetime.strptime(time_beg,time_format).timestamp()
#     epoch_end = startTimeStormList[hhh+1]#dt.datetime.strptime(time_end,time_format).timestamp()
#     TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)
#     tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4
#
#     # run file run_lidarcollect.py
#     lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext, epoch_end, epoch_beg, tzinfo)
#
#     # Remove weird data (first order filtering)
#     stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
#     pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
#     if np.size(lidarelev) == 0:
#         print('skipping lidar data check')
#     else:
#         tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
#         lidarelev[tmpii] = np.nan
#     # # run file create_contours.py
#     # elev_input = lidarelev
#     # cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)
#     #
#     # plot_ProfilesTimestack(elev_input,lidartime,lidar_xFRF)
#     # plot_ProfilesSubset(elev_input,lidartime,lidar_xFRF,len(lidartime),dt.datetime.fromtimestamp(epoch_beg).strftime("%Y-%m-%d %I:%M:%S"),dt.datetime.fromtimestamp(epoch_end).strftime("%Y-%m-%d %I:%M:%S"),tzinfo,TOI_duration)
#     if np.size(lidarelev) == 0:
#         print('no lidar data for this storm?')
#     else:
#         allRecoveryProfiles.append(lidarelev)
#         allRecoveryProfileTimes.append(lidartime)
#
#
#
# plt.figure()
# subp1 = plt.subplot2grid((1,2),(0,0))
# for qqq in range(len(preProfile)):
#     subp1.plot(lidar_xFRF,preProfile[qqq])
# subp1.set_title('All Pre-Storm Profiles')
# subp1.set_xlabel('cross-shore (m)')
# subp1.set_ylabel('NAVD88 (m)')
# subp1.set_xlim([40,140])
# subp2 = plt.subplot2grid((1,2),(0,1))
# for qqq in range(len(postProfile)):
#     subp2.plot(lidar_xFRF,postProfile[qqq])
# subp2.set_title('All Post-Storm Profiles')
# subp2.set_xlabel('cross-shore (m)')
# subp2.set_ylabel('NAVD88 (m)')
# subp2.set_xlim([40,140])
#
#
#
#
#
# from funcs.lidar_fillgaps import *
# allRecoveryContours = []
# for hhh in range(len(allRecoveryProfiles)):
#     print('working on recovery {} of {}: {}'.format(hhh,len(allRecoveryProfiles), dt.datetime.fromtimestamp(endTimeStormList[hhh])))
#
#     tempLidarElev = allRecoveryProfiles[hhh]
#     tempLidarTime = allRecoveryProfileTimes[hhh]
#     # Try filling gaps??
#     halfspan_time = 4
#     halfspan_x = 5
#     if len(tempLidarTime)>10:
#         lidar_filled = lidar_fillgaps(tempLidarElev,tempLidarTime,lidar_xFRF,halfspan_time,halfspan_x)
#
#     # # Plot the new lidar, filled in gaps
#     # plot_PrefillPostfillTimestack(tempLidarElev,lidar_filled,tempLidarTime,lidar_xFRF)
#     #
#     # Re-run file create_contours.py
#         elev_input = lidar_filled
#     else:
#         elev_input = tempLidarElev
#     cont_elev = np.arange(0,2.25,0.25)    # <<< MUST BE POSITIVELY INCREASING
#     cont_ts, cmean, cstd = create_contours(elev_input,tempLidarTime,lidar_xFRF,cont_elev)
#     allRecoveryContours.append(cont_ts)
#
#
# from funcs.lidar_fillgaps import *
# allStormContours = []
# for hhh in range(len(allStormProfiles)):
#     print('working on recovery {} of {}: {}'.format(hhh,len(allStormProfiles), dt.datetime.fromtimestamp(startTimeStormList[hhh])))
#
#     tempLidarElev = allStormProfiles[hhh]
#     tempLidarTime = allStormProfileTimes[hhh]
#     # Try filling gaps??
#     halfspan_time = 4
#     halfspan_x = 5
#     if len(tempLidarTime)>10:
#         lidar_filled = lidar_fillgaps(tempLidarElev,tempLidarTime,lidar_xFRF,halfspan_time,halfspan_x)
#
#     # # Plot the new lidar, filled in gaps
#     # plot_PrefillPostfillTimestack(tempLidarElev,lidar_filled,tempLidarTime,lidar_xFRF)
#     #
#     # Re-run file create_contours.py
#         elev_input = lidar_filled
#     else:
#         elev_input = tempLidarElev
#     cont_elev = np.arange(0,2.25,0.25)    # <<< MUST BE POSITIVELY INCREASING
#     cont_ts, cmean, cstd = create_contours(elev_input,tempLidarTime,lidar_xFRF,cont_elev)
#     allStormContours.append(cont_ts)
#
# plt.figure()
# subplot1 = plt.subplot2grid((1,1),(0,0))
# for ppp in range(len(allStormContours)):
#     tempTime = [dt.datetime.fromtimestamp(pp) for pp in allStormProfileTimes[ppp]]
#     subplot1.plot(tempTime,allStormContours[ppp][0,:],'.',color='red')
#     subplot1.plot(tempTime,allStormContours[ppp][2,:],'.',color='orange')
#     subplot1.plot(tempTime,allStormContours[ppp][4,:],'.',color='purple')
#     subplot1.plot(tempTime,allStormContours[ppp][6,:],'.',color='green')
#     subplot1.plot(tempTime,allStormContours[ppp][8,:],'.',color='blue')
#
# for ppp in range(len(allRecoveryContours)):
#     tempTime = [dt.datetime.fromtimestamp(pp) for pp in allRecoveryProfileTimes[ppp]]
#     subplot1.plot(tempTime,allRecoveryContours[ppp][0,:],'.',color='red')
#     subplot1.plot(tempTime,allRecoveryContours[ppp][2,:],'.',color='orange')
#     subplot1.plot(tempTime,allRecoveryContours[ppp][4,:],'.',color='purple')
#     subplot1.plot(tempTime,allRecoveryContours[ppp][6,:],'.',color='green')
#     subplot1.plot(tempTime,allRecoveryContours[ppp][8,:],'.',color='blue')
#
#
# preStormTime = [allStormProfileTimes[pp][0] for pp in range(len(allStormContours))]
# # preStormTime = [dt.datetime.fromtimestamp(allStormProfileTimes[pp][0]) for pp in range(len(allStormContours))]
# preStorm1m = [np.nanmean(allStormContours[pp][4,0:4]) for pp in range(len(allStormContours))]
# postStorm1m = [np.nanmean(allStormContours[pp][4,-4:]) for pp in range(len(allStormContours))]
# diffStorm = np.asarray(preStorm1m)-np.asarray(postStorm1m)
# diffRecover = np.asarray(preStorm1m)[1:]-np.asarray(postStorm1m)[0:-1]
#
#
# stormIndices = []
# for pp in range(len(preStormTime)):
#     timeDiff = abs(np.asarray(startTimeStormList)-preStormTime[pp])
#     stormIndices.append(np.argmin(timeDiff))
# stormIndices2 = np.asarray(stormIndices)[0:-1]
# plt.figure()
# plot1 = plt.subplot2grid((2,1),(0,0))
# plot1.plot(cTime,cHs)
# for qq in range(len(hsStormList)):
#     plot1.plot(timeStormList[qq],hsStormList[qq],'.',color='orange')
# plot1.set_ylabel('Hs (m)')
# plot2 = plt.subplot2grid((2,1),(1,0))
# # plot2.scatter(np.asarray(startTimeStormList)[0:-1],np.asarray(hsMaxStormList)[0:-1],s=np.asarray(wavePowerStormList)[0:-1]/100,c=np.asarray(afterStorm))
# # plot2.scatter(np.asarray(startTimeStormList)[0:-1],np.asarray(hsMaxStormList)[0:-1],c=np.asarray(wavePowerStormList)[0:-1]/100,s=np.asarray(afterStorm)/0.75)
# # plot2.scatter(np.asarray(startTimeStormList)[stormIndices2],np.asarray(preStorm1m)[0:-1],c=np.asarray(wavePowerStormList)[stormIndices2]/100,s=np.asarray(afterStorm)[stormIndices2]/0.75)
# # sc = plot2.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList])[stormIndices2],np.asarray(preStorm1m)[0:-1],s=np.asarray(wavePowerStormList)[stormIndices2]/100,c=diffStorm[0:-1]*10)#,vmin=3,vmax=35)
# sc = plot2.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList])[stormIndices2],np.asarray(preStorm1m)[0:-1],s=np.asarray(wavePowerStormList)[stormIndices2]/100,c=diffStorm[0:-1]*10)#,vmin=3,vmax=35)
#
# plot2.set_ylabel('Pre-storm 1.5 m contour (xFRF - m)')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Wave Power')
# cbaxes = inset_axes(plot2, width="30%", height="3%", loc=1)
# cb = plt.colorbar(sc,cax=cbaxes,orientation='horizontal')
# cb.set_ticks([0,100])
# cb.set_ticklabels([0,10])
# cb.set_label('Cross-shore Change During Storm (m)')
#
#
#
#
#
#
#
# plt.figure()
# plot1 = plt.subplot2grid((2,1),(0,0))
# plot1.plot(cTime,cHs)
# for qq in range(len(hsStormList)):
#     plot1.plot(timeStormList[qq],hsStormList[qq],'.',color='orange')
# plot1.set_ylabel('Hs (m)')
# plot2 = plt.subplot2grid((2,1),(1,0))
# # plot2.scatter(np.asarray(startTimeStormList)[0:-1],np.asarray(hsMaxStormList)[0:-1],s=np.asarray(wavePowerStormList)[0:-1]/100,c=np.asarray(afterStorm))
# # plot2.scatter(np.asarray(startTimeStormList)[0:-1],np.asarray(hsMaxStormList)[0:-1],c=np.asarray(wavePowerStormList)[0:-1]/100,s=np.asarray(afterStorm)/0.75)
# # plot2.scatter(np.asarray(startTimeStormList)[stormIndices2],np.asarray(preStorm1m)[0:-1],c=np.asarray(wavePowerStormList)[stormIndices2]/100,s=np.asarray(afterStorm)[stormIndices2]/0.75)
# # sc = plot2.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList])[stormIndices2],np.asarray(preStorm1m)[0:-1],s=np.asarray(wavePowerStormList)[stormIndices2]/100,c=diffStorm[0:-1]*10)#,vmin=3,vmax=35)
# sc = plot2.scatter(np.asarray([dt.datetime.fromtimestamp(pp) for pp in startTimeStormList])[stormIndices2],np.asarray(postStorm1m)[0:-1],s=np.asarray(afterStorm)[stormIndices2],c=diffRecover*10)#,vmin=3,vmax=35)
#
# plot2.set_ylabel('Post-Storm 1.0 m contour (xFRF - m)')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.legend(*sc.legend_elements("sizes", num=6),title='Days of Recovery')
# cbaxes = inset_axes(plot2, width="30%", height="3%", loc=1)
# cb = plt.colorbar(sc,cax=cbaxes,orientation='horizontal')
# cb.set_ticks([0,100])
# cb.set_ticklabels([0,10])
# cb.set_label('Cross-shore Change During Recover (m)')
#
#
#
#
#
#
#
# cmap = plt.cm.rainbow(np.linspace(0, 1, len(allStormContours) + 1))
# plt.figure()
# subplot1 = plt.subplot2grid((1,1),(0,0))
# for ppp in range(len(allStormContours)):
#     # tempTime = [dt.datetime.fromtimestamp(pp) for pp in allStormProfileTimes[ppp]]
#     intTime = allStormProfileTimes[ppp]-allStormProfileTimes[ppp][0]
#     tempTime = [dt.datetime.fromtimestamp(pp) for pp in intTime]
#     subplot1.plot(np.asarray(tempTime),allStormContours[ppp][3,:],'.',color=cmap[ppp,:])#,color='red')
#
# #
# #
# # clusterPickle = 'stormRecoveryPeriodsWaveHeightHs90with24hours.pickle'
# # output = {}
# # output['allRecoveryProfileTimes'] = allRecoveryProfileTimes
# # output['allRecoveryContours'] = allRecoveryContours
# # output['allRecoveryProfiles'] = allRecoveryProfiles
# # output['allStormContours'] = allStormContours
# # output['allStormProfileTimes'] = allStormProfileTimes
# # output['allStormProfiles'] = allStormProfiles
# # output['timeStormList'] = timeStormList
# # output['preProfile'] = preProfile
# # output['postProfile'] = postProfile
# # output['hsStormList'] = hsStormList
# # output['hsMaxStormList'] = hsMaxStormList
# # output['tpStormList'] = tpStormList
# # output['dmStormList'] = dmStormList
# # output['hourStormList'] = hourStormList
# # output['indStormList'] = indStormList
# # output['durationStormList'] = durationStormList
# # output['wavePowerStormList'] = wavePowerStormList
# # output['longshorePowerStormList'] = longshorePowerStormList
# # output['startTimeStormList'] = startTimeStormList
# # output['endTimeStormList'] = endTimeStormList
# # output['lidar_xFRF'] = lidar_xFRF
# # output['cont_elev'] = cont_elev
# # output['cHs'] = cHs
# # output['cTp'] = cTp
# # output['cDp'] = cDp
# # output['waveNorm'] = waveNorm
# # import pickle
# # with open(clusterPickle,'wb') as f:
# #     pickle.dump(output, f)

