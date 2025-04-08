import datetime as dt
import numpy as np
from netCDF4 import Dataset
import os
import more_itertools as mit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

stormPickle = 'stormHs95Over12Hours.pickle'
with open(stormPickle, "rb") as input_file:
    inputStorms = pickle.load(input_file)
# timeStormList = inputStorms['timeStormList']
# hsStormList = inputStorms['hsStormList']
# hsMaxStormList = inputStorms['hsMaxStormList']
# tpStormList = inputStorms['tpStormList']
# dmStormList = inputStorms['dmStormList']
# hourStormList = inputStorms['hourStormList']
# indStormList = inputStorms['indStormList']
# durationStormList = inputStorms['durationStormList']
# wavePowerStormList = inputStorms['wavePowerStormList']
# longshorePowerStormList = inputStorms['longshorePowerStormList']
# startTimeStormList = inputStorms['startTimeStormList']
# endTimeStormList = inputStorms['endTimeStormList']
# cHs = inputStorms['cHs']
# cTp = inputStorms['cTp']
# cDp = inputStorms['cDp']
# waveNorm = inputStorms['waveNorm']
# cTime = inputStorms['cTime']

timeStormList = inputStorms['timeStormListWIS']
hsStormList = inputStorms['hsStormListWIS']
hsMaxStormList = inputStorms['hsMaxStormListWIS']
tpStormList = inputStorms['tpStormListWIS']
dmStormList = inputStorms['dmStormListWIS']
hourStormList = inputStorms['hourStormListWIS']
indStormList = inputStorms['indStormListWIS']
durationStormList = inputStorms['durationStormListWIS']
wavePowerStormList = inputStorms['wavePowerStormListWIS']
longshorePowerStormList = inputStorms['longshorePowerStormListWIS']
startTimeStormList = inputStorms['startTimeStormListWIS']
endTimeStormList = inputStorms['endTimeStormListWIS']
cHsWIS = inputStorms['combinedHsWIS']
cTpWIS = inputStorms['combinedTpWIS']
cDpWIS = inputStorms['combinedDmWIS']
waveNorm = inputStorms['combinedDmWIS']
cTimeWIS = inputStorms['combinedTimeWIS']

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


import sys
sys.path.append('/users/dylananderson/Documents/projects/gencadeClimate/')

savePath = '/volumes/macDrive/va/'
import os
import pickle
with open(os.path.join(savePath,'dwts12hr1degRes64withRG02.pickle'), "rb") as input_file:
   dwts = pickle.load(input_file)
bmus = dwts['bmus']
bmus_corrected = dwts['bmus_corrected']
windowHs = dwts['windowHs']
windowTp = dwts['windowTp']

with open(os.path.join(savePath,'slps12hr1degRes.pickle'), "rb") as input_file:
   slps = pickle.load(input_file)
DATES = slps['DATES']

with open(os.path.join(savePath,'pcas12hr1degRes.pickle'), "rb") as input_file:
   pcas = pickle.load(input_file)
SlpGrdMean = pcas['SlpGrdMean']

isOnLandFlat = slps['isOnLandFlat']
xFlat = slps['xFlat']
yFlat = slps['yFlat']
xGrid = slps['x2']
yGrid = slps['y2']
numClusters = dwts['numClustersETC']
# num_clustersTC = dwts['numClustersTC']
kmaOrderETC = dwts['kmaOrderETC']
Km_ETC = dwts['Km_ETC']
# groupSizeTC = dwts['groupSizeTC']
# Km_TC = dwts['Km_TC']
groupSizeETC = dwts['groupSizeETC']
# duckMET = priorComputations['duckMet']
# climate = priorComputations['climate']
basin = 'atlantic'
latBot = -5
latTop = 65
lonRight = 0
lonLeft = 270

hs95 = np.nanpercentile(cHsWIS,95)
hs975 = np.nanpercentile(cHsWIS,97.5)


windowHs = []
windowTp = []
windowPercentAbove95 = []
windowPercentAbove975 = []

for qq in range(len(DATES) - 1):
    if np.remainder(qq, 5000) == 0:
        print('done up to {}'.format(DATES[qq]))
    windowIndex = np.where((cTimeWIS > DATES[qq]) & (cTimeWIS < DATES[qq + 1]))
    if len(windowIndex[0]) > 0:
        tempHs = cHsWIS[windowIndex]
        windowHs.append(np.max(cHsWIS[windowIndex]))
        windowTp.append(np.mean(cTpWIS[windowIndex]))
        aboveInd = np.where(tempHs>hs95)
        if len(aboveInd[0]) > 0:
            windowPercentAbove95.append(len(aboveInd[0]))
        else:
            windowPercentAbove95.append(np.nan)
        aboveInd975 = np.where(tempHs>hs975)
        if len(aboveInd975[0]) > 0:
            windowPercentAbove975.append(len(aboveInd975[0]))
        else:
            windowPercentAbove975.append(np.nan)
        del aboveInd
        del aboveInd975
        del windowIndex
    else:
        windowHs.append(np.nan)
        windowTp.append(np.nan)
        windowPercentAbove95.append(np.nan)
        windowPercentAbove975.append(np.nan)

windowHs = np.asarray(windowHs)
windowTp = np.asarray(windowTp)
windowPercentAbove95 = np.asarray((windowPercentAbove95))
windowPercentAbove975 = np.asarray((windowPercentAbove975))

bigIndex = np.where(windowHs>hs95)
bigBmus = bmus_corrected[bigIndex[0]]
biggerIndex = np.where(windowPercentAbove975>8)
biggerBmus = bmus_corrected[biggerIndex[0]]
numWindows = []
percentWindows = []
for hh in range(64):
    inderTemp = np.where(biggerBmus == hh)
    allTemp = np.where(bmus_corrected[600:] == hh)
    if len(inderTemp) > 0:
        numWindows.append(len(inderTemp[0]))
        percentWindows.append(len(inderTemp[0])/len(allTemp[0]))

plt.figure()
plt.pcolor(np.flipud(np.asarray(percentWindows).reshape(8,8)),vmin=0,vmax=.20,cmap='Reds')
plt.colorbar()

mons = np.asarray([dd.month for dd in DATES])
marchWindows = []
septWindows = []
for hh in range(64):
    allTemp = np.where(bmus_corrected == hh)
    subsetMonths = mons[allTemp]
    marchTemp = np.where((subsetMonths == 1) | (subsetMonths == 2) | (subsetMonths == 3))
    septTemp = np.where((subsetMonths == 8) | (subsetMonths == 9) | (subsetMonths == 7))

    if len(marchTemp) > 0:
        marchWindows.append(len(marchTemp[0])/len(allTemp[0]))
    else:
        marchWindows.append(0)
    if len(septTemp) > 0:
        septWindows.append(len(septTemp[0])/len(allTemp[0]))
    else:
        septWindows.append(0)

plt.figure()
p1 = plt.subplot2grid((2,1),(0,0))
p1.pcolor(np.flipud(np.asarray(marchWindows).reshape(8,8)),vmin=0,vmax=1,cmap='Reds')
# plt.colorbar()
p2 = plt.subplot2grid((2,1),(1,0))
pc2 = p2.pcolor(np.flipud(np.asarray(septWindows).reshape(8,8)),vmin=0,vmax=1,cmap='Reds')
# sc = plt.colorbar(pc2,ax=p2)

import more_itertools as mit
stormConsecutiveList = [list(group) for group in mit.consecutive_groups(biggerIndex[0])]

bmuList = []
for qq in range(len(stormConsecutiveList)):
    bmuList.append(bmus_corrected[stormConsecutiveList[qq]])
recoveryWindow = []
for qq in range(len(stormConsecutiveList)-1):
    recoveryWindow.append((stormConsecutiveList[qq+1][0]-stormConsecutiveList[qq][-1])*12)

recoveryWindow = np.asarray(recoveryWindow)

plt.figure()
plt.hist(recoveryWindow/24,25)
# stormBMUSstart = []
# stormBMUSend = []
# stormBMUSmid = []
# stormBMUSstart_corrected = []
# stormBMUSend_corrected = []
# stormBMUSmid_corrected = []
# stormBMUSall = np.nan
# stormBMUSall_corrected = np.nan
#
# for qq in range(len(timeStormList)):
#     tempTimeS = timeStormList[qq][0]
#     tempTimeE = timeStormList[qq][-1]
#     tempLength = np.round(len(timeStormList[qq])/2)
#     tempTimeM = timeStormList[qq][int(tempLength)]
#     inderS = np.where(DATES<tempTimeS)
#     inderM = np.where(DATES<tempTimeM)
#     inderE = np.where(DATES<tempTimeE)
#     inderAll = np.where((DATES>tempTimeS) & (DATES<tempTimeE))
#     stormBMUSstart.append(bmus[inderS[0][-1]])
#     stormBMUSmid.append(bmus[inderM[0][-1]])
#     stormBMUSend.append(bmus[inderE[0][-1]])
#     stormBMUSall = np.hstack((stormBMUSall,bmus[inderAll[0]]))
#     stormBMUSstart_corrected.append(bmus_corrected[inderS[0][-1]])
#     stormBMUSmid_corrected.append(bmus_corrected[inderM[0][-1]])
#     stormBMUSend_corrected.append(bmus_corrected[inderE[0][-1]])
#     stormBMUSall_corrected = np.hstack((stormBMUSall_corrected,bmus[inderAll[0]]))
#
# stormBMUSall = stormBMUSall[1:]
# stormBMUSall_corrected = stormBMUSall_corrected[1:]








import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import numpy as np

dwtcolors = cm.rainbow(np.linspace(0, 1, numClusters))
# plotting the EOF patterns
fig2 = plt.figure(figsize=(10, 10))
# gs1 = gridspec.GridSpec(int(np.ceil(np.sqrt(struct.numClustersETC))), int(np.ceil(np.sqrt(struct.numClustersETC))))
gs1 = gridspec.GridSpec(int(8), int(8))

gs1.update(wspace=0.00, hspace=0.00)  # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(numClusters):
    # p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh])
    num = kmaOrderETC[hh]

    spatialField = Km_ETC[(hh), 0:len(xFlat[~isOnLandFlat])] / 100 - SlpGrdMean[0:len(
        xFlat[~isOnLandFlat])] / 100
    # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))

    if basin == 'atlantic':
        X_in = xFlat[~isOnLandFlat]
        X_in_Checker = np.where(X_in < 180)
        X_in[X_in_Checker] = X_in[X_in_Checker] + 360

    else:
        X_in = xFlat[~isOnLandFlat]

    Y_in = yFlat[~isOnLandFlat]
    sea_nodes = []
    for qq in range(len(X_in)):
        sea_nodes.append(np.where((xGrid == X_in[qq]) & (yGrid == Y_in[qq])))

    rectField = np.ones((np.shape(xGrid))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    clevels = np.arange(-32, 32, 1)
    # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
    #             resolution='c')
    if basin == 'atlantic':
        m = Basemap(projection='merc', llcrnrlat=latBot, urcrnrlat=latTop,
                    llcrnrlon=lonLeft,
                    urcrnrlon=lonRight + 360, lat_ts=10,
                    resolution='c')
    else:
        m = Basemap(projection='merc', llcrnrlat=latBot, urcrnrlat=latTop,
                    llcrnrlon=lonLeft,
                    urcrnrlon=lonRight, lat_ts=10,
                    resolution='c')
    m.fillcontinents(color=dwtcolors[hh])
    if basin == 'atlantic':
        xGridCheck = xGrid
        indexChecker = np.where(xGridCheck < 180)
        xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
        cx, cy = m(xGridCheck, yGrid)
    else:
        cx, cy = m(xGrid, yGrid)

    m.drawcoastlines()
    CS = m.contourf(cx, cy, rectField, clevels, vmin=-18, vmax=18, cmap=cm.RdBu_r)  # , shading='gouraud')
    # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    tx, ty = m(320, 10)
    # ax.text(tx, ty, '{}'.format(groupSizeETC[num]))
    # inderTemp = np.where(np.asarray(stormBMUSall_corrected)==hh)
    inderTemp = np.where(biggerBmus==hh)

    if len(inderTemp)>0:
        ax.text(tx, ty, '{}'.format(len(inderTemp[0])))

    c2 += 1
    if c2 == int(np.ceil(np.sqrt(numClusters)) - 1):
        c1 += 1
        c2 = 0

    if plotIndx <= np.ceil(np.sqrt(numClusters)):
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy <= np.ceil(np.sqrt(numClusters)):
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
#
# for hh in range(num_clustersTC):
#     # p1 = plt.subplot2grid((6,6),(c1,c2))
#     ax = plt.subplot(gs1[hh+49])
#     num = hh#struct.kmaOrderTC[hh]
#
#     spatialField = Km_TC[(hh), 0:len(xFlat[~isOnLandFlat])] / 100 - SlpGrdMean[0:len(
#         xFlat[~isOnLandFlat])] / 100
#     # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))
#
#     if basin == 'atlantic':
#         X_in = xFlat[~isOnLandFlat]
#         X_in_Checker = np.where(X_in < 180)
#         X_in[X_in_Checker] = X_in[X_in_Checker] + 360
#
#     else:
#         X_in = xFlat[~isOnLandFlat]
#
#     Y_in = yFlat[~isOnLandFlat]
#     sea_nodes = []
#     for qq in range(len(X_in)):
#         sea_nodes.append(np.where((xGrid == X_in[qq]) & (yGrid == Y_in[qq])))
#
#     rectField = np.ones((np.shape(xGrid))) * np.nan
#     for tt in range(len(sea_nodes)):
#         rectField[sea_nodes[tt]] = spatialField[tt]
#
#     clevels = np.arange(-32, 32, 1)
#     # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
#     #             resolution='c')
#     if basin == 'atlantic':
#         m = Basemap(projection='merc', llcrnrlat=latBot, urcrnrlat=latTop,
#                     llcrnrlon=lonLeft,
#                     urcrnrlon=lonRight + 360, lat_ts=10,
#                     resolution='c')
#     else:
#         m = Basemap(projection='merc', llcrnrlat=latBot, urcrnrlat=latTop,
#                     llcrnrlon=lonLeft,
#                     urcrnrlon=lonRight, lat_ts=10,
#                     resolution='c')
#     m.fillcontinents(color=dwtcolors[hh+49])
#
#     if basin == 'atlantic':
#         xGridCheck = xGrid
#         indexChecker = np.where(xGridCheck < 180)
#         xGridCheck[indexChecker] = xGridCheck[indexChecker] + 360
#         cx, cy = m(xGridCheck, yGrid)
#     else:
#         cx, cy = m(xGrid, yGrid)
#
#     m.drawcoastlines()
#     CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
#     # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
#     tx, ty = m(320, 10)
#     # ax.text(tx, ty, '{}'.format(groupSizeTC[num]))
#     inderTemp = np.where(np.asarray(stormBMUSall)==(hh+49))
#     if len(inderTemp)>0:
#         ax.text(tx, ty, '{}'.format(len(inderTemp[0])))
#
#     c2 += 1
#     if c2 == int(np.ceil(np.sqrt(numClusters)) - 1):
#         c1 += 1
#         c2 = 0
#
#     if plotIndx <= np.ceil(np.sqrt(numClusters)):
#         ax.xaxis.set_ticks([])
#         ax.xaxis.set_ticklabels([])
#     if plotIndy > 0:
#         ax.yaxis.set_ticklabels([])
#         ax.yaxis.set_ticks([])
#     counter = counter + 1
#     if plotIndy <= np.ceil(np.sqrt(numClusters)):
#         plotIndy = plotIndy + 1
#     else:
#         plotIndy = 0
#         plotIndx = plotIndx + 1


