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
local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'

# wave_base17 = 'waverider-17m/'
wave_base17 = 'waves_17mwaverider/'
files17 = sorted((f for f in os.listdir(local_base+wave_base17) if not f.startswith(".")), key=str.lower) #os.listdir(local_base+wave_base17)
# files17.sort()
files_path17 = [os.path.join(os.path.abspath(local_base+wave_base17), x) for x in files17]

# wave_base26 = 'waverider-26m/'
wave_base26 = 'waves_26mwaverider/'
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

recentInds = np.where(np.asarray(tWave17) > DT.datetime(2015,1,1))

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

recentInds26 = np.where(np.asarray(tWave26) > DT.datetime(2015,1,1))

hs26 = hs26[recentInds26]
tp26 = tp26[recentInds26]
dir26 = dir26[recentInds26]
tWave26 = np.asarray(tWave26)[recentInds26]

timeWave26 = [round_to_nearest_half_hour(tt) for tt in tWave26]

from dateutil.relativedelta import relativedelta
st = dt.datetime(2015, 1, 1)
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
cDp = waveNorm
# offpos = np.where((waveNorm>90))
# offneg = np.where((waveNorm<-90))
# waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
# waveNorm[offneg[0]] = waveNorm[offneg[0]]*0

# def wavetransform_point(H0, theta0, H1, theta1, T, h2, h1, g, breakcrit):



# wavedirWIS = '/volumes/macDrive/WIS63218/'
wavedirWIS = 'waves_WIS63218/'

# # Need to sort the files to ensure correct temporal order...
# filesWIS = os.listdir(local_base+wavedirWIS)
# filesWIS.sort()
# files_pathWIS = [os.path.join(os.path.abspath(wavedirWIS), x) for x in filesWIS][1:]


wave_baseWIS = 'waves_WIS63218/'
filesWIS = sorted((f for f in os.listdir(local_base+wave_baseWIS) if not f.startswith(".")), key=str.lower)#os.listdir(local_base+wave_baseWIS)
# filesWIS.sort()
files_pathWIS = [os.path.join(os.path.abspath(local_base+wave_baseWIS), x) for x in filesWIS]


# wis = Dataset(files_path[0])

def getWIS(file):
    waves = Dataset(file)

    waveHs = waves.variables['waveHs'][:]
    waveTp = waves.variables['waveTp'][:]
    waveMeanDirection = waves.variables['waveMeanDirection'][:]

    waveTm = waves.variables['waveTm'][:]
    waveTm1 = waves.variables['waveTm1'][:]
    waveTm2 = waves.variables['waveTm2'][:]

    waveHsWindsea = waves.variables['waveHsWindsea'][:]
    waveTmWindsea = waves.variables['waveTmWindsea'][:]
    waveMeanDirectionWindsea = waves.variables['waveMeanDirectionWindsea'][:]
    waveSpreadWindsea = waves.variables['waveSpreadWindsea'][:]

    timeW = waves.variables['time'][:]

    waveTpSwell = waves.variables['waveTpSwell'][:]
    waveHsSwell = waves.variables['waveHsSwell'][:]
    waveMeanDirectionSwell = waves.variables['waveMeanDirectionSwell'][:]
    waveSpreadSwell = waves.variables['waveSpreadSwell'][:]


    output = dict()
    output['waveHs'] = waveHs
    output['waveTp'] = waveTp
    output['waveMeanDirection'] = waveMeanDirection

    output['waveTm'] = waveTm
    output['waveTm1'] = waveTm1
    output['waveTm2'] = waveTm2

    output['waveTpSwell'] = waveTpSwell
    output['waveHsSwell'] = waveHsSwell
    output['waveMeanDirectionSwell'] = waveMeanDirectionSwell
    output['waveSpreadSwell'] = waveSpreadSwell

    output['waveHsWindsea'] = waveHsWindsea
    output['waveTpWindsea'] = waveTmWindsea
    output['waveMeanDirectionWindsea'] = waveMeanDirectionWindsea
    output['waveSpreadWindsea'] = waveSpreadWindsea

    output['t'] = timeW

    return output

HsWIS = []
TpWIS = []
DmWIS = []
hsSwellWIS = []
tpSwellWIS = []
dmSwellWIS = []
hsWindseaWIS = []
tpWindseaWIS = []
dmWindseaWIS = []

timeWaveWIS = []
for i in files_pathWIS:
    wavesWIS = getWIS(i)
    HsWIS = np.append(HsWIS,wavesWIS['waveHs'])
    TpWIS = np.append(TpWIS,wavesWIS['waveTp'])
    DmWIS = np.append(DmWIS,wavesWIS['waveMeanDirection'])
    hsSwellWIS = np.append(hsSwellWIS,wavesWIS['waveHsSwell'])
    tpSwellWIS = np.append(tpSwellWIS,wavesWIS['waveTpSwell'])
    dmSwellWIS = np.append(dmSwellWIS,wavesWIS['waveMeanDirectionSwell'])
    hsWindseaWIS = np.append(hsWindseaWIS,wavesWIS['waveHsWindsea'])
    tpWindseaWIS = np.append(tpWindseaWIS,wavesWIS['waveTpWindsea'])
    dmWindseaWIS = np.append(dmWindseaWIS,wavesWIS['waveMeanDirectionWindsea'])
    #timeTemp = [datenum_to_datetime(x) for x in waves['t'].flatten()]
    timeWaveWIS = np.append(timeWaveWIS,wavesWIS['t'].flatten())



tWaveWIS = [DT.datetime.fromtimestamp(x) for x in timeWaveWIS]
copyOverIndex = np.where(cTime>tWaveWIS[-1])

waveNormWIS = DmWIS - 72
negWIS = np.where((waveNormWIS > 180))
waveNormWIS[negWIS[0]] = waveNormWIS[negWIS[0]]-360

combinedHsWIS = np.hstack((HsWIS,cHs[copyOverIndex]))
combinedTpWIS = np.hstack((TpWIS,cTp[copyOverIndex]))
combinedDmWIS = np.hstack((waveNormWIS,waveNorm[copyOverIndex]))
combinedTimeWIS = np.hstack((tWaveWIS,cTime[copyOverIndex]))


picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'stormWaves_WISandFRF.pickle','wb') as file:
    pickle.dump([combinedHsWIS,combinedTpWIS,combinedDmWIS,combinedTimeWIS],file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'stormWaves_FRF.pickle','wb') as file:
    pickle.dump([cHs,cTp,cDp,cTime],file)
