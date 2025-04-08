import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

stormPickle = 'stormHs95Over12Hours.pickle'
with open(stormPickle, "rb") as input_file:
    inputStorms = pickle.load(input_file)
timeStormList = inputStorms['timeStormList']
hsStormList = inputStorms['hsStormList']
hsMaxStormList = inputStorms['hsMaxStormList']
tpStormList = inputStorms['tpStormList']
dmStormList = inputStorms['dmStormList']
hourStormList = inputStorms['hourStormList']
indStormList = inputStorms['indStormList']
durationStormList = inputStorms['durationStormList']
wavePowerStormList = inputStorms['wavePowerStormList']
longshorePowerStormList = inputStorms['longshorePowerStormList']
startTimeStormList = inputStorms['startTimeStormList']
endTimeStormList = inputStorms['endTimeStormList']
cHs = inputStorms['cHs']
cTp = inputStorms['cTp']
cDp = inputStorms['cDp']
waveNorm = inputStorms['waveNorm']
cTime = inputStorms['cTime']

plt.figure()
plt.plot(cTime,cHs)
# timeStormList = inputStorms['timeStormListWIS']
# hsStormList = inputStorms['hsStormListWIS']
# hsMaxStormList = inputStorms['hsMaxStormListWIS']
# tpStormList = inputStorms['tpStormListWIS']
# dmStormList = inputStorms['dmStormListWIS']
# hourStormList = inputStorms['hourStormListWIS']
# indStormList = inputStorms['indStormListWIS']
# durationStormList = inputStorms['durationStormListWIS']
# wavePowerStormList = inputStorms['wavePowerStormListWIS']
# longshorePowerStormList = inputStorms['longshorePowerStormListWIS']
# startTimeStormList = inputStorms['startTimeStormListWIS']
# endTimeStormList = inputStorms['endTimeStormListWIS']
# cHs = inputStorms['combinedHsWIS']
# cTp = inputStorms['combinedTpWIS']
# cDp = inputStorms['combinedDmWIS']
# waveNorm = inputStorms['combinedDmWIS']
# cTime = inputStorms['combinedTimeWIS']


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





morphPickle = 'tidalAveragedMetrics.pickle'
with open(morphPickle, "rb") as input_file:
    inputMorph = pickle.load(input_file)
interpTimeS = inputMorph['interpTimeS']
eofSubaerial = inputMorph['eofSubaerial']
lidar_xFRF = inputMorph['lidar_xFRF']
interpX = inputMorph['interpX']
mslContour = inputMorph['mslContour']
highTideTimes = inputMorph['highTideTimes']
highTideUTCtime = inputMorph['highTideUTCtime']
mhwContour = inputMorph['mhwContour']
tidalAverageWithData = inputMorph['tidalAverageWithData']
tidalAverage = inputMorph['tidalAverage']
duneContour = inputMorph['duneContour']
mhhwContour = inputMorph['mhhwContour']
smoothUpperTidalAverage = inputMorph['smoothUpperTidalAverage']
mslVolume = inputMorph['mslVolume']
mhhwVolume = inputMorph['mhhwVolume']
from loess import loess_1d
from datetime import datetime
plt.figure()
p1 = plt.subplot2grid((4,2),(0,0))
p1Index = np.where((highTideTimes>datetime(2017,7,25)) & (highTideTimes<datetime(2017,8,25)))
l1 = p1.plot(highTideTimes[p1Index],mhhwContour[p1Index]-109,label='Dune Lidar')
p1timeSubset = highTideUTCtime[p1Index]
p1mhhwSubset = mhhwContour[p1Index]
badData = np.where(~np.isnan(p1mhhwSubset))
p1timeSubset2 = p1timeSubset[badData]
p1mhhwSubset2 = p1mhhwSubset[badData]
xout, yout, wout = loess_1d.loess_1d(p1timeSubset2, p1mhhwSubset2, xnew=p1timeSubset, degree=1, frac=0.35,npoints=None, rotate=False, sigy=None)
xoutDatetime = np.asarray([datetime.fromtimestamp(x) for x in xout])
l2 = p1.plot(xoutDatetime[15:],yout[15:]-109,'k--',linewidth=2,label='Smoothed Recovery')
p1.set_xlim([datetime(2017,7,25),datetime(2017,9,25)])
p1.set_ylabel('Pre-storm MHW X (m)')
p1.set_xticks([datetime(2017,7,29),datetime(2017,8,13),datetime(2017,8,28)])
p1.set_xticklabels(['0','15','30'])
p1.set_xlabel('Days')
p1.set_title('7/29/2017: $Hs_{max}$ = 2.52 m')
plt.legend()

p4 = plt.subplot2grid((4,2),(1,0))
p4Index = np.where((highTideTimes>datetime(2019,7,20)) & (highTideTimes<datetime(2019,8,25)))
p4.plot(highTideTimes[p4Index],mhhwContour[p4Index]-112)
p4timeSubset = highTideUTCtime[p4Index]
p4mhhwSubset = mhhwContour[p4Index]
badData4 = np.where(~np.isnan(p4mhhwSubset))
p4timeSubset2 = p4timeSubset[badData4]
p4mhhwSubset2 = p4mhhwSubset[badData4]
xout4, yout4, wout = loess_1d.loess_1d(p4timeSubset2, p4mhhwSubset2, xnew=p4timeSubset, degree=1, frac=0.35,npoints=None, rotate=False, sigy=None)
xoutDatetime4 = np.asarray([datetime.fromtimestamp(x) for x in xout4])
p4.plot(xoutDatetime4[15:],yout4[15:]-112,'k--',linewidth=2)
p4.set_xlim([datetime(2019,7,20),datetime(2019,9,20)])
p4.set_ylabel('Pre-storm MHW X (m)')
p4.set_xticks([datetime(2019,7,24),datetime(2019,8,8),datetime(2019,8,23)])
p4.set_xticklabels(['0','15','30'])
p4.set_xlabel('Days')
p4.set_title('7/25/2019: $Hs_{max}$ = 1.56 m')

p2 = plt.subplot2grid((4,2),(2,0))
p2Index = np.where((highTideTimes>datetime(2018,1,25)) & (highTideTimes<datetime(2018,3,20)))
p2.plot(highTideTimes[p2Index],mhhwContour[p2Index]-120)
p2timeSubset = highTideUTCtime[p2Index]
p2mhhwSubset = mhhwContour[p2Index]
badData2 = np.where(~np.isnan(p2mhhwSubset))
p2timeSubset2 = p2timeSubset[badData2]
p2mhhwSubset2 = p2mhhwSubset[badData2]
xout2, yout2, wout = loess_1d.loess_1d(p2timeSubset2, p2mhhwSubset2, xnew=p2timeSubset, degree=1, frac=0.35,npoints=None, rotate=False, sigy=None)
xoutDatetime2 = np.asarray([datetime.fromtimestamp(x) for x in xout2])
p2.plot(xoutDatetime2[17:],yout2[17:]-120,'k--',linewidth=2)
p2.set_xlim([datetime(2018,1,25),datetime(2018,3,25)])
p2.set_ylabel('Pre-storm MHW X (m)')
p2.set_xticks([datetime(2018,1,28,20),datetime(2018,2,12),datetime(2018,2,26,6),datetime(2018,3,14)])
p2.set_xticklabels(['0','15','30','45'])
p2.set_xlabel('Days')
p2.set_title('1/29/2018: $Hs_{max}$ = 3.04 m')

p3 = plt.subplot2grid((4,2),(3,0))
p3Index = np.where((highTideTimes>datetime(2024,1,23)) & (highTideTimes<datetime(2024,3,23)))
p3.plot(highTideTimes[p3Index],mhhwContour[p3Index]-130)
p3timeSubset = highTideUTCtime[p3Index]
p3mhhwSubset = mhhwContour[p3Index]
badData3 = np.where(~np.isnan(p3mhhwSubset))
p3timeSubset2 = p3timeSubset[badData3]
p3mhhwSubset2 = p3mhhwSubset[badData3]
xout3, yout3, wout = loess_1d.loess_1d(p3timeSubset2, p3mhhwSubset2, xnew=p3timeSubset, degree=1, frac=0.35,npoints=None, rotate=False, sigy=None)
xoutDatetime3 = np.asarray([datetime.fromtimestamp(x) for x in xout3])
p3.plot(xoutDatetime3[18:],yout3[18:]-130,'k--',linewidth=2)
p3.set_xlim([datetime(2024,1,23),datetime(2024,3,23)])
p3.set_ylabel('Pre-storm MHW X (m)')
p3.set_xticks([datetime(2024,1,26,20),datetime(2024,2,10),datetime(2024,2,24,6),datetime(2024,3,12)])
p3.set_xticklabels(['0','15','30','45'])
p3.set_xlabel('Days')
p3.set_title('2/1/2024: $Hs_{max}$ = 2.7 m')



p5 = plt.subplot2grid((4,2),(2,1))
p5Index = np.where((highTideTimes>datetime(2018,3,1)) & (highTideTimes<datetime(2018,3,11)))
p5.plot(highTideTimes[p5Index],mhhwContour[p5Index]-115)
p5timeSubset = highTideUTCtime[p5Index]
p5mhhwSubset = mhhwContour[p5Index]
badData5 = np.where(~np.isnan(p5mhhwSubset))
p5timeSubset2 = p5timeSubset[badData5]
p5mhhwSubset2 = p5mhhwSubset[badData5]
xout5, yout5, wout = loess_1d.loess_1d(p5timeSubset2, p5mhhwSubset2, xnew=p5timeSubset, degree=1, frac=0.35,npoints=None, rotate=False, sigy=None)
xoutDatetime5 = np.asarray([datetime.fromtimestamp(x) for x in xout5])
p5.plot(xoutDatetime5[6:],yout5[6:]-115,'k--',linewidth=2)
p5.set_xlim([datetime(2018,2,28),datetime(2018,4,28)])
p5.set_ylabel('Pre-storm MHW X (m)')
p5.set_xticks([datetime(2018,3,2),datetime(2018,3,17)])
p5.set_xticklabels(['0','15'])
p5.set_title('3/4/2018: $Hs_{max}$ = 5.44 m')
p5.set_xlabel('Days')

p6 = plt.subplot2grid((4,2),(3,1))
p6Index = np.where((highTideTimes>datetime(2023,8,29)) & (highTideTimes<datetime(2023,9,12)))
p6.plot(highTideTimes[p6Index],mhhwContour[p6Index]-105)
p6timeSubset = highTideUTCtime[p6Index]
p6mhhwSubset = mhhwContour[p6Index]
badData6 = np.where(~np.isnan(p6mhhwSubset))
p6timeSubset2 = p6timeSubset[badData6]
p6mhhwSubset2 = p6mhhwSubset[badData6]
xout6, yout6, wout = loess_1d.loess_1d(p6timeSubset2, p6mhhwSubset2, xnew=p6timeSubset, degree=1, frac=0.35,npoints=None, rotate=False, sigy=None)
xoutDatetime6 = np.asarray([datetime.fromtimestamp(x) for x in xout6])
p6.plot(xoutDatetime6[6:],yout6[6:]-105,'k--',linewidth=2)
p6.set_xlim([datetime(2023,8,28),datetime(2023,10,28)])
p6.set_ylabel('Pre-storm MHW X (m)')
p6.set_xticks([datetime(2023,8,31),datetime(2023,9,14)])
p6.set_xticklabels(['0','15'])
p6.set_title('9/1/2023: $Hs_{max}$ = 4.02 m')
p6.set_xlabel('Days')

p7 = plt.subplot2grid((4,2),(1,1))
p7Index = np.where((highTideTimes>datetime(2019,9,5)) & (highTideTimes<datetime(2019,10,21)))
p7.plot(highTideTimes[p7Index],mhhwContour[p7Index]-108)
p7timeSubset = highTideUTCtime[p7Index]
p7mhhwSubset = mhhwContour[p7Index]
badData7 = np.where(~np.isnan(p7mhhwSubset))
p7timeSubset2 = p7timeSubset[badData7]
p7mhhwSubset2 = p7mhhwSubset[badData7]
xout7, yout7, wout = loess_1d.loess_1d(p7timeSubset2, p7mhhwSubset2, xnew=p7timeSubset, degree=1, frac=0.25,npoints=None, rotate=False, sigy=None)
xoutDatetime7 = np.asarray([datetime.fromtimestamp(x) for x in xout7])
p7.plot(xoutDatetime7[4:],yout7[4:]-108,'k--',linewidth=2)
p7.set_xlim([datetime(2019,9,4),datetime(2019,11,5)])
p7.set_ylabel('Pre-storm MHW X (m)')
p7.set_xticks([datetime(2019,9,6),datetime(2019,9,21)])
p7.set_xticklabels(['0','15'])
p7.set_title('9/6/2019: $Hs_{max}$ = 6.9 m')
p7.set_xlabel('Days')

p8 = plt.subplot2grid((4,2),(0,1))
p8Index = np.where((highTideTimes>datetime(2020,12,24)) & (highTideTimes<datetime(2021,1,22)))
p8.plot(highTideTimes[p8Index],mhhwContour[p8Index]-85)
p8timeSubset = highTideUTCtime[p8Index]
p8mhhwSubset = mhhwContour[p8Index]
badData8 = np.where(~np.isnan(p8mhhwSubset))
p8timeSubset2 = p8timeSubset[badData8]
p8mhhwSubset2 = p8mhhwSubset[badData8]
xout8, yout8, wout = loess_1d.loess_1d(p8timeSubset2, p8mhhwSubset2, xnew=p8timeSubset, degree=1, frac=0.25,npoints=None, rotate=False, sigy=None)
xoutDatetime8 = np.asarray([datetime.fromtimestamp(x) for x in xout8])
p8.plot(xoutDatetime8[4:],yout8[4:]-85,'k--',linewidth=2)
p8.set_xlim([datetime(2020,12,21),datetime(2021,2,21)])
p8.set_ylabel('Pre-storm MHW X (m)')
p8.set_xticks([datetime(2020,12,24),datetime(2021,1,8)])
p8.set_xticklabels(['0','15'])
p8.set_title('12/24/2020: $Hs_{max}$ = 4.9 m')
p8.set_xlabel('Days')

p8.set_ylim([-30,5])
p7.set_ylim([-30,5])
p6.set_ylim([-30,5])
p5.set_ylim([-30,5])
p4.set_ylim([-30,5])
p3.set_ylim([-30,5])
p2.set_ylim([-30,5])
p1.set_ylim([-30,5])


plt.figure()
plt.plot(highTideTimes,mhhwContour)
badDataALL = np.where(~np.isnan(mhhwContour))
highTideUTCtime2 = highTideUTCtime[badDataALL]
mhhwContour2 = mhhwContour[badDataALL]
xoutAll, youtAll, wout = loess_1d.loess_1d(highTideUTCtime2, mhhwContour2, xnew=highTideUTCtime, degree=1, frac=0.2,npoints=12, rotate=False, sigy=None)
xoutDatetimeAll = np.asarray([datetime.fromtimestamp(x) for x in xoutAll])
plt.plot(xoutDatetimeAll,youtAll,'k--',linewidth=2)