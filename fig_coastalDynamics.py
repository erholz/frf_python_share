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


picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_26Nov2024/'
# picklefile_dir = './'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
with open(picklefile_dir+'waves_8m&17m_2015_2024.pickle','rb') as file:
    [data_wave8m,data_wave17m,data_wave8m_filled] = pickle.load(file)
with open(picklefile_dir+'stormHs95_Over12Hours.pickle','rb') as f:
    output = pickle.load(f)
list(output)
storm_start = np.array(output['startTimeStormList'])
storm_end = np.array(output['endTimeStormList'])
storm_startWIS = np.array(output['startTimeStormListWIS'])
storm_endWIS = np.array(output['endTimeStormListWIS'])


fig, ax = plt.subplots(3,1)
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
Hs = data_wave8m_filled[:,0]
Tp = data_wave8m_filled[:,1]
ax[0].plot(tplot,Hs)
ax[1].plot(tplot,Tp)
ax[2].plot(tplot,(Hs**2)*Tp,'.')

fig, ax = plt.subplots(2,1)
yplot = total_beachVol_pca
nsmooth = 12
ymean = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
ts = pd.Series(yplot)
ystd = ts.rolling(window=nsmooth, center=True).std()
bad_id = (abs(yplot - ymean) >= 3 * ystd)
yplot[bad_id] = np.nan
ysmooth = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
ax[0].plot(tplot,yplot)
yplot = total_obsBeachWid_pca
nsmooth = 12
ymean = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
ts = pd.Series(yplot)
ystd = ts.rolling(window=nsmooth, center=True).std()
bad_id = (abs(yplot - ymean) >= 3 * ystd)
yplot[bad_id] = np.nan
ysmooth = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
ax[1].plot(tplot,yplot)


