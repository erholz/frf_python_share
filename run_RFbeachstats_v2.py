import pickle
from math import sqrt
import numpy as np
from numpy import concatenate
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib.dates import DateFormatter
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import tensorboard
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Activation, Flatten, Input
from keras.utils import plot_model
import datetime as dt
# import pydot
# import visualkeras
import random
import scipy as sp
import pandas as pd  # to do datetime conversions
from numpy.random import seed
from funcs.create_contours import *
from funcs.calculate_beachvol import *
from funcs.interpgap import interpolate_with_max_gap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.dates as mdates
from sklearn import tree



########## LOAD DATA ##########

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'

with open(picklefile_dir+'preppedHydroTopobathy.pickle', 'rb') as file:
   lidar_xFRF, time_fullspan, _, _, _, hydro_fullspan = pickle.load(file)

with open(picklefile_dir+'PCAoutput.pickle', 'rb') as file:
   xplot, _, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV = pickle.load(file)

with open(picklefile_dir+'stormy_times_fullspan.pickle','rb') as file:
    _, stormy_fullspan, storm_timestart_all, storm_timeend_all = pickle.load(file)



dataNormT = dataNorm_fullspan.T
dataT = dataNormT * dataStd.T + dataMean.T
obs_profile = dataT.T
# fig, ax = plt.subplots()
# ax.plot(data_profile)

pca_profile = np.empty(shape=dataNorm_fullspan.shape)
for jj in np.arange(time_fullspan.size):
   mode1 = EOFs[0, :] * PCs_fullspan[jj,0]
   mode2 = EOFs[1, :] * PCs_fullspan[jj,1]
   mode3 = EOFs[2, :] * PCs_fullspan[jj,2]
   mode4 = EOFs[3, :] * PCs_fullspan[jj,3]
   mode5 = EOFs[4, :] * PCs_fullspan[jj,4]
   mode6 = EOFs[5, :] * PCs_fullspan[jj,5]
   mode7 = EOFs[6, :] * PCs_fullspan[jj,6]
   mode8 = EOFs[7, :] * PCs_fullspan[jj,7]
   profjj_norm = mode1 + mode2 + mode3 + mode4 + mode5 + mode6 + mode7 + mode8
   profjj_T = profjj_norm.T*dataStd.T + dataMean.T
   pca_profile[:,jj] = profjj_T.T
# fig, ax = plt.subplots()
# ax.plot(pca_profile)


########## CALCULATE BEACH STATS ##########

prof_fullspan = obs_profile
prof_fullspan[:,stormy_fullspan == 1] = np.nan

# Calculate beach volume and width
mlw = -0.62
mwl = -0.13
zero = 0
mhw = 0.36
dune_toe = 3.22
upper_lim = 5.95
cont_elev = np.array([mlw,mwl,zero,mhw,dune_toe]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(prof_fullspan.T,time_fullspan,xplot,cont_elev)
# beachVol, beachVol_xc, dBeachVol_dt, total_beachVol, total_dBeachVol_dt, total_obsBeachWid = calculate_beachvol(prof_fullspan.T,time_fullspan,lidar_xFRF,cont_elev,cont_ts)
# total_beachVol[total_beachVol == 0] = np.nan
mhw_xc = cont_ts[3,:]
mhw_xc_smoothed = np.convolve(mhw_xc,np.ones(12)/12,mode="same")
mhw_xc_smoothed[0:6] = np.nan
mlw_xc = cont_ts[0,:]
dunetoe_xc = cont_ts[4,:]
beachWid = mlw_xc - mhw_xc

fig, ax = plt.subplots()
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
# ax.plot(prof_fullspan)
ax.plot(tplot,mhw_xc,'o-',color='tab:orange',linewidth=0.5)
ax.plot(tplot,mhw_xc_smoothed,'.-',color='k',linewidth=0.5)


ws = 0.28
dean_fullspan = hydro_fullspan[1,:]/(hydro_fullspan[2,:]*ws)
fig, ax = plt.subplots()
# ax.plot(tplot,mhw_xc_smoothed)
ax.plot(tplot,PCs_fullspan[:,0])
ax2 = ax.twinx()
ax2.plot(tplot,dean_fullspan,'tab:red')


########## DEFINE INPUT DATA ##########

beachstat_fullspan = mhw_xc_smoothed[:]
beachstat_max = np.nanmax(beachstat_fullspan)
beachstat_min = np.nanmin(beachstat_fullspan)
beachstat_scaled = (beachstat_fullspan - beachstat_min) / (beachstat_max - beachstat_min)


########## FILTER OUT STORM TIMES ##########

beachstat_fullspan[(stormy_fullspan == 1)] = np.nan
hydro_fullspan[:,(stormy_fullspan == 1)] = np.nan


############### CREATE INPUT DATASETS W/ VARIABLE NLOOK ###############

numhydro = 4
nummorph = 3
numoutput = 1
num_features = 5
inputData = np.empty((1,num_features))*np.nan
outputData = np.empty((1,))*np.nan
ii_tplot = []
for tt in np.arange(13,time_fullspan.size-1):

    # get input
    ds_dean = dean_fullspan[tt]
    ds_deanm6 = dean_fullspan[tt-6]
    ds_deanm12 = dean_fullspan[tt-12]
    ds_beachstat = beachstat_fullspan[tt]
    ds_beachstatm6 = np.nanmean(beachstat_fullspan[tt - 9:tt-3])
    ds_beachstatm12 = np.nanmean(beachstat_fullspan[tt - 15:tt-9])
    ds_Hs = hydro_fullspan[1, tt]
    ds_Hsm6 = np.nanmean(hydro_fullspan[1, tt-9:tt-3])
    ds_Hsm12 = np.nanmean(hydro_fullspan[1, tt-15:tt-9])
    ds_Tp = hydro_fullspan[2, tt]
    ds_Tpm6 = np.nanmean(hydro_fullspan[2, tt - 9:tt-3])
    ds_Tpm12 = np.nanmean(hydro_fullspan[2, tt - 15:tt-9])
    # check for nans....
    # ds_data = np.column_stack((ds_Hs,ds_Tp,ds_Hsm6,ds_Tpm6,ds_beachstatm6,ds_Hsm12,ds_Tpm12,ds_beachstatm12,ds_beachstat))
    ds_data = np.column_stack((ds_Hs, ds_Hsm6, ds_beachstatm6, ds_Hsm12, ds_beachstatm12, ds_beachstat))
    if np.sum(np.isnan(ds_data)) == 0:
        # print(str(tt))
        input_newDS = np.empty((1,num_features))*np.nan
        input_newDS[:] = np.squeeze(ds_data[0,:-1])
        output_newDS = np.empty((1))*np.nan
        output_newDS[:] = ds_data[0,-1]
        inputData = np.append(inputData,input_newDS,axis=0)
        outputData = np.append(outputData, output_newDS,axis=0)
        ii_tplot = np.append(ii_tplot,tt)

inputData = inputData[1:,:]
outputData = outputData[1:]


############### SPLIT DATA INTO TEST/TRAIN ###############

# separate test and train IDs
frac = 0.65          # num used for training
num_datasets = inputData.shape[0]
Ntrain = int(np.floor(num_datasets*frac))
Ntest = num_datasets - Ntrain
tmpii = random.sample(range(num_datasets), Ntrain)
iitrain = np.isin(np.arange(num_datasets),tmpii)
iitest = ~iitrain
# load training
train_X = np.empty((Ntrain,num_features))
train_X[:,:] = inputData[iitrain,:]
train_y = np.empty((Ntrain,))
train_y[:] = outputData[iitrain]
# load testing
test_X = np.empty((Ntest,num_features))
test_X[:,:] = inputData[iitest,:]
test_y = np.empty((Ntest,))
test_y[:] = outputData[iitest]


############### DESIGN AND FIT NETWORK ###############

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(train_X,train_y)
ypred = regressor.predict(test_X)

fig, ax = plt.subplots()
ax.plot(tplot,beachstat_fullspan,'.-')
ax.plot(tplot[ii_tplot[iitest].astype(int)],ypred,'.-')

fig, ax = plt.subplots()
yplot = np.empty(shape=time_fullspan.shape)*np.nan
yplot[ii_tplot[iitest].astype(int)] = ypred
ax.plot(beachstat_fullspan-yplot)
# ax.set_ylim(-10,20)

fig, ax = plt.subplots()
ax.plot([np.nanmin(beachstat_fullspan),np.nanmax(beachstat_fullspan)],[np.nanmin(beachstat_fullspan),np.nanmax(beachstat_fullspan)],'k')
ax.plot(test_y,ypred,'o',alpha=0.1)
ax.grid()
ax.set_xlabel('observed $Xc_{MHW}$ [m]')
ax.set_ylabel('predicted $Xc_{MHW}$ [m]')
rmse = np.sqrt(np.nanmean((test_y-ypred)**2))
ax.set_title('RMSE = '+ str("%0.1f" % rmse) + ' m')
fig.tight_layout()

############### INVESTIGATE MODEL ###############

# feature_names = [f"feature {i}" for i in range(7)]
# feature_names = ['$Hs$','$Tp$','$Hs_{t-6}$','$Tp_{t-6}$','$X_{t-6}$','$Hs_{t-12}$','$Tp_{t-12}$','$X_{t-12}$']
feature_names = ['$Hs$','$Hs_{t-6}$','$X_{t-6}$','$Hs_{t-12}$','$X_{t-12}$']

importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
forest_importances = pd.Series(importances)
fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
ax.bar(np.arange(num_features),importances)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
ax.set_yscale('log')
ax.set_xticks(np.arange(num_features))
ax.set_xticklabels(feature_names)

fig, ax = plt.subplots()
tree.plot_tree(regressor.estimators_[0])


############### TEST PREDICITION ###############

plotflag = True
nnplot = [1,4,5,6,7,13,14,15,19,22,23,24,26,27,29,30,31,33,34,41,42,43,44,47,49]
xxplot = np.arange(time_fullspan.size)
# for nn in nnplot[:5]:
# for nn in np.arange(45,50):
for nn in np.arange(0,storm_timestart_all.size):
# for nn in np.arange(storm_timeend_all.size):

    tstart = storm_timeend_all[nn] + int(2 * 24 * 3600 )
    iistart = np.where(np.isin(time_fullspan, tstart))[0].astype(int)

    # SHORT_TERM PREDICTION
    Npred = int(30*24)

    prev_pred = np.empty((Npred,)) * np.nan
    prev_obs = np.empty((Npred,)) * np.nan
    prev_obs_unsmooth = np.empty((Npred,)) * np.nan
    numnan_hydro = np.empty((Npred,)) * np.nan
    for tt in np.arange(Npred):

        ttlook = iistart + tt - 1

        # get input
        ds_dean = dean_fullspan[ttlook]
        ds_deanm6 = dean_fullspan[ttlook - 6]
        ds_deanm12 = dean_fullspan[ttlook - 12]
        ds_beachstat = beachstat_fullspan[ttlook]
        ds_Hs = hydro_fullspan[1, ttlook]
        ds_Hsm6 = np.nanmean(hydro_fullspan[1, np.arange(ttlook - 9,ttlook - 3)])
        ds_Hsm12 = np.nanmean(hydro_fullspan[1, np.arange(ttlook - 15,ttlook - 9)])
        ds_Tp = hydro_fullspan[2, ttlook]
        ds_Tpm6 = np.nanmean(hydro_fullspan[2, np.arange(ttlook - 9,ttlook - 3)])
        ds_Tpm12 = np.nanmean(hydro_fullspan[2, np.arange(ttlook - 15,ttlook - 9)])
        # check for nans....
        if tt <= 15:
            ds_beachstatm6 = np.nanmean(beachstat_fullspan[np.arange(ttlook - 9,ttlook - 3)])
            ds_beachstatm12 = np.nanmean(beachstat_fullspan[np.arange(ttlook - 15,ttlook - 9)])
        elif tt > 15:
            ds_beachstatm6 = np.nanmean(prev_pred[np.arange(tt - 9,tt - 3)])
            ds_beachstatm12 = np.nanmean(prev_pred[np.arange(tt - 15,tt - 9)])
        # check for nans....
        # ds_data = np.column_stack((ds_Hs, ds_Tp, ds_Hsm6, ds_Tpm6, ds_beachstatm6, ds_Hsm12, ds_Tpm12, ds_beachstatm12))
        ds_data = np.column_stack((ds_Hs, ds_Hsm6, ds_beachstatm6, ds_Hsm12, ds_beachstatm12))
        if np.sum(np.isnan(ds_data)) == 0:
            input_newDS = np.empty((1, num_features)) * np.nan
            input_newDS[:] = np.squeeze(ds_data[0, :])
            output_newDS = np.empty((1)) * np.nan
            output_newDS[:] = ds_data[0, -1]
            test_X = input_newDS[:]
            ypred = regressor.predict(test_X)
            prev_pred[tt] = ypred
            prev_obs[tt] = ds_beachstat
            prev_obs_unsmooth[tt] = mhw_xc[ttlook+1]
    if sum(~np.isnan(prev_pred)) >= 20:
        ttmp = storm_timestart_all[nn] - int(6 * 3600)
        iitmp = np.where(np.isin(time_fullspan, ttmp))[0].astype(int)
        tplotpre = tplot[np.arange(iitmp,iistart-1)]
        yplot1 = mhw_xc[np.arange(iitmp,iistart-1)]
        yplot2 = mhw_xc_smoothed[np.arange(iitmp, iistart - 1)]
        tplotpost = tplot[np.arange(iistart,iistart+Npred)]
        tplotall = tplot[np.arange(iitmp,iistart+Npred)]
        watlevplot = hydro_fullspan[0,np.arange(iitmp,iistart+Npred)]
        Hsplot = hydro_fullspan[1,np.arange(iitmp,iistart+Npred)]
        Tpplot = hydro_fullspan[2,np.arange(iitmp,iistart+Npred)]
        dirplot = hydro_fullspan[3,np.arange(iitmp,iistart+Npred)]
        if len(tplotpre) > 0:
            t0 = pd.to_datetime(storm_timeend_all[nn], unit='s', origin='unix')
            fig, ax = plt.subplots()
            ax.plot(tplotpre, yplot1, 'o', color='0.7')
            ax.plot(tplotpre, yplot2, 'k')
            ax.plot(tplotpost, prev_obs_unsmooth, 'o', color='0.7')
            ax.plot(tplotpost, prev_obs, 'k')
            ax.plot(tplotpost, prev_pred, 'xm', alpha=0.5)
            pred_smooth = np.convolve(prev_pred, np.ones(12) / 12, mode="valid")
            ax.plot(tplotpost[5:-6], pred_smooth, '-r', linewidth=2, alpha=0.5)
            # ax.set_title(str(tplot[iistart]))
            x1 = pd.to_datetime(storm_timestart_all[nn], unit='s', origin='unix')
            x2 = pd.to_datetime(storm_timeend_all[nn], unit='s', origin='unix')
            y1, y2 = (np.nanmin(prev_obs_unsmooth) - 7, np.nanmax(prev_obs_unsmooth) + 7)
            left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
            patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='c')
            ax.add_patch(patch)
            ax.set_ylim(y1, y2)
            ax.grid()
            ax.set_ylabel('$Xc_{MHW}$ [m]')
            # fig.set_size_inches(8.5, 2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.set_title(tplotpre[0].strftime("%d %B %Y"))
            fig.set_size_inches(6.5, 2)
            fig.tight_layout()
            # fig.tight_layout()
            # fig, ax = plt.subplots(3,1)
            # ax[0].plot(tplotpre,yplot1,'o',color='0.7')
            # ax[0].plot(tplotpre, yplot2,'k')
            # ax[0].plot(tplotpost, prev_obs_unsmooth, 'o',color='0.7')
            # ax[0].plot(tplotpost,prev_obs,'k')
            # ax[0].plot(tplotpost,prev_pred,'xm',linewidth=40)
            # pred_smooth = np.convolve(prev_pred,np.ones(12)/12,mode="valid")
            # ax[0].plot(tplotpost[5:-6],pred_smooth,'-r',linewidth=2,alpha=0.5)
            # # ax.set_title(str(tplot[iistart]))
            # x1 = pd.to_datetime(storm_timestart_all[nn], unit='s', origin='unix')
            # x2 = pd.to_datetime(storm_timeend_all[nn], unit='s', origin='unix')
            # y1, y2 = (np.nanmin(prev_obs_unsmooth)-7, np.nanmax(prev_obs_unsmooth)+7)
            # left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
            # patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='c')
            # ax[0].add_patch(patch)
            # ax[0].set_ylim(y1, y2)
            # ax[0].set_xticklabels([])
            # ax[0].grid()
            # ax[0].set_ylabel('$Xc_{MHW}$ [m]')
            # ax[1].plot(tplotall,watlevplot,'o-')
            # ax[1].set_ylabel('$\\overline{\\eta}$ [m]')
            # ax[1].set_xticklabels([])
            # ax2 = ax[1].twinx()
            # ax2.plot(tplotall,Hsplot,'s-',color='tab:orange')
            # ax2.set_ylabel('$H_s$ [m]')
            # ax[2].plot(tplotall,Tpplot,'*-',color='tab:red')
            # ax[2].set_ylabel('$T_p$ [s]')
            # ax3 = ax[2].twinx()
            # ax3.plot(tplotall,dirplot,'^',color='tab:purple')
            # fig.set_size_inches(8.5,6)
            # ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            # ax3.set_ylabel('$\\theta_p$ [deg]')
            # fig.tight_layout()




