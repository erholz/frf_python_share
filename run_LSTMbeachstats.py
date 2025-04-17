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

prof_fullspan = pca_profile
prof_fullspan[:,stormy_fullspan == 1] = np.nan

# Calculate beach volume and width
mlw = -0.62
mwl = -0.13
zero = 0
mhw = 0.36
dune_toe = 3.22
upper_lim = 5.95
cont_elev = np.array([mlw,mwl,mhw,dune_toe]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(prof_fullspan.T,time_fullspan,xplot,cont_elev)
beachVol, beachVol_xc, dBeachVol_dt, total_beachVol, total_dBeachVol_dt, total_obsBeachWid = calculate_beachvol(prof_fullspan.T,time_fullspan,xplot,cont_elev,cont_ts)
total_beachVol[total_beachVol == 0] = np.nan
mhw_xc = cont_ts[2,:]
mlw_xc = cont_ts[0,:]
dunetoe_xc = cont_ts[3,:]
beachWid = mlw_xc - dunetoe_xc


########## DEFINE AND SCALE INPUT DATA ##########

beachstat_fullspan = total_beachVol[:]
beachstat_max = np.nanmax(beachstat_fullspan)
beachstat_min = np.nanmin(beachstat_fullspan)
beachstat_scaled = (beachstat_fullspan - beachstat_min) / (beachstat_max - beachstat_min)
hydro_scaled = np.empty(shape=hydro_fullspan.shape)*np.nan
hydro_min = np.empty((4,))
hydro_max = np.empty((4,))
hydro_avg = np.empty((4,))
hydro_stdev = np.empty((4,))
for nn in np.arange(4):
    unscaled = hydro_fullspan[nn,:]
    hydro_min[nn] = np.nanmin(unscaled)
    hydro_max[nn] = np.nanmax(unscaled)
    hydro_avg[nn] = np.nanmean(unscaled)
    hydro_stdev[nn] = np.nanstd(unscaled)
    hydro_scaled[nn,:] = (unscaled - hydro_min[nn]) / (hydro_max[nn] - hydro_min[nn])


############### CREATE INPUT DATASETS W/ VARIABLE NLOOK ###############

Nlook = 4
num_steps = Nlook-1
numhydro = 4
numPCs = 1
num_features = numhydro + numPCs

inputData = np.empty((1,num_steps,num_features))*np.nan
outputData = np.empty((1,numPCs))*np.nan
for tt in np.arange(time_fullspan.size-num_steps):

    ttlook = np.arange(tt,tt + Nlook)

    # get input hydro
    ds_watlev = hydro_scaled[0,ttlook]
    ds_Hs = hydro_scaled[1,ttlook]
    ds_Tp = hydro_scaled[2,ttlook]
    ds_wdir = hydro_scaled[3,ttlook]
    # get input PC amplitudes
    ds_beachstat = beachstat_scaled[ttlook]
    # check for nans....
    ds_data = np.column_stack((ds_watlev.T,ds_Hs.T,ds_Tp.T,ds_wdir.T,ds_beachstat))
    if np.sum(np.isnan(ds_data)) == 0:
        # print(str(tt))
        input_newDS = np.empty((1,num_steps,num_features))*np.nan
        input_newDS[0,:,:] = ds_data[:-1,:]
        output_newDS = np.empty((1,numPCs))*np.nan
        output_newDS[:] = ds_data[-1,4:].T
        inputData = np.append(inputData,input_newDS,axis=0)
        outputData = np.append(outputData, output_newDS,axis=0)

inputData = inputData[1:,:,:]
outputData = outputData[1:,:]


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
train_X = np.empty((Ntrain,num_steps,num_features))
train_X[:,:,:] = inputData[iitrain,:,:]
# train_y = np.empty((Ntrain,))
# train_y[:] = outputData_keep[iitrain]
train_y = np.empty((Ntrain,numPCs))
train_y[:] = outputData[iitrain,:]
# load testing
test_X = np.empty((Ntest,num_steps,num_features))
test_X[:,:,:] = inputData[iitest,:,:]
# test_y = np.empty((Ntest,))
# test_y[:] = outputData_keep[iitest]
test_y = np.empty((Ntest,numPCs))
test_y[:] = outputData[iitest,:]


############### DESIGN AND FIT NETWORK ###############

# design network
model = Sequential()
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.15))
model.add(Dense(numPCs))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=80, batch_size=24, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

# plot history
fig, ax = plt.subplots()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
ax.set_xlabel('epoch (test/train cycle)')
ax.set_ylabel('error')

# test network
yhat = model.predict(test_X)
inv_yhat = yhat * (beachstat_max - beachstat_min) + beachstat_min
inv_test_y = test_y * (beachstat_max - beachstat_min) + beachstat_min
fig, ax = plt.subplots()
ax.plot([np.min(inv_test_y)-20,np.max(inv_test_y)+20],[np.min(inv_test_y)-20,np.max(inv_test_y)+20],'-k')
ax.plot(inv_test_y,inv_yhat,'o')
ax.set_xlabel('observed - all testing times')
ax.set_ylabel('predicted - all testing times')
ax.grid()



############### TEST TIME SERIES PREDICTION ###############

# picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_20Feb2025/'
# with open(picklefile_dir+'stormy_times_fullspan.pickle','rb') as file:
#    _,storm_flag,storm_timestart_all,storm_timeend_all = pickle.load(file)

tplot = pd.to_datetime(storm_timestart_all, unit='s', origin='unix')
plotflag = True
for nn in np.arange(0,storm_timestart_all.size,5):
# for nn in np.arange(storm_timeend_all.size):

    tstart = storm_timestart_all[nn] + int(1 * 24 * 3600)
    iistart = np.where(np.isin(time_fullspan, tstart))[0].astype(int)

    # SHORT_TERM PREDICTION
    Npred = 500

    prev_pred = np.empty((Npred,)) * np.nan
    prev_obs = np.empty((Npred,)) * np.nan
    numnan_hydro = np.empty((Npred,)) * np.nan
    init_obs = np.empty((Nlook - 1,)) * np.nan
    for tt in np.arange(Npred):

        iisetnn_hydro = np.arange(iistart + tt, iistart + tt + Nlook - 1)  # shift entire window

        # grab actual data as long Npred < Nlook
        if tt <= (Nlook-1):
            # find and fill nans in PCs
            beachstat_setnn = beachstat_scaled[iisetnn_hydro]
            ytest = np.empty(shape=beachstat_setnn.shape) * np.nan
            ytest[:] = beachstat_setnn[:]
            if (np.sum(np.isnan(ytest)) > 1):
                print('warning - too many nans for post-storm ' + str(nn) + ', moving on')
                plotflag = False
                break
            else:
                if len(ytest) != 0:
                    plotflag = True
                    ds_beachstat = np.empty(shape=ytest.shape) * np.nan
                    yv = ytest
                    if (sum(np.isnan(yv)) > 0) & (Nlook > 2):
                        print(sum(np.isnan(yv)))
                        xq = np.arange(Nlook - 1 - tt)
                        xv = xq[~np.isnan(yv)]
                        yv = yv[~np.isnan(yv)]
                        beachstatjj_interptmp = np.interp(xq, xv, yv)
                        ds_beachstat[:] = beachstatjj_interptmp
                    else:
                        ds_beachstat[:] = yv
            iisetnn_beachstat = np.arange(tt, Nlook - 1)
            ds_beachstat = ds_beachstat[iisetnn_beachstat]
            # add previous predictions to fill out rest of PCs, if tt > 0
            if (tt > 0) & (tt <= Nlook - 1):
                ds_beachstat = np.append(ds_beachstat, prev_pred[0:tt])
            elif tt == 0:
                init_obs[:] = ds_beachstat[:]
        else:
            ytest = np.empty((Nlook - 1, 1)) * np.nan
            ds_beachstat = prev_pred[tt - Nlook:tt - 1]

        # find and fill nans in water levels
        ds_watlev = hydro_scaled[0, iisetnn_hydro]
        ds_Hs = hydro_scaled[1, iisetnn_hydro]
        ds_Tp = hydro_scaled[2, iisetnn_hydro]
        ds_wdir = hydro_scaled[3, iisetnn_hydro]
        if sum(~np.isnan(ds_watlev)) != 0:
            yv = ds_watlev
            if(sum(np.isnan(yv)) > 0) & (sum(np.isnan(yv)) < len(yv)) & (len(yv) > 0):
                xq = np.arange(Nlook - 1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_watlev[:] = hydro_interptmp
            # find and fill nans in waveheights
            yv = ds_Hs
            if (sum(np.isnan(yv)) > 0) & (sum(np.isnan(yv)) < len(yv)) & (len(yv) > 0):
                xq = np.arange(Nlook - 1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Hs[:] = hydro_interptmp
            # find and fill nans in wave periods
            yv = ds_Tp
            if (sum(np.isnan(yv)) > 0) & (sum(np.isnan(yv)) < len(yv)) & (len(yv) > 0):
                xq = np.arange(Nlook - 1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Tp[:] = hydro_interptmp
            # find and fill nans in wave directions
            yv = ds_wdir
            if (sum(np.isnan(yv)) > 0) & (sum(np.isnan(yv)) < len(yv)) & (len(yv) > 0):
                xq = np.arange(Nlook - 1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_wdir[:] = hydro_interptmp

        # check for nans
        numnan_hydro[tt] = np.sum(np.isnan(np.vstack((ds_watlev, ds_Hs, ds_Tp, ds_wdir))))

        if numnan_hydro[tt] == 0:
            # make input matrix for input model
            num_datasets = 1
            inputData = np.empty((num_datasets, num_steps, num_features))
            inputData[0, :, 0] = ds_watlev[:]
            inputData[0, :, 1] = ds_Hs[:]
            inputData[0, :, 2] = ds_Tp[:]
            inputData[0, :, 3] = ds_wdir[:]
            inputData[0, :, 4] = ds_beachstat[:]

            # make predicition
            test_X = np.empty(shape=inputData.shape) * np.nan
            test_X[:] = inputData[:]
            yhat = model.predict(test_X)

            # save last prediction as input for the next set
            prev_pred[tt] = yhat[:]
            prev_obs[tt] = beachstat_scaled[iisetnn_hydro[-1] + 1]

    if plotflag:
        # inverse scale the results
        inv_yhat = prev_pred * (beachstat_max - beachstat_min) + beachstat_min
        inv_test_y = prev_obs * (beachstat_max - beachstat_min) + beachstat_min
        inv_input_y = init_obs * (beachstat_max - beachstat_min) + beachstat_min
        slope, intercept, rval_modes, pval_modes, stderr_modes = sp.stats.linregress(inv_test_y, inv_yhat)
        rmse_modes = np.sqrt(np.nanmean((inv_test_y - inv_yhat) ** 2))
        nrmse_modes = np.sqrt(np.nanmean((inv_test_y - inv_yhat) ** 2)) / np.nanmean(inv_test_y)

        # # now plot prediction vs observed over time
        # fig, ax = plt.subplots(1, 2)
        # ax[0].scatter(inv_test_y, inv_yhat, 5, np.arange(Npred), alpha=0.95, cmap='plasma')
        # ax[0].grid()
        # # ax.set_ylim(minval, maxval)
        # # ax.set_xlim(minval, maxval)
        # ax[0].set_xlabel('observed')
        # ax[0].set_ylabel('predicted')

        # plot against observed data
        fig, ax = plt.subplots()
        xplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
        iiplot = np.arange(iistart, iistart + Nlook - 1)
        normxplot = np.arange(-(Nlook - 1), 0)
        # ax.plot(xplot[iiplot], inv_input_y, '.-k')
        ax.plot(normxplot, inv_input_y, '.-k')
        # min_taxis = xplot[iiplot[0]]
        min_taxis = normxplot[0]
        iiplot = np.arange(iistart + Nlook, iistart + Nlook + Npred)
        normxplot = np.arange(0, Npred)
        # ax.plot(xplot[iiplot], inv_test_y,'.-',color='grey')
        ax.plot(normxplot, inv_test_y, '.-', color='grey')
        # max_taxis = xplot[iiplot[-1]]
        max_taxis = normxplot[-1]
        ax.grid()
        ax.set_xlabel('$t_{predict}$ [hr]')
        ax.set_ylabel('$Xc_{MHW}$ [m], observed')
        ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        # ax2.scatter(xplot[iiplot],inv_yhat,10,iiplot,cmap='plasma')
        ax2.scatter(normxplot, inv_yhat, 10, np.arange(normxplot.size), cmap='plasma', vmin=0, vmax=Npred)
        ax2.set_ylabel('$Xc_{MHW}$ [m], predicted')
        ax.set_title(str(iistart))
        # date_form = DateFormatter("%m/%d/%y")
        # ax.xaxis.set_major_formatter(date_form)
        ax2.set_xlim(min_taxis, max_taxis)
        ax.set_xlim(min_taxis, max_taxis)
        # ax.tick_params(axis='x', labelrotation=45)
        # ax2.tick_params(axis='x', labelrotation=45)
        fig.set_size_inches(8.352, 2.792)
        plt.tight_layout()
