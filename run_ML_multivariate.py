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
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import tensorboard
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Flatten
from keras.utils import plot_model
import datetime as dt
import pydot
import visualkeras
import random
import scipy as sp
import pandas as pd  # to do datetime conversions




############### Step 1 - Load and prep data ###############

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PCApostDVol.pickle', 'rb') as file:
    xplot,time_fullspan,dataNorm_fullspan,dataMean,dataStd,PCs_fullspan,EOFs,APEV,data_profIDs_dVolThreshMet,reconstruct_profNorm_fullspan,reconstruct_prof_fullspan,data_hydro = pickle.load(file)


# Re-scale data
Nlook = 24*4
num_datasets = data_hydro.shape[0]
hydro_datasetsForML_scaled = np.empty(shape=data_hydro.shape)
PCs_scaled = np.empty(shape=PCs_fullspan.shape)
hydro_min = np.empty((4,))
hydro_max = np.empty((4,))
hydro_avg = np.empty((4,))
hydro_stdev = np.empty((4,))
PCs_min = np.empty((PCs_fullspan.shape[1],))
PCs_max = np.empty((PCs_fullspan.shape[1],))
PCs_avg = np.empty((PCs_fullspan.shape[1],))
PCs_stdev = np.empty((PCs_fullspan.shape[1],))
for nn in np.arange(4):
    unscaled = data_hydro[:,:,nn].reshape((Nlook*num_datasets,1))
    hydro_min[nn] = np.nanmin(unscaled)
    hydro_max[nn] = np.nanmax(unscaled)
    hydro_avg[nn] = np.nanmean(unscaled)
    hydro_stdev[nn] = np.nanstd(unscaled)
    unscaled_reshape = unscaled.reshape((num_datasets,Nlook))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(unscaled)
    scaled_reshape = scaled.reshape((num_datasets,Nlook))
    hydro_datasetsForML_scaled[:,:,nn] = scaled_reshape
for nn in np.arange(PCs_fullspan.shape[1]):
    unscaled = PCs_fullspan[:,nn].reshape((PCs_fullspan.shape[0],1))
    PCs_min[nn] = np.nanmin(unscaled)
    PCs_max[nn] = np.nanmax(unscaled)
    PCs_avg[nn] = np.nanmean(unscaled)
    PCs_stdev[nn] = np.nanstd(unscaled)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(unscaled)
    PCs_scaled[:, nn] = np.squeeze(scaled)

num_features = 9
num_steps = 24*4 - 1
num_datasets = data_hydro.shape[0]
inputData = np.empty((num_datasets,num_steps,num_features))
inputData[:] = np.nan
# outputData = np.empty((num_datasets,))
outputData = np.empty((num_datasets,5))
outputData[:] = np.nan
numinset = np.empty((num_datasets,))
numinset[:] = np.nan
avgdt = np.empty((num_datasets,))
avgdt[:] = np.nan
nanhydro_data = np.zeros((num_datasets,))
nanPCs_data = np.zeros((num_datasets,))
for jj in np.arange(num_datasets):
    dsjj = jj
    # get input hydro
    ds_watlev = data_hydro[dsjj,:,0]
    ds_Hs = data_hydro[dsjj,:,1]
    ds_Tp = data_hydro[dsjj,:,2]
    ds_wdir = data_hydro[dsjj,:,3]
    # if sum(nanflag_hydro[dsjj,:]) > 0:
    #     nanhydro_data[jj] = 1
    # get input PC amplitudes
    tmpii = data_profIDs_dVolThreshMet[dsjj,:]
    PCs_setjj = PCs_scaled[tmpii, :]
    numinset[jj] = len(tmpii)
    # load into training matrices
    inputData[jj, :, 0] = ds_watlev[0:-1]
    inputData[jj, :, 1] = ds_Hs[0:-1]
    inputData[jj, :, 2] = ds_Tp[0:-1]
    inputData[jj, :, 3] = ds_wdir[0:-1]
    inputData[jj, :, 4] = PCs_setjj[0:-1,0]
    inputData[jj, :, 5] = PCs_setjj[0:-1,1]
    inputData[jj, :, 6] = PCs_setjj[0:-1,2]
    inputData[jj, :, 7] = PCs_setjj[0:-1,3]
    inputData[jj, :, 8] = PCs_setjj[0:-1,4]
    outputData[jj,0] = PCs_setjj[-1,0]
    outputData[jj,1] = PCs_setjj[-1,1]
    outputData[jj,2] = PCs_setjj[-1,2]
    outputData[jj,3] = PCs_setjj[-1,3]
    outputData[jj, 4] = PCs_setjj[-1, 4]
    if np.nansum(np.isnan(PCs_setjj)) > 0:
        nanPCs_data[jj] = 1


############### Step 2 - Split into test/train ###############

# remove few odd sets with nans in hydro data
iiremove = (nanhydro_data > 0) + (nanPCs_data > 0)
iiremove[0] = True
iikeep = ~iiremove
inputData_keep = inputData[iikeep,:,:]
outputData_keep = outputData[iikeep,:]

# separate test and train IDs
frac = 1/2
num_datasets = sum(iikeep)
Ntrain = int(np.floor(num_datasets*frac))
Ntest = num_datasets - Ntrain
tmpii = random.sample(range(num_datasets), Ntrain)
iitrain = np.isin(np.arange(num_datasets),tmpii)
iitest = ~iitrain
# load training
train_X = np.empty((Ntrain,num_steps,num_features))
train_X[:,:,:] = inputData_keep[iitrain,:,:]
# train_y = np.empty((Ntrain,))
# train_y[:] = outputData_keep[iitrain]
train_y = np.empty((Ntrain,5))
train_y[:] = outputData_keep[iitrain,:]
# load testing
test_X = np.empty((Ntest,num_steps,num_features))
test_X[:,:,:] = inputData_keep[iitest,:,:]
# test_y = np.empty((Ntest,))
# test_y[:] = outputData_keep[iitest]
test_y = np.empty((Ntest,5))
test_y[:] = outputData_keep[iitest,:]


############### Step 3 - Design/Fit network ###############

# design network
model = Sequential()
model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.25))
# model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(5))

# custom loss function
def customLoss(y_true, y_pred):

    # loss = data_loss*weight_data + phys_loss*weight_phys
    weight_data = 0.5
    weight_phys = 0.5
    data_loss = keras.losses.MAE(y_true, y_pred)
    inv_ypred = prev_pred * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]
    inv_ytrue = prev_obs * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]
    dx = 0.1
    vol_true = np.nansum(inv_ytrue*dx)
    vol_pred = np.nansum(inv_ypred*dx)
    phys_loss = np.abs(vol_true - vol_pred)
    loss = weight_phys*phys_loss + weight_data*data_loss

    return loss

model.compile(loss=customLoss, optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=60, batch_size=40, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
fig, ax = plt.subplots()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
ax.set_xlabel('epoch (test/train cycle)')
ax.set_ylabel('error')

############### Step 4 - Test prediction ###############

# X_scaled = (X - X_min) / (X_max - X_min)
# X = X_scaled * (X_max - X_min) + X_min
yhat = model.predict(test_X)
inv_yhat = yhat * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]
inv_test_y = test_y * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]

# fig, ax = plt.subplots()
# ax.plot(inv_test_y,inv_yhat,'.',alpha=0.1)
# plt.grid()
rval_modes = np.empty((5,))*np.nan
pval_modes = np.empty((5,))*np.nan
stderr_modes = np.empty((5,))*np.nan
for jj in np.arange(5):
    slope, intercept, rval_modes[jj], pval_modes[jj], stderr_modes[jj] = sp.stats.linregress(inv_test_y[:,jj], inv_yhat[:,jj])


minval = -75
maxval = 75
fig, ax = plt.subplots(1,5)
ax[0].plot([minval,maxval],[minval,maxval],'k')
ax[0].plot(inv_test_y[:,0],inv_yhat[:,0],'.',alpha=0.1)
ax[0].grid()
ax[0].set_ylim(minval,maxval)
ax[0].set_xlim(minval,maxval)
ax[0].set_title('Mode 1 \n r^2 = '+str("%0.3f" % rval_modes[0]))
ax[1].plot([minval,maxval],[minval,maxval],'k')
ax[1].plot(inv_test_y[:,1],inv_yhat[:,1],'.',alpha=0.1)
ax[1].grid()
ax[1].set_ylim(minval,maxval)
ax[1].set_xlim(minval,maxval)
ax[1].set_title('Mode 2 \n r^2 = '+str("%0.3f" % rval_modes[1]))
ax[1].set_yticklabels([])
ax[2].plot([minval,maxval],[minval,maxval],'k')
ax[2].plot(inv_test_y[:,2],inv_yhat[:,2],'.',alpha=0.1)
ax[2].grid()
ax[2].set_ylim(minval,maxval)
ax[2].set_xlim(minval,maxval)
ax[2].set_title('Mode 3 \n r^2 = '+str("%0.3f" % rval_modes[2]))
ax[2].set_yticklabels([])
ax[3].plot([minval,maxval],[minval,maxval],'k')
ax[3].plot(inv_test_y[:,3],inv_yhat[:,3],'.',alpha=0.1)
ax[3].grid()
ax[3].set_ylim(minval,maxval)
ax[3].set_xlim(minval,maxval)
ax[3].set_title('Mode 4 \n r^2 = '+str("%0.3f" % rval_modes[3]))
ax[3].set_yticklabels([])
ax[4].plot([minval,maxval],[minval,maxval],'k')
ax[4].plot(inv_test_y[:,4],inv_yhat[:,4],'.',alpha=0.1)
ax[4].grid()
ax[4].set_ylim(minval,maxval)
ax[4].set_xlim(minval,maxval)
ax[4].set_title('Mode 5 \n r^2 = '+str("%0.3f" % rval_modes[4]))
ax[4].set_yticklabels([])


# compare observed at predicted output profiles...
mode1_obs = np.tile(EOFs[0,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,0]
mode2_obs = np.tile(EOFs[1,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,1]
mode3_obs = np.tile(EOFs[2,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,2]
mode4_obs = np.tile(EOFs[3,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,3]
mode5_obs = np.tile(EOFs[4,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,4]
profsobs_norm = mode1_obs + mode2_obs + mode3_obs + mode4_obs + mode5_obs
profsobs_T = profsobs_norm.T * dataStd.T + dataMean.T
profobs = profsobs_T.T
fig, ax = plt.subplots()
ax.plot(xplot,profobs)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Observed* (PCs)')
mode1_pred = np.tile(EOFs[0,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,0]
mode2_pred = np.tile(EOFs[1,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,1]
mode3_pred = np.tile(EOFs[2,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,2]
mode4_pred = np.tile(EOFs[3,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,3]
mode5_pred = np.tile(EOFs[4,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,4]
profspred_norm = mode1_pred + mode2_pred + mode3_pred + mode4_pred + mode5_pred
profspred_T = profspred_norm.T * dataStd.T + dataMean.T
profpred = profspred_T.T
fig, ax = plt.subplots()
ax.plot(xplot,profpred)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Predicted')

fig, ax = plt.subplots()
ax.plot(xplot,profpred-profobs)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Obs - Pred')


############### Step 5 - Evaluate particular instance ###############

picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_20Feb2025/'
with open(picklefile_dir+'stormy_times_fullspan.pickle','rb') as file:
   _,storm_flag,storm_timestart_all,storm_timeend_all = pickle.load(file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'data_fullspan.pickle','rb') as file:
    data_fullspan = pickle.load(file)
    watlev_fullspan = np.squeeze(data_fullspan["fullspan_tidegauge"])
picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_20Feb2025/'
with open(picklefile_dir+'waves_8m&17m_2015_2024.pickle','rb') as file:
    [_,_,data_wave8m_filled] = pickle.load(file)

# find time post-storm where data is adequate...
numnan_PCs = np.empty(shape=storm_timeend_all.shape)*np.nan
numnan_watlev = np.empty(shape=storm_timeend_all.shape)*np.nan
numnan_Hs = np.empty(shape=storm_timeend_all.shape)*np.nan
numnan_Tp = np.empty(shape=storm_timeend_all.shape)*np.nan
numnan_dir = np.empty(shape=storm_timeend_all.shape)*np.nan
for nn in np.arange(storm_timeend_all.size):
    iistart = np.where(np.isin(time_fullspan,storm_timeend_all[nn]))[0].astype(int)
    iisetnn = np.arange(iistart,iistart+Nlook)
    PCs_setnn = PCs_fullspan[iisetnn,0]
    numnan_PCs[nn] = np.sum(np.isnan(PCs_setnn))
    numnan_watlev[nn] = np.sum(np.isnan(watlev_fullspan[iisetnn,]))
    numnan_Hs[nn] = np.sum(np.isnan(data_wave8m_filled[iisetnn, 0]))
    numnan_Tp[nn] = np.sum(np.isnan(data_wave8m_filled[iisetnn, 1]))
    numnan_dir[nn] = np.sum(np.isnan(data_wave8m_filled[iisetnn, 2]))
fig, ax = plt.subplots()
tplot = pd.to_datetime(storm_timeend_all, unit='s', origin='unix')
ax.plot(tplot,numnan_PCs,'*')
ax.plot(tplot,numnan_watlev,'s')
ax.plot(tplot,numnan_Hs,'^')
ax.plot(tplot,numnan_Tp,'o')
ax.plot(tplot,numnan_dir,'.')

# scale the hydro data according to previously done scaling routines
# X_scaled = (X - X_min) / (X_max - X_min)
# X = X_scaled * (X_max - X_min) + X_min
watlev_scaled = (watlev_fullspan - hydro_min[0]) / (hydro_max[0] - hydro_min[0])
waveHs_scaled = (data_wave8m_filled[:,0] - hydro_min[1]) / (hydro_max[1] - hydro_min[1])
waveTp_scaled = (data_wave8m_filled[:,1] - hydro_min[2]) / (hydro_max[2] - hydro_min[2])
wavedir_scaled = (data_wave8m_filled[:,2] - hydro_min[3]) / (hydro_max[3] - hydro_min[3])

# now perform the prediction on any one of those post-storm times that meets data needs
storm_times_withdata = storm_timeend_all[numnan_PCs+numnan_watlev+numnan_Hs+numnan_Tp+numnan_dir < 5]
tplot = pd.to_datetime(storm_times_withdata, unit='s', origin='unix')
plotflag = True
# for nn in np.arange(5):
for nn in np.arange(storm_times_withdata.size):

    tstart = storm_timeend_all[nn] + 2*24*3600
    iistart = np.where(np.isin(time_fullspan, tstart))[0].astype(int)

    # SHORT_TERM PREDICTION
    Npred = Nlook-1
    prev_pred = np.empty((Npred,5))*np.nan
    prev_obs = np.empty((Npred,5))*np.nan
    numnan_hydro = np.empty((Npred,))*np.nan
    for tt in np.arange(Npred):

        # grab actual data as long Npred < Nlook
        iisetnn_PCs = np.arange(iistart + tt, iistart + Nlook-1)
        iisetnn_hydro = np.arange(iistart + tt, iistart + tt + Nlook-1)

        # find and fill nans in PCs
        PCs_setnn = PCs_scaled[iisetnn_PCs, 0:5]
        if np.sum(np.isnan(PCs_setnn)) > 5:
            print('warning - too many nans in PC for post-storm '+str(nn)+', moving on')
            plotflag = False
            break
        else:
            plotflag = True
            ds_PCs = np.empty(shape=PCs_setnn.shape)*np.nan
            for jj in np.arange(5):
                yv = PCs_setnn[:, jj]
                if sum(np.isnan(yv)) > 0:
                    xq = np.arange(Nlook-1-tt)
                    xv = xq[~np.isnan(yv)]
                    yv = yv[~np.isnan(yv)]
                    PCjj_interptmp = np.interp(xq,xv,yv)
                    ds_PCs[:,jj] = PCjj_interptmp
                else:
                    ds_PCs[:, jj] = yv
            # add previous predictions to fill out rest of PCs, if tt > 0
            if tt > 0:
                ds_PCs = np.vstack((ds_PCs,prev_pred[0:tt,:]))

            # find and fill nans in water levels
            ds_watlev = watlev_scaled[iisetnn_hydro]
            yv = ds_watlev
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_watlev[:] = hydro_interptmp
            # find and fill nans in waveheights
            ds_Hs = waveHs_scaled[iisetnn_hydro]
            yv = ds_Hs
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Hs[:] = hydro_interptmp
            # find and fill nans in wave periods
            ds_Tp = waveTp_scaled[iisetnn_hydro]
            yv = ds_Tp
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Tp[:] = hydro_interptmp
            # find and fill nans in wave directions
            ds_wdir = wavedir_scaled[iisetnn_hydro]
            yv = ds_wdir
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_wdir[:] = hydro_interptmp

            # check for nans
            numnan_hydro[tt] = np.sum(np.isnan(np.vstack((ds_watlev,ds_Hs,ds_Tp,ds_wdir))))

            # make input matrix for input model
            num_datasets = 1
            inputData = np.empty((num_datasets, num_steps, num_features))
            inputData[0, :, 0] = ds_watlev[:]
            inputData[0, :, 1] = ds_Hs[:]
            inputData[0, :, 2] = ds_Tp[:]
            inputData[0, :, 3] = ds_wdir[:]
            inputData[0, :, 4] = ds_PCs[:, 0]
            inputData[0, :, 5] = ds_PCs[:, 1]
            inputData[0, :, 6] = ds_PCs[:, 2]
            inputData[0, :, 7] = ds_PCs[:, 3]
            inputData[0, :, 8] = ds_PCs[:, 4]
            # outputData = np.empty((num_datasets, 5))
            # outputData[0, 0] = ds_PCs[-1, 0]
            # outputData[0, 1] = ds_PCs[-1, 1]
            # outputData[0, 2] = ds_PCs[-1, 2]
            # outputData[0, 3] = ds_PCs[-1, 3]
            # outputData[0, 4] = ds_PCs[-1, 4]

            # make predicition
            test_X = np.empty(shape=inputData.shape)*np.nan
            test_X[:] = inputData[:]
            yhat = model.predict(test_X)

            # save last prediction as input for the next set
            prev_pred[tt,:] = yhat[:]
            prev_obs[tt,:] = PCs_scaled[iisetnn_hydro[-1]+1, 0:5]

    if plotflag:
        # inverse scale the results
        inv_yhat = prev_pred * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]
        inv_test_y = prev_obs * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]
        rval_modes = np.empty((5,))*np.nan
        pval_modes = np.empty((5,))*np.nan
        stderr_modes = np.empty((5,))*np.nan
        for jj in np.arange(5):
            slope, intercept, rval_modes[jj], pval_modes[jj], stderr_modes[jj] = sp.stats.linregress(inv_test_y[:,jj], inv_yhat[:,jj])


        # now plot prediction vs observed over time
        # fig, ax = plt.subplots()
        # ax.plot(numnan_hydro,'o')
        fig, ax = plt.subplots(1,5)
        fig.set_size_inches(8.7, 2.1)
        minval = -75
        maxval = 75
        scatsz = 5
        ax[0].plot([minval,maxval],[minval,maxval],'k')
        ax[0].scatter(inv_test_y[:,0],inv_yhat[:,0],scatsz,np.arange(Npred),alpha=0.95,cmap='plasma')
        ax[0].grid()
        ax[0].set_ylim(minval,maxval)
        ax[0].set_xlim(minval,maxval)
        ax[0].set_title('Mode 1 \n r^2 = '+str("%0.3f" % rval_modes[0]))
        ax[1].plot([minval,maxval],[minval,maxval],'k')
        ax[1].scatter(inv_test_y[:,1],inv_yhat[:,1],scatsz,np.arange(Npred),alpha=0.95,cmap='plasma')
        ax[1].grid()
        ax[1].set_ylim(minval,maxval)
        ax[1].set_xlim(minval,maxval)
        ax[1].set_title('Mode 2 \n r^2 = '+str("%0.3f" % rval_modes[1]))
        ax[1].set_yticklabels([])
        ax[2].plot([minval,maxval],[minval,maxval],'k')
        ax[2].scatter(inv_test_y[:,2],inv_yhat[:,2],scatsz,np.arange(Npred),alpha=0.95,cmap='plasma')
        ax[2].grid()
        ax[2].set_ylim(minval,maxval)
        ax[2].set_xlim(minval,maxval)
        ax[2].set_title('Mode 3 \n r^2 = '+str("%0.3f" % rval_modes[2]))
        ax[2].set_yticklabels([])
        ax[3].plot([minval,maxval],[minval,maxval],'k')
        ax[3].scatter(inv_test_y[:,3],inv_yhat[:,3],scatsz,np.arange(Npred),alpha=0.5,cmap='plasma')
        ax[3].grid()
        ax[3].set_ylim(minval,maxval)
        ax[3].set_xlim(minval,maxval)
        ax[3].set_title('Mode 4 \n r^2 = '+str("%0.3f" % rval_modes[3]))
        ax[3].set_yticklabels([])
        ax[4].plot([minval,maxval],[minval,maxval],'k')
        ph = ax[4].scatter(inv_test_y[:,4],inv_yhat[:,4],scatsz,np.arange(Npred),alpha=0.95,cmap='plasma')
        ax[4].grid()
        ax[4].set_ylim(minval,maxval)
        ax[4].set_xlim(minval,maxval)
        ax[4].set_title('Mode 5 \n r^2 = '+str("%0.3f" % rval_modes[4]))
        ax[4].set_yticklabels([])
        cbar = fig.colorbar(ph, ax=ax[4])
        cbar.set_label('prediction time [hrs]')
        plt.tight_layout()

        # plot predicted versus observed profiles
        mode1_obs = np.tile(EOFs[0,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,0]
        mode2_obs = np.tile(EOFs[1,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,1]
        mode3_obs = np.tile(EOFs[2,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,2]
        mode4_obs = np.tile(EOFs[3,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,3]
        mode5_obs = np.tile(EOFs[4,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,4]
        profsobs_norm = mode1_obs + mode2_obs + mode3_obs + mode4_obs + mode5_obs
        profsobs_T = profsobs_norm.T * dataStd.T + dataMean.T
        profobs = profsobs_T.T
        mode1_pred = np.tile(EOFs[0,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,0]
        mode2_pred = np.tile(EOFs[1,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,1]
        mode3_pred = np.tile(EOFs[2,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,2]
        mode4_pred = np.tile(EOFs[3,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,3]
        mode5_pred = np.tile(EOFs[4,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,4]
        profspred_norm = mode1_pred + mode2_pred + mode3_pred + mode4_pred + mode5_pred
        profspred_T = profspred_norm.T * dataStd.T + dataMean.T
        profpred = profspred_T.T

        # plot against observed data
        dataprof_fullspan = (dataNorm_fullspan.T * dataStd) + dataMean
        iiplot = np.arange(iistart + Nlook, iistart + Nlook + Npred)
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(10.7,3.3)
        cmapbw = plt.cm.Greys(np.linspace(0, 1, Nlook))
        ax[0].set_prop_cycle('color', cmapbw)
        ax[0].plot(xplot, dataprof_fullspan[np.arange(iistart, iistart + Nlook), :].T,linewidth=2)
        cmap = plt.cm.plasma(np.linspace(0, 1, Npred))
        ax[0].set_prop_cycle('color', cmap)
        ax[0].plot(xplot, dataprof_fullspan[iiplot, :].T)
        ax[0].set_ylabel('z, obs-data [m]')
        ax[1].plot(xplot, dataprof_fullspan[iistart + Nlook, :].T, 'k')
        ax[1].set_prop_cycle('color', cmap)
        ax[1].plot(xplot,profobs)
        ax[1].set_xlabel('xFRF [m]')
        ax[1].set_ylabel('z, obs-PCA [m]')
        ax[1].set_title(tplot[nn])
        ax[2].plot(xplot, dataprof_fullspan[iistart + Nlook, :].T, 'k')
        ax[2].set_prop_cycle('color', cmap)
        ax[2].plot(xplot,profpred)
        ax[2].set_ylabel('z, predicted [m]')
        plt.tight_layout()


