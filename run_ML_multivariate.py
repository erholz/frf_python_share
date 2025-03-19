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
with open(picklefile_dir + 'topobathyhydro_ML_final_18Mar2025_Nlook96_PCApostDVol_shifted.pickle', 'rb') as file:
    xplot_shift, time_fullspan, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV,reconstruct_profNorm_fullspan,reconstruct_prof_fullspan,dataobs_shift_fullspan,dataobs_fullspan,data_profIDs_dVolThreshMet,data_hydro,datahydro_fullspan = pickle.load(file)

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
hydro_fullspan_scaled = np.empty(shape=datahydro_fullspan.shape)*np.nan
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
    # (X - X_min) / (X_max - X_min)
    hydro_fullspan_scaled[nn,:] = (datahydro_fullspan[nn,:] - hydro_min[nn]) / (hydro_max[nn] - hydro_min[nn])
for nn in np.arange(PCs_fullspan.shape[1]):
    unscaled = PCs_fullspan[:,nn].reshape((PCs_fullspan.shape[0],1))
    PCs_min[nn] = np.nanmin(unscaled)
    PCs_max[nn] = np.nanmax(unscaled)
    PCs_avg[nn] = np.nanmean(unscaled)
    PCs_stdev[nn] = np.nanstd(unscaled)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(unscaled)
    PCs_scaled[:, nn] = np.squeeze(scaled)

numhydro = 4
numPCs = 8
num_features = numhydro + numPCs
num_steps = 24*4 - 1
num_datasets = data_hydro.shape[0]
inputData = np.empty((num_datasets,num_steps,num_features))
inputData[:] = np.nan
# outputData = np.empty((num_datasets,))
outputData = np.empty((num_datasets,numPCs))
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
    inputData[jj, :, 9] = PCs_setjj[0:-1, 5]
    inputData[jj, :, 10] = PCs_setjj[0:-1, 6]
    inputData[jj, :, 11] = PCs_setjj[0:-1, 7]
    outputData[jj,0] = PCs_setjj[-1,0]
    outputData[jj,1] = PCs_setjj[-1,1]
    outputData[jj,2] = PCs_setjj[-1,2]
    outputData[jj,3] = PCs_setjj[-1,3]
    outputData[jj, 4] = PCs_setjj[-1, 4]
    outputData[jj, 5] = PCs_setjj[-1, 5]
    outputData[jj, 6] = PCs_setjj[-1, 6]
    outputData[jj, 7] = PCs_setjj[-1, 7]
    if np.nansum(np.isnan(PCs_setjj)) > 0:
        nanPCs_data[jj] = 1

############### Step 1b - Change NLook ###############

Nlook = 48
num_steps = Nlook

inputData = np.empty((0,num_steps,num_features))*np.nan
outputData = np.empty((0,numPCs))*np.nan
for tt in np.arange(time_fullspan.size):
    tlook = np.arange(tt,tt + num_steps)



############### Step 2 - Split into test/train ###############

# remove few odd sets with nans in hydro data
iiremove = (nanhydro_data > 0) + (nanPCs_data > 0)
iiremove[0] = True
iikeep = ~iiremove
inputData_keep = inputData[iikeep,:,:]
outputData_keep = outputData[iikeep,:]

# separate test and train IDs
frac = 0.6          # num used for training
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
train_y = np.empty((Ntrain,numPCs))
train_y[:] = outputData_keep[iitrain,:]
# load testing
test_X = np.empty((Ntest,num_steps,num_features))
test_X[:,:,:] = inputData_keep[iitest,:,:]
# test_y = np.empty((Ntest,))
# test_y[:] = outputData_keep[iitest]
test_y = np.empty((Ntest,numPCs))
test_y[:] = outputData_keep[iitest,:]


############### Step 3 - Design/Fit network ###############

# design network
model = Sequential()
# model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.25))
model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(numPCs))

# custom loss function
def customLoss_wrapper(input_data):

    input_data = tf.cast(input_data, tf.float32)
    ytrue_prevobs = input_data[:,-1,4:]
    inv_ytrue_prevobs = ytrue_prevobs * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
    dx = 0.1
    vol_true_prev = keras.backend.sum(inv_ytrue_prevobs*dx,axis=1)

    def customLoss(y_true, y_pred):

        # loss = data_loss*weight_data + phys_loss*weight_phys
        weight_dataEOF = 1
        weight_datavol = 0.
        weight_dataelev = 0
        weight_dataelev_shore = 0.
        # weight_dataDVol = 0.1
        dataEOF_loss = keras.losses.MAE(y_true, y_pred)
        inv_ypred = y_pred * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
        inv_ytrue = y_true * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
        dataelev_loss = keras.losses.MAE(inv_ytrue, inv_ypred)
        # dataelevshore_loss = inv_ypred[:,0]-inv_ytrue[:,0]
        vol_true = keras.backend.sum(inv_ytrue*dx,axis=1)
        vol_pred = keras.backend.sum(inv_ypred*dx,axis=1)
        datavol_loss = keras.backend.abs(vol_true - vol_pred)
        # dataDVol_loss = keras.backend.abs(vol_true_prev - vol_pred)
        sum_loss = weight_dataEOF*dataEOF_loss + weight_dataelev*dataelev_loss #+ weight_datavol*datavol_loss + weight_dataelev_shore*dataelevshore_loss #+ weight_dataDVol*dataDVol_loss

        return sum_loss
    return customLoss

model.compile(loss=customLoss_wrapper(train_X), optimizer='adam')

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
inv_yhat = yhat * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
inv_test_y = test_y * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]

# fig, ax = plt.subplots()
# ax.plot(inv_test_y,inv_yhat,'.',alpha=0.1)
# plt.grid()
rval_modes = np.empty((numPCs,))*np.nan
pval_modes = np.empty((numPCs,))*np.nan
stderr_modes = np.empty((numPCs,))*np.nan
rmse_modes = np.empty((numPCs,))*np.nan
nrmse_modes = np.empty((numPCs,))*np.nan
for jj in np.arange(numPCs):
    slope, intercept, rval_modes[jj], pval_modes[jj], stderr_modes[jj] = sp.stats.linregress(inv_test_y[:,jj], inv_yhat[:,jj])
    rmse_modes[jj] = np.sqrt(np.nanmean((inv_test_y[:,jj] - inv_yhat[:,jj])**2))
    nrmse_modes[jj] = np.sqrt(np.nanmean((inv_test_y[:, jj] - inv_yhat[:, jj]) ** 2))/np.nanmean(inv_test_y[:,jj])


minval = -75
maxval = 75
fig, ax = plt.subplots(1,8)
for jj in range(int(numPCs)):
    ax[jj].plot([minval,maxval],[minval,maxval],'k')
    ax[jj].plot(inv_test_y[:,jj],inv_yhat[:,jj],'.',alpha=0.05)
    ax[jj].grid()
    ax[jj].set_ylim(minval,maxval)
    ax[jj].set_xlim(minval,maxval)
    # ax[jj].set_title('Mode '+str(jj+1)+' \n r^2 = '+str("%0.3f" % rval_modes[jj]))
    ax[jj].set_title('Mode ' + str(jj + 1) + ' \n NRMSE = ' + str("%0.1f" % nrmse_modes[jj]))
fig.set_size_inches(13.5,2.)
plt.tight_layout()


# compare observed at predicted output profiles...
mode1_obs = np.tile(EOFs[0,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,0]
mode2_obs = np.tile(EOFs[1,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,1]
mode3_obs = np.tile(EOFs[2,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,2]
mode4_obs = np.tile(EOFs[3,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,3]
mode5_obs = np.tile(EOFs[4,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,4]
mode6_obs = np.tile(EOFs[5,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,5]
mode7_obs = np.tile(EOFs[6,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,6]
mode8_obs = np.tile(EOFs[7,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,7]
profsobs_norm = mode1_obs + mode2_obs + mode3_obs + mode4_obs + mode5_obs + mode6_obs + mode7_obs + mode8_obs
profsobs_T = profsobs_norm.T * dataStd.T + dataMean.T
profobs = profsobs_T.T
fig, ax = plt.subplots()
ax.plot(xplot_shift,profobs)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Observed* (PCs)')
mode1_pred = np.tile(EOFs[0,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,0]
mode2_pred = np.tile(EOFs[1,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,1]
mode3_pred = np.tile(EOFs[2,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,2]
mode4_pred = np.tile(EOFs[3,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,3]
mode5_pred = np.tile(EOFs[4,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,4]
mode6_pred = np.tile(EOFs[5,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,5]
mode7_pred = np.tile(EOFs[6,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,6]
mode8_pred = np.tile(EOFs[7,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,7]
profspred_norm = mode1_pred + mode2_pred + mode3_pred + mode4_pred + mode5_pred + mode6_pred + mode7_pred + mode8_pred
profspred_T = profspred_norm.T * dataStd.T + dataMean.T
profpred = profspred_T.T
fig, ax = plt.subplots()
ax.plot(xplot_shift,profpred)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Predicted')

fig, ax = plt.subplots()
ax.plot(xplot_shift,profpred-profobs)
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

    prev_pred = np.empty((Npred,numPCs))*np.nan
    prev_obs = np.empty((Npred,numPCs))*np.nan
    numnan_hydro = np.empty((Npred,))*np.nan
    for tt in np.arange(Npred):

        # grab actual data as long Npred < Nlook
        iisetnn_PCs = np.arange(iistart + tt, iistart + Nlook-1)
        iisetnn_hydro = np.arange(iistart + tt, iistart + tt + Nlook-1)

        # find and fill nans in PCs
        PCs_setnn = PCs_scaled[iisetnn_PCs, 0:numPCs]
        if np.sum(np.isnan(PCs_setnn)) > 5:
            print('warning - too many nans in PC for post-storm '+str(nn)+', moving on')
            plotflag = False
            break
        else:
            plotflag = True
            ds_PCs = np.empty(shape=PCs_setnn.shape)*np.nan
            for jj in np.arange(numPCs):
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
            inputData[0, :, 9] = ds_PCs[:, 5]
            inputData[0, :, 10] = ds_PCs[:, 6]
            inputData[0, :, 11] = ds_PCs[:, 7]
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
            prev_obs[tt,:] = PCs_scaled[iisetnn_hydro[-1]+1, 0:numPCs]

    if plotflag:
        # inverse scale the results
        inv_yhat = prev_pred * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
        inv_test_y = prev_obs * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
        rval_modes = np.empty((numPCs,))*np.nan
        pval_modes = np.empty((numPCs,))*np.nan
        stderr_modes = np.empty((numPCs,))*np.nan
        rmse_modes = np.empty((numPCs,))*np.nan
        nrmse_modes = np.empty((numPCs,))*np.nan
        for jj in np.arange(numPCs):
            slope, intercept, rval_modes[jj], pval_modes[jj], stderr_modes[jj] = sp.stats.linregress(inv_test_y[:,jj], inv_yhat[:,jj])
            rmse_modes[jj] = np.sqrt(np.nanmean((inv_test_y[:,jj] - inv_yhat[:,jj])**2))
            nrmse_modes[jj] = np.sqrt(np.nanmean((inv_test_y[:, jj] - inv_yhat[:, jj]) ** 2))/np.nanmean(inv_test_y[:,jj])

        # now plot prediction vs observed over time
        # fig, ax = plt.subplots()

        minval = -75
        maxval = 75
        scatsz = 5
        fig, ax = plt.subplots(1, 8)
        for jj in range(int(numPCs)):
            ax[jj].plot([minval, maxval], [minval, maxval], 'k')
            ax[jj].scatter(inv_test_y[:,jj],inv_yhat[:,jj],scatsz,np.arange(Npred),alpha=0.95,cmap='plasma')
            ax[jj].grid()
            ax[jj].set_ylim(minval, maxval)
            ax[jj].set_xlim(minval, maxval)
            # ax[jj].set_title('Mode '+str(jj+1)+' \n r^2 = '+str("%0.3f" % rval_modes[jj]))
            ax[jj].set_title('Mode ' + str(jj + 1) + ' \n NRMSE = ' + str("%0.1f" % nrmse_modes[jj]))
        fig.set_size_inches(13.5, 2.)
        plt.tight_layout()

        # compare observed at predicted output profiles...
        mode1_obs = np.tile(EOFs[0, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 0]
        mode2_obs = np.tile(EOFs[1, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 1]
        mode3_obs = np.tile(EOFs[2, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 2]
        mode4_obs = np.tile(EOFs[3, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 3]
        mode5_obs = np.tile(EOFs[4, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 4]
        mode6_obs = np.tile(EOFs[5, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 5]
        mode7_obs = np.tile(EOFs[6, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 6]
        mode8_obs = np.tile(EOFs[7, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 7]
        profsobs_norm = mode1_obs + mode2_obs + mode3_obs + mode4_obs + mode5_obs + mode6_obs + mode7_obs + mode8_obs
        profsobs_T = profsobs_norm.T * dataStd.T + dataMean.T
        profobs = profsobs_T.T
        mode1_pred = np.tile(EOFs[0, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 0]
        mode2_pred = np.tile(EOFs[1, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 1]
        mode3_pred = np.tile(EOFs[2, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 2]
        mode4_pred = np.tile(EOFs[3, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 3]
        mode5_pred = np.tile(EOFs[4, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 4]
        mode6_pred = np.tile(EOFs[5, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 5]
        mode7_pred = np.tile(EOFs[6, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 6]
        mode8_pred = np.tile(EOFs[7, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 7]
        profspred_norm = mode1_pred + mode2_pred + mode3_pred + mode4_pred + mode5_pred + mode6_pred + mode7_pred + mode8_pred
        profspred_T = profspred_norm.T * dataStd.T + dataMean.T
        profpred = profspred_T.T

        # plot against observed data
        xplot = xplot_shift[:]
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


