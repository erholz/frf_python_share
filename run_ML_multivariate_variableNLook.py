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





############### Step 1 - Load and prep data ###############

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PCApostDVol.pickle', 'rb') as file:
#     xplot,time_fullspan,dataNorm_fullspan,dataMean,dataStd,PCs_fullspan,EOFs,APEV,data_profIDs_dVolThreshMet,reconstruct_profNorm_fullspan,reconstruct_prof_fullspan,data_hydro = pickle.load(file)
# with open(picklefile_dir + 'topobathyhydro_ML_final_18Mar2025_Nlook96_PCApostDVol_shifted.pickle', 'rb') as file:
#     xplot_shift, time_fullspan, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV,reconstruct_profNorm_fullspan,reconstruct_prof_fullspan,dataobs_shift_fullspan,dataobs_fullspan,data_profIDs_dVolThreshMet,data_hydro,datahydro_fullspan = pickle.load(file)

with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_Nlook60_PCApostDVol_shifted.pickle', 'rb') as file:
    xplot_shift, time_fullspan, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV, reconstruct_profNorm_fullspan, reconstruct_prof_fullspan, dataobs_shift_fullspan, dataobs_fullspan, data_profIDs_dVolThreshMet, data_hydro, datahydro_fullspan = pickle.load(file)

# Re-scale data
Nlook = int(24*2.5)
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

############### Step 1.2 - Remove "large" dPCs and rescale ###############

PCs_scaled = np.empty(shape=PCs_fullspan.shape)
PCs_min = np.empty((PCs_fullspan.shape[1],))
PCs_max = np.empty((PCs_fullspan.shape[1],))
PCs_avg = np.empty((PCs_fullspan.shape[1],))
PCs_stdev = np.empty((PCs_fullspan.shape[1],))

PC1 = PCs_scaled[:,0]
dPC1 = PC1[1:] - PC1[0:-1]
flagii = np.where((dPC1 > 0.02) | (dPC1 < -0.02))[0]
PC1[flagii+1] = np.nan
PCs_scaled[flagii+1,0] = np.nan
PCs_fullspan[flagii+1,0] = np.nan
# sum(np.sum(~np.isnan(PC1),axis=1) == 48)

PC2 = PCs_scaled[:,1]
dPC2 = PC2[1:] - PC2[0:-1]
flagii = np.where((dPC2 > 0.05) | (dPC2 < -0.05))[0]
PC2[flagii+1] = np.nan
PCs_scaled[flagii+1,1] = np.nan
PCs_fullspan[flagii+1,1] = np.nan
# sum(np.sum(~np.isnan(PC2),axis=1) == 48)

PC3 = PCs_scaled[:,2]
dPC3 = PC3[1:] - PC3[0:-1]
flagii = np.where((dPC3 > 0.05) | (dPC3 < -0.05))[0]
PC3[flagii+1] = np.nan
PCs_scaled[flagii+1,2] = np.nan
PCs_fullspan[flagii+1,2] = np.nan
# sum(np.sum(~np.isnan(PC3),axis=1) == 48)

PC4 = PCs_scaled[:,3]
dPC4 = PC4[1:] - PC4[0:-1]
flagii = np.where((dPC4 > 0.15) | (dPC4 < -0.13))[0]
PC4[flagii+1] = np.nan
PCs_scaled[flagii+1,3] = np.nan
PCs_fullspan[flagii+1,3] = np.nan

PC5 = PCs_scaled[:,4]
dPC5 = PC5[1:] - PC5[0:-1]
flagii = np.where((dPC5 > 0.2) | (dPC5 < -0.27))[0]
PC5[flagii+1] = np.nan
PCs_scaled[flagii+1,4] = np.nan
PCs_fullspan[flagii+1,4] = np.nan

PC6 = PCs_scaled[:,5]
dPC6 = PC6[1:] - PC6[0:-1]
flagii = np.where((dPC6 > 0.13) | (dPC6 < -0.11))[0]
PC6[flagii+1] = np.nan
PCs_scaled[flagii+1,5] = np.nan
PCs_fullspan[flagii+1,5] = np.nan

PC7 = PCs_scaled[:,6]
dPC7 = PC7[1:] - PC7[0:-1]
flagii = np.where((dPC7 > 0.13) | (dPC7 < -0.13))[0]
PC7[flagii+1] = np.nan
PCs_scaled[flagii+1,6] = np.nan
PCs_fullspan[flagii+1,6] = np.nan

PC8 = PCs_scaled[:,7]
dPC8 = PC8[1:] - PC8[0:-1]
flagii = np.where((dPC8 > 0.13) | (dPC8 < -0.11))[0]
PC8[flagii+1] = np.nan
PCs_scaled[flagii+1,7] = np.nan
PCs_fullspan[flagii+1,7] = np.nan

for nn in np.arange(PCs_fullspan.shape[1]):
    unscaled = PCs_fullspan[:,nn].reshape((PCs_fullspan.shape[0],1))
    PCs_min[nn] = np.nanmin(unscaled)
    PCs_max[nn] = np.nanmax(unscaled)
    PCs_avg[nn] = np.nanmean(unscaled)
    PCs_stdev[nn] = np.nanstd(unscaled)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(unscaled)
    PCs_scaled[:, nn] = np.squeeze(scaled)

PCs_iikeep = np.sum(np.isnan(PCs_scaled),axis=1) == 0
# with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_slowlyVaryingPCs.pickle', 'wb') as file:
#     pickle.dump([time_fullspan, PCs_iikeep],file)



############### Step 2 - Change NLook ###############

Nlook = int(2.*24)
num_steps = Nlook-1
numhydro = 4
numPCs = 8
num_features = numhydro + numPCs

inputData = np.empty((1,num_steps,num_features))*np.nan
outputData = np.empty((1,numPCs))*np.nan
for tt in np.arange(time_fullspan.size-num_steps):

    ttlook = np.arange(tt,tt + Nlook)

    # get input hydro
    ds_watlev = hydro_fullspan_scaled[0,ttlook]
    ds_Hs = hydro_fullspan_scaled[1,ttlook]
    ds_Tp = hydro_fullspan_scaled[2,ttlook]
    ds_wdir = hydro_fullspan_scaled[3,ttlook]
    # get input PC amplitudes
    ds_mode1 = PCs_scaled[ttlook, 0]
    ds_mode2 = PCs_scaled[ttlook, 1]
    ds_mode3 = PCs_scaled[ttlook, 2]
    ds_mode4 = PCs_scaled[ttlook, 3]
    ds_mode5 = PCs_scaled[ttlook, 4]
    ds_mode6 = PCs_scaled[ttlook, 5]
    ds_mode7 = PCs_scaled[ttlook, 6]
    ds_mode8 = PCs_scaled[ttlook, 7]

    # check for nans....
    ds_data = np.column_stack((ds_watlev.T,ds_Hs.T,ds_Tp.T,ds_wdir.T,ds_mode1,ds_mode2,ds_mode3,ds_mode4,ds_mode5,ds_mode6,ds_mode7,ds_mode8))
    # ds_data = np.column_stack((ds_watlev.T, ds_Hs.T, ds_Tp.T, ds_wdir.T, ds_mode1, ds_mode2, ds_mode3, ds_mode4, ds_mode5, ds_mode6, ds_mode7))
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



############### Step 3 - Split into test/train ###############

# remove few odd sets with nans in hydro data
inputData_keep = inputData[:]
outputData_keep = outputData[:]

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


############### Step 4 - Design/Fit network ###############

# design network
model = Sequential()
model.add(LSTM(24, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.05))
# model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2])))
# output_1 = Dense(2)
# output_2 = Dense(numPCs-2)
# model.add(outputs=[output_1,output_2])
model.add(Dense(numPCs))

# custom loss function, defined as some data_loss*weight_data + phys_loss*weight_phys
def customLoss_wrapper(input_data):

    input_data = tf.cast(input_data, tf.float32)
    # ytrue_prevobs = input_data[:,-1,4:]
    # inv_ytrue_prevobs = ytrue_prevobs * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
    dx = 0.1
    # vol_true_prev = keras.backend.sum(inv_ytrue_prevobs*dx,axis=1)

    def customLoss(y_true, y_pred):

        # assign weights
        weight_dataEOF = 0.9
        weight_datavol = 0
        weight_dataelev = 0.1
        weight_dataDVol = 0
        # calculate profile stats
        inv_ypred = y_pred * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
        inv_ytrue = y_true * (PCs_max[0:numPCs] - PCs_min[0:numPCs]) + PCs_min[0:numPCs]
        vol_true = keras.backend.sum(inv_ytrue*dx,axis=1)
        vol_pred = keras.backend.sum(inv_ypred*dx,axis=1)
        # calculate individual losses
        dataelev_loss = keras.losses.MAE(inv_ytrue, inv_ypred)
        datavol_loss = keras.backend.abs(vol_true - vol_pred)
        # dataDVol_loss = keras.backend.abs(vol_true_prev - vol_pred)
        dataEOF_loss = keras.losses.MAE(y_true, y_pred)
        ## dataEOF_loss1 = keras.losses.MAE(y_true[0:4], y_pred[0:4])
        ## dataEOF_loss2 = keras.losses.MAE(y_true[5:], y_pred[5:])
        # calculate total loss
        sum_loss = weight_dataEOF*dataEOF_loss + weight_dataelev*dataelev_loss + weight_datavol*datavol_loss #+ weight_dataDVol*dataDVol_loss


        return sum_loss
    return customLoss

model.compile(loss=customLoss_wrapper(train_X), optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=60, batch_size=24, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

# plot history
fig, ax = plt.subplots()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
ax.set_xlabel('epoch (test/train cycle)')
ax.set_ylabel('error')

# ############### Step 4 v2 - try concatenating input & output -->  CONCATENATE NOT AVAIL FOR LSTM ###############
#
# model = Sequential()
# inp = Input(shape=(train_X.shape[1], train_X.shape[2]))
# input_custom = LSTM(45)(inp)
# input_custom = Dense(1)(input_custom)  # input
# output_custom_temp = Dense(numPCs)            # output layer
# output_custom = keras.layers.concatenate([input_custom, output_custom_temp])
# model_custom = Model(inputs=[input_custom], outputs=[output_custom])


# ############### Step 4 v3 - try fit_generator --> GENERATOR NOT AVAIL FOR LSTM ###############

# def generator(x, y, batch_size, Ntrain):
#     curIndex = 0
#     batch_x = np.zeros((batch_size,2))
#     batch_y = np.zeros((batch_size,1))
#     while True:
#         for i in range(batch_size):
#             batch_x[i] = x[curIndex,:]
#             batch_y[i] = y[curIndex,:]
#             i += 1;
#             if i == Ntrain:
#                 i = 0
#         yield batch_x, batch_y
#
# # set the seeds so that we get the same initialization across different trials
# seed_numpy = 0
# seed_tensorflow = 0
# seed(seed_numpy)
# tf.random.set_seed(seed_tensorflow)


# batch_size = 60
# model = Sequential()
# model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.15))
# model.add(Dense(numPCs))

# model.fit_generator(generator(train_X,train_y,batch_size), epochs=50)


############### Step 4 v4 - try sample_weights --> ONE LOSS, CANNOT ACCEPT 8 WEIGHTS ###############

# model = Sequential()
# model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.15))
# model.add(Dense(numPCs))
# model.compile(loss='mae',optimizer='adam',loss_weight=[0.3,0.2,0.15,0.1,0.075,0.075,0.075,0.025])
#
# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=60, validation_data=(test_X, test_y), verbose=2,
#                     shuffle=False)

############### Step 5 - Evaluate prediction ###############

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
fig, ax = plt.subplots(1,numPCs)
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
mode8_obs = 0#np.tile(EOFs[7,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,7]
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
mode8_pred = 0#np.tile(EOFs[7,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,7]
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


############### Step 6 - Evaluate particular instance ###############

picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_20Feb2025/'
with open(picklefile_dir+'stormy_times_fullspan.pickle','rb') as file:
   _,storm_flag,storm_timestart_all,storm_timeend_all = pickle.load(file)

tplot = pd.to_datetime(storm_timeend_all, unit='s', origin='unix')
plotflag = True
for nn in np.arange(5):
# for nn in np.arange(storm_timeend_all.size):

    tstart = storm_timeend_all[nn] + int(1*24*3600)
    iistart = np.where(np.isin(time_fullspan, tstart))[0].astype(int)

    # SHORT_TERM PREDICTION
    # Npred = Nlook-1
    Npred = 3

    prev_pred = np.empty((Npred,numPCs))*np.nan
    prev_obs = np.empty((Npred,numPCs))*np.nan
    numnan_hydro = np.empty((Npred,))*np.nan
    for tt in np.arange(Npred):

        # grab actual data as long Npred < Nlook
        iisetnn_PCs = np.arange(iistart + tt, iistart + Nlook-1)        # do not shift entire window, complement with prev pred.
        iisetnn_hydro = np.arange(iistart + tt, iistart + tt + Nlook-1) # shift entire window

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
                    print(sum(np.isnan(yv)))
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
            ds_watlev = hydro_fullspan_scaled[0,iisetnn_hydro]
            yv = ds_watlev
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_watlev[:] = hydro_interptmp
            # find and fill nans in waveheights
            ds_Hs = hydro_fullspan_scaled[1,iisetnn_hydro]
            yv = ds_Hs
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Hs[:] = hydro_interptmp
            # find and fill nans in wave periods
            ds_Tp = hydro_fullspan_scaled[2,iisetnn_hydro]
            yv = ds_Tp
            if sum(np.isnan(yv) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Tp[:] = hydro_interptmp
            # find and fill nans in wave directions
            ds_wdir = hydro_fullspan_scaled[3,iisetnn_hydro]
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
        mode8_obs = 0#np.tile(EOFs[7, :], (inv_test_y.shape[0], 1)).T * inv_test_y[:, 7]
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
        mode8_pred = 0#np.tile(EOFs[7, :], (inv_yhat.shape[0], 1)).T * inv_yhat[:, 7]
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


############### MISC  ###############


## Look at statistics of EOF1

y1 = PCs_fullspan[:,0]
y2 = PCs_scaled[:,0]
dy1 = y1[1:]-y1[0:-1]
dy2 = y2[1:]-y2[0:-1]
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')

fig, ax1 = plt.subplots()
ax1.plot(tplot,y1,'o')
ax2 = ax1.twinx()
ax2.plot(tplot,y2,'r.')
fig, ax1 = plt.subplots()
ax1.plot(tplot[1:],dy1,'o')
ax2 = ax1.twinx()
ax2.plot(tplot[1:],dy2,'r.')
fig, ax = plt.subplots()
ax.hist(dy1,bins=50)

PC1 = np.column_stack((inputData[:,:,4],outputData[:,0]))
dPC1 = PC1[:,1:] - PC1[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC1.T,'.')
flagrow,flagcol = np.where((dPC1 > 0.15) | (dPC1 < -0.15))
PC1[flagrow,flagcol+1] = np.nan
dPC1 = PC1[:,1:] - PC1[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC1.T,'.')
sum(np.sum(~np.isnan(PC1),axis=1) == 48)

PC2 = np.column_stack((inputData[:,:,5],outputData[:,1]))
dPC2 = PC2[:,1:] - PC2[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC2.T,'.')
flagrow,flagcol = np.where((dPC2 > 0.2) | (dPC2 < -0.2))
PC2[flagrow,flagcol+1] = np.nan
sum(np.sum(~np.isnan(PC2),axis=1) == 48)


PC3 = np.column_stack((inputData[:,:,6],outputData[:,2]))
dPC3 = PC3[:,1:] - PC3[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC3.T,'.')
flagrow,flagcol = np.where((dPC3 > 0.2) | (dPC3 < -0.2))
PC3[flagrow,flagcol+1] = np.nan
sum(np.sum(~np.isnan(PC3),axis=1) == 48)

PC4 = np.column_stack((inputData[:,:,7],outputData[:,3]))
dPC4 = PC4[:,1:] - PC4[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC4.T,'.')
flagrow,flagcol = np.where((dPC4 > 0.2) | (dPC4 < -0.2))
PC4[flagrow,flagcol+1] = np.nan
sum(np.sum(~np.isnan(PC4),axis=1) == 48)

PC5 = np.column_stack((inputData[:,:,8],outputData[:,4]))
dPC5 = PC5[:,1:] - PC5[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC5.T,'.')
flagrow,flagcol = np.where((dPC5 > 0.3) | (dPC5 < -0.3))
PC5[flagrow,flagcol+1] = np.nan
sum(np.sum(~np.isnan(PC5),axis=1) == 48)

PC6 = np.column_stack((inputData[:,:,9],outputData[:,5]))
dPC6 = PC6[:,1:] - PC6[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC6.T,'.')
flagrow,flagcol = np.where((dPC6 > 0.16) | (dPC6 < -0.16))
PC6[flagrow,flagcol+1] = np.nan

PC7 = np.column_stack((inputData[:,:,10],outputData[:,6]))
dPC7 = PC7[:,1:] - PC7[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC7.T,'.')
flagrow,flagcol = np.where((dPC7 > 0.2) | (dPC7 < -0.2))
PC7[flagrow,flagcol+1] = np.nan


PC8 = np.column_stack((inputData[:,:,11],outputData[:,7]))
dPC8 = PC8[:,1:] - PC8[:,0:-1]
fig, ax = plt.subplots()
ax.plot(dPC8.T,'.')
flagrow,flagcol = np.where((dPC8 > 0.35) | (dPC8 < -0.35))
PC8[flagrow,flagcol+1] = np.nan


########### LSTM for VOLUME and/or WIDTH #############

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'

with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_Nlook60_PCApostDVol_shifted.pickle', 'rb') as file:
    _, time_fullspan, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV, reconstruct_profNorm_fullspan, reconstruct_prof_fullspan, dataobs_shift_fullspan, dataobs_fullspan, data_profIDs_dVolThreshMet, data_hydro, datahydro_fullspan = pickle.load(file)
dataPCA_fullspan = reconstruct_prof_fullspan
dx = 0.1
xplot_shift = np.arange(reconstruct_prof_fullspan.shape[0])*dx

picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_20Feb2025/'
with open(picklefile_dir+'stormy_times_fullspan.pickle','rb') as file:
   _,storm_flag,storm_timestart_all,storm_timeend_all = pickle.load(file)

# apply 12-hr (~tidal) moving window over the data
hydro_fullspan_smooth = np.empty(shape=datahydro_fullspan.shape)*np.nan
nsmooth = 12
for nn in np.arange(4):
    ytmp = datahydro_fullspan[nn,:]
    hydro_fullspan_smooth[nn,:] = np.convolve(ytmp, np.ones(nsmooth) / nsmooth, mode='same')

# Calculate beach volume and width
mlw = -0.62
mwl = -0.13
zero = 0
mhw = 0.36
dune_toe = 3.22
upper_lim = 5.95
cont_elev = np.array([mlw,mwl,mhw,dune_toe,upper_lim]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts_pca, cmean, cstd = create_contours(dataPCA_fullspan.T,time_fullspan,xplot_shift,cont_elev)
beachVol_pca, beachVol_xc_pca, dBeachVol_dt_pca, total_beachVol_pca, total_dBeachVol_dt_pca, total_obsBeachWid_pca =  calculate_beachvol(dataPCA_fullspan.T,time_fullspan,xplot_shift,cont_elev,cont_ts_pca)
total_beachVol_pca[total_beachVol_pca == 0] = np.nan

tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')

beachVol = total_beachVol_pca
nsmooth = 20
ymean = np.convolve(beachVol, np.ones(nsmooth) / nsmooth, mode='same')
ts = pd.Series(beachVol)
ystd = ts.rolling(window=nsmooth, center=True).std()
bad_id = (abs(beachVol - ymean) >= 3 * ystd)
beachVol[bad_id] = np.nan
fig, ax = plt.subplots()
ax.plot(tplot,total_beachVol_pca,'o')
ax.plot(tplot,beachVol,'.')

beachWid = total_obsBeachWid_pca
nsmooth = 20
ymean = np.convolve(beachWid, np.ones(nsmooth) / nsmooth, mode='same')
ts = pd.Series(beachWid)
ystd = ts.rolling(window=nsmooth, center=True).std()
bad_id = (abs(beachWid - ymean) >= 3 * ystd)
beachWid[bad_id] = np.nan
fig, ax = plt.subplots()
ax.plot(tplot,total_obsBeachWid_pca,'o')
ax.plot(tplot,beachWid,'.')

mhw_xc = cont_ts_pca[2,:]
mlw_xc = cont_ts_pca[0,:]
dunetoe_xc = cont_ts_pca[3,:]
fig, ax = plt.subplots()
ax.plot(tplot,mhw_xc,'o')
xplot_tmp = np.arange(time_fullspan.size)
orig_x = xplot_tmp[~np.isnan(mhw_xc)]
orig_y = mhw_xc[~np.isnan(mhw_xc)]
target_x = xplot_tmp
max_gap = 8
mhw_xc = interpolate_with_max_gap(orig_x,orig_y,target_x,max_gap,False,False)
ax.plot(tplot,mhw_xc,'.')
ax.plot(tplot,mlw_xc)
ax.plot(tplot,dunetoe_xc)

# smooth
nsmooth = 12
beachVol_smooth = np.convolve(beachVol, np.ones(nsmooth) / nsmooth, mode='same')
beachWid_smooth = np.convolve(beachWid, np.ones(nsmooth) / nsmooth, mode='same')
mhw_xc_smooth = np.convolve(mhw_xc, np.ones(nsmooth) / nsmooth, mode='same')
mlw_xc_smooth = np.convolve(mlw_xc, np.ones(nsmooth) / nsmooth, mode='same')
dunetoe_xc_smooth = np.convolve(dunetoe_xc, np.ones(nsmooth) / nsmooth, mode='same')
fig, ax = plt.subplots()
ax.plot(tplot,hydro_fullspan_smooth.T,'.')
ax.plot(tplot,beachVol_smooth,'.')
ax.plot(tplot,beachWid_smooth,'.')
ax.plot(tplot,mhw_xc_smooth,'.')
ax.plot(tplot,mlw_xc_smooth,'.')
ax.plot(tplot,dunetoe_xc_smooth,'.')

# (X - X_min) / (X_max - X_min)
beachWid_min = np.nanmin(beachWid)
beachWid_max = np.nanmax(beachWid)
beachWid_mean = np.nanmean(beachWid)
beachWid_std = np.nanstd(beachWid)
beachVol_min = np.nanmin(beachVol)
beachVol_max = np.nanmax(beachVol)
beachVol_mean = np.nanmean(beachVol)
beachVol_std = np.nanstd(beachVol)
mhw_xc_min = np.nanmin(mhw_xc)
mhw_xc_max = np.nanmax(mhw_xc)
mhw_xc_mean = np.nanmean(mhw_xc)
mhw_xc_std = np.nanstd(mhw_xc)
mlw_xc_min = np.nanmin(mlw_xc)
mlw_xc_max = np.nanmax(mlw_xc)
mlw_xc_mean = np.nanmean(mlw_xc)
mlw_xc_std = np.nanstd(mlw_xc)
dunetoe_xc_min = np.nanmin(dunetoe_xc)
dunetoe_xc_max = np.nanmax(dunetoe_xc)
dunetoe_xc_mean = np.nanmean(dunetoe_xc)
dunetoe_xc_std = np.nanstd(dunetoe_xc)
beachWid_scaled = (beachWid - beachWid_min) / (beachWid_max - beachWid_min)
beachVol_scaled = (beachVol - beachVol_min) / (beachVol_max - beachVol_min)
mlw_xc_scaled = (mlw_xc - mlw_xc_min) / (mlw_xc_max - mlw_xc_min)
mhw_xc_scaled = (mhw_xc - mhw_xc_min) / (mhw_xc_max - mhw_xc_min)
dunetoe_xc_scaled = (dunetoe_xc - dunetoe_xc_min) / (dunetoe_xc_max - dunetoe_xc_min)
# beachWid_scaled = (beachWid - beachWid_mean) / beachWid_std
# beachVol_scaled = (beachVol - beachVol_mean) / beachVol_std
# mlw_xc_scaled = (mlw_xc - mlw_xc_mean) / mlw_xc_std
# mhw_xc_scaled = (mhw_xc - mhw_xc_mean) / mhw_xc_std
# dunetoe_xc_scaled = (dunetoe_xc - dunetoe_xc_mean) / dunetoe_xc_std


## Scale the hydro data
hydro_min = np.empty((4,))
hydro_max = np.empty((4,))
hydro_mean = np.empty((4,))
hydro_std = np.empty((4,))
hydro_fullspan_scaled = np.empty(shape=datahydro_fullspan.shape)*np.nan
for nn in np.arange(4):
    unscaled = datahydro_fullspan[nn,:]
    hydro_min[nn] = np.nanmin(unscaled)
    hydro_max[nn] = np.nanmax(unscaled)
    hydro_mean[nn] = np.nanmean(unscaled)
    hydro_std[nn] = np.nanstd(unscaled)
    hydro_fullspan_scaled[nn,:] = (unscaled - hydro_min[nn]) / (hydro_max[nn] - hydro_min[nn])
    # hydro_fullspan_scaled[nn, :] = (unscaled - hydro_mean[nn]) / hydro_std[nn]

####### LSTM #######

beachstat_fullspan = np.empty(shape=mhw_xc_scaled.shape)*np.nan
beachstat_fullspan[:] = mhw_xc_scaled[:]
beachstat_max = mhw_xc_max
beachstat_min = mhw_xc_min

Nlook = 12        # best for mhw
lstm_units = 49   # best for mhw
# Nlook = 40
# lstm_units = 80
num_steps = Nlook-1
numhydro = 4
numPCs = 1
num_features = numhydro + numPCs

inputData = np.empty((1,num_steps,num_features))*np.nan
outputData = np.empty((1,numPCs))*np.nan
for tt in np.arange(time_fullspan.size-num_steps):

    ttlook = np.arange(tt,tt + Nlook)

    # get input hydro
    ds_watlev = hydro_fullspan_scaled[0,ttlook]
    ds_Hs = hydro_fullspan_scaled[1,ttlook]
    ds_Tp = hydro_fullspan_scaled[2,ttlook]
    ds_wdir = hydro_fullspan_scaled[3,ttlook]
    # get input PC amplitudes
    ds_beachstat = beachstat_fullspan[ttlook]
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

# remove few odd sets with nans in hydro data
inputData_keep = inputData[:]
outputData_keep = outputData[:]

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

# design network
# 49/50, 47
model = Sequential()
model.add(LSTM(49, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.3))
model.add(Dense(numPCs))
model.compile(loss='mae', optimizer='adam')

# fit network
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
history = model.fit(train_X, train_y, epochs=60, batch_size=24, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

# plot history
fig, ax = plt.subplots()
plt.plot(history.history['loss'],'k-', label='train')
plt.plot(history.history['val_loss'],'r*-', label='test')
plt.legend()
plt.show()
ax.set_xlabel('epoch (test/train cycle)')
ax.set_ylabel('testing error')

# test network
yhat = model.predict(test_X)
inv_yhat = yhat * (beachstat_max - beachstat_min) + beachstat_min
inv_test_y = test_y * (beachstat_max - beachstat_min) + beachstat_min
fig, ax = plt.subplots()
ax.plot([np.min(inv_test_y)-5,np.max(inv_test_y)+5],[np.min(inv_test_y)-5,np.max(inv_test_y)+5],':k')
ax.plot(inv_test_y,inv_yhat,'o',alpha=0.2)
ax.set_xlabel('observed $Xc_{MHW}$ [m]')
ax.set_ylabel('predicted $Xc_{MHW}$ [m]')
ax.set_xlim([np.min(inv_test_y)-3,np.max(inv_test_y)+3])
ax.set_ylim([np.min(inv_test_y)-3,np.max(inv_test_y)+3])
ax.grid()
RMSE = np.sqrt(np.mean((inv_test_y - inv_yhat)**2))
ax.set_title('RMSE = '+str("%0.2f" % RMSE)+' m')
fig.set_size_inches(4.1,3.8)
plt.tight_layout()


# test time series prediction
tplot = pd.to_datetime(storm_timeend_all, unit='s', origin='unix')
# plotflag = True
# iistart_plot = [686, 10808, 64975]  # for mhw_xc
iistart_plot = [ 1339, 880, 1552]
figwid = []
fight = []
ii_withdata = np.where(~np.isnan(beachstat_fullspan))[0]
# iistart_plot = ii_withdata[random.sample(range(0,ii_withdata.size), 30)]
# for nn in np.arange(20):
for nn in np.arange(len(iistart_plot)):
# for nn in np.arange(storm_timeend_all.size):

    # tstart = storm_timeend_all[nn] + int(2.5*24*3600)
    # iistart = np.where(np.isin(time_fullspan, tstart))[0].astype(int)
    iistart = iistart_plot[nn]

    # SHORT_TERM PREDICTION
    # Npred = Nlook*100
    Npred = 250

    prev_pred = np.empty((Npred,))*np.nan
    prev_obs = np.empty((Npred,))*np.nan
    numnan_hydro = np.empty((Npred,))*np.nan
    init_obs = np.empty((Nlook-1,))*np.nan
    for tt in np.arange(Npred):

        iisetnn_hydro = np.arange(iistart + tt, iistart + tt + Nlook-1) # shift entire window

        # grab actual data as long Npred < Nlook
        if tt <= Nlook:
            iisetnn_beachstat = np.arange(iistart + tt, iistart + Nlook-1)        # do not shift entire window, complement with prev pred.
            # find and fill nans in PCs
            beachstat_setnn = beachstat_fullspan[iisetnn_beachstat]
            ytest = np.empty(shape=beachstat_setnn.shape)*np.nan
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
                    if sum(np.isnan(yv)) > 0:
                        print(sum(np.isnan(yv)))
                        xq = np.arange(Nlook - 1 - tt)
                        xv = xq[~np.isnan(yv)]
                        yv = yv[~np.isnan(yv)]
                        beachstatjj_interptmp = np.interp(xq, xv, yv)
                        ds_beachstat[:] = beachstatjj_interptmp
                    else:
                        ds_beachstat[:] = yv
                    # add previous predictions to fill out rest of PCs, if tt > 0
                    if (tt > 0) & (tt <= Nlook - 1):
                        ds_beachstat = np.append(ds_beachstat, prev_pred[0:tt])
                    elif tt == 0:
                        init_obs[:] = ds_beachstat[:]
        else:
            ytest = np.empty((Nlook-1,1))*np.nan
            ds_beachstat = prev_pred[tt - Nlook:tt - 1]

        # find and fill nans in water levels
        ds_watlev = hydro_fullspan_scaled[0,iisetnn_hydro]
        ds_Hs = hydro_fullspan_scaled[1,iisetnn_hydro]
        ds_Tp = hydro_fullspan_scaled[2,iisetnn_hydro]
        ds_wdir = hydro_fullspan_scaled[3,iisetnn_hydro]
        if sum(~np.isnan(ds_watlev)) != 0:
            yv = ds_watlev
            if (sum(np.isnan(yv)) > 0) == True:
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_watlev[:] = hydro_interptmp
            # find and fill nans in waveheights
            yv = ds_Hs
            if (sum(np.isnan(yv)) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Hs[:] = hydro_interptmp
            # find and fill nans in wave periods
            yv = ds_Tp
            if (sum(np.isnan(yv)) > 0):
                xq = np.arange(Nlook-1)
                xv = xq[~np.isnan(yv)]
                yv = yv[~np.isnan(yv)]
                hydro_interptmp = np.interp(xq, xv, yv)
                ds_Tp[:] = hydro_interptmp
            # find and fill nans in wave directions
            yv = ds_wdir
            if (sum(np.isnan(yv)) > 0):
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
        inputData[0, :, 4] = ds_beachstat[:]

        # make predicition
        test_X = np.empty(shape=inputData.shape)*np.nan
        test_X[:] = inputData[:]
        yhat = model.predict(test_X)

        # save last prediction as input for the next set
        prev_pred[tt] = yhat[:]
        prev_obs[tt] = beachstat_fullspan[iisetnn_hydro[-1]+1]

    if plotflag:
        # inverse scale the results
        inv_yhat = prev_pred * (beachstat_max - beachstat_min) + beachstat_min
        inv_test_y = prev_obs * (beachstat_max - beachstat_min) + beachstat_min
        inv_input_y = init_obs * (beachstat_max - beachstat_min) + beachstat_min
        slope, intercept, rval_modes, pval_modes, stderr_modes = sp.stats.linregress(inv_test_y, inv_yhat)
        rmse_modes = np.sqrt(np.nanmean((inv_test_y - inv_yhat)**2))
        nrmse_modes = np.sqrt(np.nanmean((inv_test_y - inv_yhat) ** 2))/np.nanmean(inv_test_y)

        # now plot prediction vs observed over time
        # fig, ax = plt.subplots(1, 2)
        # ax[0].scatter(inv_test_y,inv_yhat,5,np.arange(Npred),alpha=0.95,cmap='plasma')
        # ax[0].grid()
        # # ax.set_ylim(minval, maxval)
        # # ax.set_xlim(minval, maxval)
        # ax[0].set_xlabel('observed')
        # ax[0].set_ylabel('predicted')

        # plot against observed data
        fig, ax = plt.subplots()
        xplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
        iiplot = np.arange(iistart, iistart + Nlook-1)
        normxplot = np.arange(-(Nlook-1),0)
        # ax.plot(xplot[iiplot], inv_input_y, '.-k')
        ax.plot(normxplot, inv_input_y, '.-k')
        # min_taxis = xplot[iiplot[0]]
        min_taxis = normxplot[0]
        iiplot = np.arange(iistart + Nlook, iistart + Nlook + Npred)
        normxplot = np.arange(0,Npred)
        # ax.plot(xplot[iiplot], inv_test_y,'.-',color='grey')
        ax.plot(normxplot, inv_test_y, '.-', color='grey')
        # max_taxis = xplot[iiplot[-1]]
        max_taxis = normxplot[-1]
        ax.grid()
        ax.set_xlabel('$t_{predict}$ [hr]')
        ax.set_ylabel('$Xc_{MHW}$ [m], observed')
        ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        # ax2.scatter(xplot[iiplot],inv_yhat,10,iiplot,cmap='plasma')
        ax2.scatter(normxplot,inv_yhat,10,np.arange(normxplot.size),cmap='plasma',vmin=0,vmax=Npred)
        ax2.set_ylabel('$Xc_{MHW}$ [m], predicted')
        ax.set_title(str(iistart))
        # date_form = DateFormatter("%m/%d/%y")
        # ax.xaxis.set_major_formatter(date_form)
        ax2.set_xlim(min_taxis,max_taxis)
        ax.set_xlim(min_taxis,max_taxis)
        # ax.tick_params(axis='x', labelrotation=45)
        # ax2.tick_params(axis='x', labelrotation=45)
        fig.set_size_inches(8.352, 2.792)
        plt.tight_layout()
        # yplot = inv_input_y[1:] - inv_input_y[0:-1]
        # ax[1].plot(xplot[iiplot[1:]], yplot, '.-k')
        # iiplot = np.arange(iistart + Nlook, iistart + Nlook + Npred)
        # yplot = inv_test_y[1:] - inv_test_y[0:-1]
        # ax[1].plot(xplot[iiplot[1:]], yplot, '.k')
        # yplot = inv_yhat[1:] - inv_yhat[0:-1]
        # ax[1].scatter(xplot[iiplot[1:]], yplot, 5, iiplot[1:], cmap='plasma')
        # ax[1].grid()
        # ax[1].set_xlabel('time')


# 1339, 880, 1552, 45746, 18331, 18346