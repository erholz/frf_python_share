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
hydro_std = np.empty((4,))
PCs_min = np.empty((PCs_fullspan.shape[1],))
PCs_max = np.empty((PCs_fullspan.shape[1],))
PCs_avg = np.empty((PCs_fullspan.shape[1],))
PCs_std = np.empty((PCs_fullspan.shape[1],))
for nn in np.arange(4):
    unscaled = data_hydro[:,:,nn].reshape((Nlook*num_datasets,1))
    hydro_min[nn] = np.nanmin(unscaled)
    hydro_max[nn] = np.nanmax(unscaled)
    hydro_avg[nn] = np.nanmean(unscaled)
    hydro_std[nn] = np.nanstd(unscaled)
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
    PCs_std[nn] = np.nanstd(unscaled)
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
model.add(LSTM(45, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(5))
model.compile(loss='mae', optimizer='adam')

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

# X = X_scaled * (X_max - X_min) + X_min

yhat = model.predict(test_X)
# inv_yhat = yhat * (PCs_max[0] - PCs_min[0]) + PCs_min[0]
# inv_test_y = test_y * (PCs_max[0] - PCs_min[0]) + PCs_min[0]
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

fig, ax = plt.subplots(1,5)
ax[0].plot([minval,maxval],[minval,maxval],'k')
ax[0].plot(inv_test_y[:,0],inv_yhat[:,0],'.',alpha=0.1)
ax[0].grid()
minval = -75
maxval = 75
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
mode1_pred = np.tile(EOFs[0,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,0]
mode2_pred = np.tile(EOFs[1,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,1]
mode3_pred = np.tile(EOFs[2,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,2]
mode4_pred = np.tile(EOFs[3,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,3]
mode5_pred = np.tile(EOFs[4,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,4]
profspred_norm = mode1_pred + mode2_pred + mode3_pred + mode4_pred + mode5_pred
profspred_T = profspred_norm.T * dataStd.T + dataMean.T
profpred = profspred_T.T
fig, ax = plt.subplots()
ax.plot(xplot,profpred)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Predicted')
mode1_obs = np.tile(EOFs[0,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,0]
mode2_obs = np.tile(EOFs[1,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,1]
mode3_obs = np.tile(EOFs[2,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,2]
mode4_obs = np.tile(EOFs[3,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,3]
mode5_obs = np.tile(EOFs[4,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,4]
profsobs_norm = mode1_obs + mode2_obs + mode3_obs + mode4_obs + mode5_obs
profsobs_T = profsobs_norm.T * dataStd.T + dataMean.T
fig, ax = plt.subplots()
profobs = profsobs_T.T
ax.plot(xplot,profobs)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Observed* (PCs)')

fig, ax = plt.subplots()
ax.plot(xplot,profpred-profobs)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Obs - Pred')


############### Step 5 - Evaluate particular instance ###############



pred_prof = profpred[:,jj]
obs_prof = profobs[:,jj]
inv_testX = test_X[jj,:,5:] * (PCs_max[0:5] - PCs_min[0:5]) + PCs_min[0:5]
mode1_obsjj = np.tile(EOFs[0,:],(inv_testX[:,0].size,1)).T * inv_testX[:,0]
mode2_obsjj = np.tile(EOFs[1,:],(inv_testX[:,1].size,1)).T * inv_testX[:,1]
mode3_obsjj = np.tile(EOFs[2,:],(inv_testX[:,2].size,1)).T * inv_testX[:,2]
mode4_obsjj = np.tile(EOFs[3,:],(inv_testX[:,3].size,1)).T * inv_testX[:,3]
mode5_obsjj = np.tile(EOFs[4,:],(inv_testX[:,4].size,1)).T * inv_testX[:,4]
profsobsjj_norm = mode1_obsjj + mode2_obsjj + mode3_obsjj + mode4_obsjj + mode5_obsjj
profsobsjj_T = profsobsjj_norm.T * dataStd.T + dataMean.T
profobsjj = profsobsjj_T.T
fig, ax = plt.subplots()
ax.plot(profobsjj,linewidth=0.5)
ax.plot(obs_prof,'--',color=[0.5,0.5,0.5],linewidth=2)
ax.plot(pred_prof,'k:')

