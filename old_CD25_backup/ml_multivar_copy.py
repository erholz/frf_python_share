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


# FUNC from sample - convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

############### Step 1 - Load and prep data ###############

# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
picklefile_dir = 'D:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_10Dec2024/'
with open(picklefile_dir+'datasets_ML_14Dec2024.pickle', 'rb') as file:
    datasets_ML = pickle.load(file)
    num_datasets = len(datasets_ML)
with open(picklefile_dir+'topobathy_finalCheckBeforePCA_Zdunetoe_3p2m.pickle','rb') as file:
    topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)
with open(picklefile_dir+'topobathy_PCA_Zdunetoe_3p2m_Lmin_75.pickle','rb') as file:
    dataNorm,dataMean,dataStd,PCs,EOFs,APEV,reconstruct_profileNorm,reconstruct_profile = pickle.load(file)
with open(picklefile_dir+'topobathy_reshape_indexKeeper.pickle','rb') as file:
    tt_unique,origin_set,dataset_index_fullspan,dataset_index_plot = pickle.load(file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_12Jan2025/'
with open(picklefile_dir + 'hydro_datasetsForML.pickle', 'rb') as file:
    hydro_datasetsForML,nanflag_hydro = pickle.load(file)

# Re-scale data
Nlook = 24*4
num_datasets = len(datasets_ML)
hydro_datasetsForML_scaled = np.empty(shape=hydro_datasetsForML.shape)
PCs_scaled = np.empty(shape=PCs.shape)
hydro_min = np.empty((4,))
hydro_max = np.empty((4,))
hydro_avg = np.empty((4,))
hydro_std = np.empty((4,))
PCs_min = np.empty((PCs.shape[1],))
PCs_max = np.empty((PCs.shape[1],))
PCs_avg = np.empty((PCs.shape[1],))
PCs_std = np.empty((PCs.shape[1],))
for nn in np.arange(4):
    unscaled = hydro_datasetsForML[:,:,nn].reshape((Nlook*num_datasets,1))
    hydro_min[nn] = np.nanmin(unscaled)
    hydro_max[nn] = np.nanmax(unscaled)
    hydro_avg[nn] = np.nanmean(unscaled)
    hydro_std[nn] = np.nanstd(unscaled)
    unscaled_reshape = unscaled.reshape((num_datasets,Nlook))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(unscaled)
    scaled_reshape = scaled.reshape((num_datasets,Nlook))
    hydro_datasetsForML_scaled[:,:,nn] = scaled_reshape
for nn in np.arange(PCs.shape[1]):
    unscaled = PCs[:,nn].reshape((PCs.shape[0],1))
    PCs_min[nn] = np.nanmin(unscaled)
    PCs_max[nn] = np.nanmax(unscaled)
    PCs_avg[nn] = np.nanmean(unscaled)
    PCs_std[nn] = np.nanstd(unscaled)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(unscaled)
    PCs_scaled[:, nn] = np.squeeze(scaled)

iiDS = iiDS_passFinalCheck[:]
num_features = 8
num_steps = 24*4 - 1
num_datasets = iiDS.size
# num_datasets = 200
inputData = np.empty((num_datasets,num_steps,num_features))
inputData[:] = np.nan
# outputData = np.empty((num_datasets,))
outputData = np.empty((num_datasets,4))
outputData[:] = np.nan
numinset = np.empty((num_datasets,))
numinset[:] = np.nan
avgdt = np.empty((num_datasets,))
avgdt[:] = np.nan
nanhydro_data = np.zeros((num_datasets,))
nanPCs_data = np.zeros((num_datasets,))
for jj in np.arange(1,num_datasets):
    dsjj = iiDS[jj]
    # get input hydro
    ds_watlev = hydro_datasetsForML_scaled[dsjj,:,0]
    ds_Hs = hydro_datasetsForML_scaled[dsjj,:,1]
    ds_Tp = hydro_datasetsForML_scaled[dsjj,:,2]
    ds_wdir = hydro_datasetsForML_scaled[dsjj,:,3]
    if sum(nanflag_hydro[dsjj,:]) > 0:
        nanhydro_data[jj] = 1
    # get input PC amplitudes
    tmpii = np.where(np.isin(iirow_finalcheck, dataset_index_plot[dsjj, :]))[0].astype(int)
    PCs_setjj = PCs_scaled[tmpii, :]
    numinset[jj] = sum(tmpii)
    # load into training matrices
    inputData[jj, :, 0] = ds_watlev[0:-1]
    inputData[jj, :, 1] = ds_Hs[0:-1]
    inputData[jj, :, 2] = ds_Tp[0:-1]
    inputData[jj, :, 3] = ds_wdir[0:-1]
    inputData[jj, :, 4] = PCs_setjj[0:-1,0]
    inputData[jj, :, 5] = PCs_setjj[0:-1,1]
    inputData[jj, :, 6] = PCs_setjj[0:-1,2]
    inputData[jj, :, 7] = PCs_setjj[0:-1,3]
    # outputData[jj] = PCs_setjj[-1,0]
    outputData[jj,0] = PCs_setjj[-1,0]
    outputData[jj,1] = PCs_setjj[-1,1]
    outputData[jj,2] = PCs_setjj[-1,3]
    outputData[jj,3] = PCs_setjj[-1,4]
    if np.nansum(np.isnan(PCs_setjj)) > 0:
        nanPCs_data[jj] = 1



    # # test to see if we pull correct PCs for each setjj of 96
    # tmpii = dataset_index_plot[dsjj,:].astype(int)
    # Z_setjj = topobathy_check_xshoreFill[:,tmpii]
    # fig, ax = plt.subplots()
    # ax.plot(Z_setjj)
    # tmpii = np.where(np.isin(iirow_finalcheck,dataset_index_plot[dsjj,:]))[0].astype(int)
    # PCs_setjj = PCs[tmpii,:]
    # mode1 = np.tile(EOFs[0,:],(tmpii.size,1)).T * PCs_setjj[:,0]
    # mode2 = np.tile(EOFs[1,:],(tmpii.size,1)).T * PCs_setjj[:,1]
    # mode3 = np.tile(EOFs[2,:],(tmpii.size,1)).T * PCs_setjj[:,2]
    # mode4 = np.tile(EOFs[3,:],(tmpii.size,1)).T * PCs_setjj[:,3]
    # mode5 = np.tile(EOFs[4,:],(tmpii.size,1)).T * PCs_setjj[:, 4]
    # mode6 = np.tile(EOFs[5,:],(tmpii.size,1)).T * PCs_setjj[:, 5]
    # profs_norm = mode1 + mode2 + mode3 + mode4 + mode5 + mode6
    # profs_setjjT = profs_norm.T * dataStd.T + dataMean.T
    # profs_setjj = profs_setjjT.T
fig, ax = plt.subplots()
ax.plot(numinset,'o')
# fig, ax = plt.subplots()
# ax.plot(avgdt,'o')



# # FROM SAMPLE -- load data and prep data
# fname = "C:/Users/rdchlerh/Downloads/raw.csv"
# def parse(x):
# 	return dt.strptime(x, '%Y %m %d %H')
# dataset = read_csv(fname,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# dataset.drop('No', axis=1, inplace=True)
# # manually specify column names
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# dataset.index.name = 'date'
# # mark all NA values with 0
# dataset['pollution'].fillna(0, inplace=True)
# # drop the first 24 hours
# dataset = dataset[24:]
# # summarize first 5 rows
# print(dataset.head(5))
# # save to file
# dataset.to_csv('C:/Users/rdchlerh/Downloads/pollution.csv')
#
# # load dataset
# fname = "C:/Users/rdchlerh/Downloads/pollution.csv"
# dataset = read_csv(fname, header=0, index_col=0)
# values = dataset.values
# # integer encode wind direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# # ensure all data is float
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# # drop columns we don't want to predict
# reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# print(reframed.head())



############### Step 2 - Split into test/train ###############

# remove few odd sets with nans in hydro data
iiremove = (nanhydro_data > 0) + (nanPCs_data > 0)
iiremove[0] = True
iikeep = ~iiremove
inputData_keep = inputData[iikeep,:,:]
outputData_keep = outputData[iikeep,:]

# separate test and train IDs
frac = 1/3
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
train_y = np.empty((Ntrain,4))
train_y[:] = outputData_keep[iitrain,:]
# load testing
test_X = np.empty((Ntest,num_steps,num_features))
test_X[:,:,:] = inputData_keep[iitest,:,:]
# test_y = np.empty((Ntest,))
# test_y[:] = outputData_keep[iitest]
test_y = np.empty((Ntest,4))
test_y[:] = outputData_keep[iitest,:]


# # FROM SAMPLE -- split into train and test sets
# values = reframed.values
# n_train_hours = 365 * 24
# train = values[:n_train_hours, :]
# test = values[n_train_hours:, :]
# # split into input (all but last column) and output (1 column)
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
# # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#
# # design network
# model = Sequential()
# model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
#
# # # Define the Keras TensorBoard callback.
# # logdir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
# # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# # callbacks=[tensorboard_callback]

############### Step 3 - Design/Fit network ###############

# design network
model = Sequential()
model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=40, batch_size=20, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
fig, ax = plt.subplots()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
ax.set_xlabel('epoch (test/train cycle)')
ax.set_ylabel('error')

############### Step 4 - Make prediction ###############

# X = X_scaled * (X_max - X_min) + X_min

yhat = model.predict(test_X)
# inv_yhat = yhat * (PCs_max[0] - PCs_min[0]) + PCs_min[0]
# inv_test_y = test_y * (PCs_max[0] - PCs_min[0]) + PCs_min[0]
# fig, ax = plt.subplots()
# ax.plot(inv_test_y,inv_yhat,'.',alpha=0.1)
# plt.grid()
inv_yhat = yhat * (PCs_max[0:4] - PCs_min[0:4]) + PCs_min[0:4]
inv_test_y = test_y * (PCs_max[0:4] - PCs_min[0:4]) + PCs_min[0:4]
fig, ax = plt.subplots(2,2)
ax[0,0].plot(inv_test_y[:,0],inv_yhat[:,0],'.',alpha=0.1)
plt.grid()
ax[0,1].plot(inv_test_y[:,1],inv_yhat[:,1],'.',alpha=0.1)
plt.grid()
ax[1,0].plot(inv_test_y[:,2],inv_yhat[:,2],'.',alpha=0.1)
plt.grid()
ax[1,1].plot(inv_test_y[:,3],inv_yhat[:,3],'.',alpha=0.1)
plt.grid()

# compare observed at predicted output profiles...
mode1_pred = np.tile(EOFs[0,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,0]
mode2_pred = np.tile(EOFs[1,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,1]
mode3_pred = np.tile(EOFs[2,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,2]
mode4_pred = np.tile(EOFs[3,:],(inv_test_y.shape[0],1)).T * inv_test_y[:,3]
profspred_norm = mode1_pred + mode2_pred + mode3_pred + mode4_pred
profspred_T = profspred_norm.T * dataStd.T + dataMean.T
profpred = profspred_T.T
fig, ax = plt.subplots()
ax.plot(profpred)
mode1_obs = np.tile(EOFs[0,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,0]
mode2_obs = np.tile(EOFs[1,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,1]
mode3_obs = np.tile(EOFs[2,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,2]
mode4_obs = np.tile(EOFs[3,:],(inv_yhat.shape[0],1)).T * inv_yhat[:,3]
profsobs_norm = mode1_obs + mode2_obs + mode3_obs + mode4_obs
profsobs_T = profsobs_norm.T * dataStd.T + dataMean.T
fig, ax = plt.subplots()
profobs = profsobs_T.T
ax.plot(profobs)
jj = 3500
pred_prof = profpred[:,jj]
obs_prof = profobs[:,jj]
inv_testX = test_X[jj,:,4:] * (PCs_max[0:4] - PCs_min[0:4]) + PCs_min[0:4]
mode1_obsjj = np.tile(EOFs[0,:],(inv_testX[:,0].size,1)).T * inv_testX[:,0]
mode2_obsjj = np.tile(EOFs[1,:],(inv_testX[:,1].size,1)).T * inv_testX[:,1]
mode3_obsjj = np.tile(EOFs[2,:],(inv_testX[:,2].size,1)).T * inv_testX[:,2]
mode4_obsjj = np.tile(EOFs[3,:],(inv_testX[:,3].size,1)).T * inv_testX[:,3]
profsobsjj_norm = mode1_obsjj + mode2_obsjj + mode3_obsjj + mode4_obsjj
profsobsjj_T = profsobsjj_norm.T * dataStd.T + dataMean.T
profobsjj = profsobsjj_T.T
fig, ax = plt.subplots()
ax.plot(profobsjj,linewidth=0.5)
ax.plot(obs_prof,'--',color=[0.5,0.5,0.5],linewidth=2)
ax.plot(pred_prof,'k:')




jj=1
mode1_obsjj = np.tile(EOFs[0,:],(inputData[jj,:,4].size,1)).T * inputData[jj,:,4]
mode2_obsjj = np.tile(EOFs[1,:],(inputData[jj,:,5].size,1)).T * inputData[jj,:,5]
mode3_obsjj = np.tile(EOFs[2,:],(inputData[jj,:,6].size,1)).T * inputData[jj,:,6]
mode4_obsjj = np.tile(EOFs[3,:],(inputData[jj,:,7].size,1)).T * inputData[jj,:,7]
profsobsjj_norm = mode1_obsjj + mode2_obsjj + mode3_obsjj + mode4_obsjj
profsobsjj_T = profsobsjj_norm.T * dataStd.T + dataMean.T
profobsjj = profsobsjj_T.T
fig, ax = plt.subplots()
ax.plot(profobsjj)



# # FROM SAMPlE - make a prediction
# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
#
# fig, ax = plt.subplots()
# yplot = values[:,-1]
# tplot = np.arange(yplot.size)/(24*365)
# plt.plot(tplot,yplot)
# fig, ax = plt.subplots(2,1)
# ax[0].plot(inv_y,label='true')
# ax[0].set_title('obs')
# ax[0].set_ylabel('PPM')
# # ax[0].plot(inv_yhat,label='pred')
# ax[1].plot(inv_y - inv_yhat)
# ax[1].set_title('obs-pred')
# ax[1].set_ylabel('PPM')

fname ='C:/Users/rdchlerh/Downloads/model_plot.png'
plot_model(model,to_file=fname, show_shapes=True, show_layer_names=True, show_layer_activations=True, expand_nested=True, show_trainable=True)

# visualkeras.layered_view(model,to_file=fname, legend=True) # without custom font


weights = model.layers[0].get_weights()
w1 = weights[0]
w2 = weights[1]
w3 = weights[2]

model.summary()
for x in model.layers[0].weights:
    print(x.name,'-->',x.shape)


# plot LSTM layer weights and biases
weights = model.layers[0].get_weights()
fig, ax = plt.subplots(1,4)
# INPUT WEIGHTS
ax[0].pcolormesh(weights[0][:,:25],vmin=-0.65,vmax=0.65,cmap='coolwarm')
# cbar1 = fig.colorbar(ph, ax=ax[0])
plt.suptitle('kernel weights')
ax[0].set_xlabel('LSTM unit/cell')
# ax[0].set_ylabel('input variable')
ax[0].set_yticks(np.arange(0.5,8,1))
ax[0].set_yticklabels(['PPM(t-1)','dew(t-1)','temp(t-1)','press(t-1)','Wdir(t-1)','Wspd(t-1)','Snow(t-1)','Rain(t-1)'])
ax[0].set_title('Input')
# FORGET WEIGHTS
ax[1].pcolormesh(weights[0][:,25:50],vmin=-0.65,vmax=0.65,cmap='coolwarm')
ax[1].set_xlabel('LSTM unit/cell')
ax[1].set_yticks(np.arange(0.5,8,1))
ax[1].set_title('Forget')
ax[1].set_yticklabels([])
# CELL WEIGHTS
ax[2].pcolormesh(weights[0][:,50:75],vmin=-0.65,vmax=0.65,cmap='coolwarm')
ax[2].set_xlabel('LSTM unit/cell')
ax[2].set_yticks(np.arange(0.5,8,1))
ax[2].set_title('Cell')
ax[2].set_yticklabels([])
# OUTPUT WEIGHTS
ax[3].pcolormesh(weights[0][:,75:100],vmin=-0.65,vmax=0.65,cmap='coolwarm')
ax[3].set_xlabel('LSTM unit/cell')
ax[3].set_yticks(np.arange(0.5,8,1))
ax[3].set_title('Output')
ax[3].set_yticklabels([])

# Repeat for recurrent kernel weights
weights = model.layers[0].get_weights()
fig, ax = plt.subplots(1,4)
# INPUT WEIGHTS
ax[0].pcolormesh(weights[1][:,:25],vmin=-0.65,vmax=0.65,cmap='coolwarm')
# cbar1 = fig.colorbar(ph, ax=ax[0])
plt.suptitle('recurrent kernel weights')
ax[0].set_xlabel('LSTM unit/cell')
ax[0].set_ylabel('LSTM unit/cell')
# ax[0].set_yticks(np.arange(0.5,8,1))
# ax[0].set_yticklabels(['PPM(t-1)','dew(t-1)','temp(t-1)','press(t-1)','Wdir(t-1)','Wspd(t-1)','Snow(t-1)','Rain(t-1)'])
ax[0].set_title('Input')
# FORGET WEIGHTS
ax[1].pcolormesh(weights[1][:,25:50],vmin=-0.65,vmax=0.65,cmap='coolwarm')
ax[1].set_xlabel('LSTM unit/cell')
# ax[1].set_yticks(np.arange(0.5,8,1))
ax[1].set_title('Forget')
ax[1].set_yticklabels([])
# CELL WEIGHTS
ax[2].pcolormesh(weights[1][:,50:75],vmin=-0.65,vmax=0.65,cmap='coolwarm')
ax[2].set_xlabel('LSTM unit/cell')
# ax[2].set_yticks(np.arange(0.5,8,1))
ax[2].set_title('Cell')
ax[2].set_yticklabels([])
# OUTPUT WEIGHTS
ax[3].pcolormesh(weights[1][:,75:100],vmin=-0.65,vmax=0.65,cmap='coolwarm')
ax[3].set_xlabel('LSTM unit/cell')
# ax[3].set_yticks(np.arange(0.5,8,1))
ax[3].set_title('Output')
ax[3].set_yticklabels([])







fig, ax = plt.subplots()
ph = ax.pcolormesh(weights[2],vmin=-0.65,vmax=0.65,cmap='coolwarm')
cbar3 = fig.colorbar(ph, ax=ax)
plt.title('bias')



