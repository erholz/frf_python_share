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

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
with open(picklefile_dir+'datasets_ML_14Dec2024.pickle', 'rb') as file:
    datasets_ML = pickle.load(file)
    num_datasets = len(datasets_ML)
with open(picklefile_dir+'topobathy_finalCheckBeforePCA_Zdunetoe_3p2m.pickle','rb') as file:
    topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)

iiDS = iiDS_passFinalCheck[:]
num_features = 8
num_steps = 24*4
num_datasets = iiDS.size
train_X = np.empty((num_datasets,num_steps,num_features))
train_X[:] = np.nan
for jj in np.arange(iiDS.size):
    # get input hydro
    varname = 'dataset_' + str(int(jj))
    exec('waterlevel = datasets_ML["' + varname + '"]["set_waterlevel"]')
    ds_watlev = waterlevel
    exec('Hs = datasets_ML["' + varname + '"]["set_Hs8m"]')
    ds_Hs = Hs
    exec('Tp = datasets_ML["' + varname + '"]["set_Tp8m"]')
    ds_Tp = Tp
    exec('wdir = datasets_ML["' + varname + '"]["set_dir8m"]')
    ds_wdir = wdir
    # load into training matrix
    train_X[jj, :, 0] = ds_watlev
    train_X[jj, :, 1] = ds_Hs
    train_X[jj, :, 2] = ds_Tp
    train_X[jj, :, 3] = ds_wdir



# # from sample - load data and prep data
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

# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input (all but last column) and output (1 column)
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# # Define the Keras TensorBoard callback.
# logdir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# callbacks=[tensorboard_callback]

# fit network
history = model.fit(train_X, train_y, epochs=40, batch_size=32, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


fig, ax = plt.subplots()
yplot = values[:,-1]
tplot = np.arange(yplot.size)/(24*365)
plt.plot(tplot,yplot)
fig, ax = plt.subplots(2,1)
ax[0].plot(inv_y,label='true')
ax[0].set_title('obs')
ax[0].set_ylabel('PPM')
# ax[0].plot(inv_yhat,label='pred')
ax[1].plot(inv_y - inv_yhat)
ax[1].set_title('obs-pred')
ax[1].set_ylabel('PPM')

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



