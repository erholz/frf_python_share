import pickle
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load the dataframe
from sklearn.decomposition import PCA  # to apply PCA
import os
from funcs.getFRF_funcs.getFRF_lidar import *
from funcs.create_contours import *
import scipy as sp
from astropy.convolution import convolve
import seaborn as sns



## LOAD TOPOBATHY
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'topobathy_scale&shift.pickle','rb') as file:
   topobathy_shift_plot,topobathy_scale_plot = pickle.load(file)



# DEFINE DATASET FOR PCA
nx = topobathy_shift_plot.shape[0]
nt = topobathy_shift_plot.shape[1]
dx = 0.1
xplot = dx*np.arange(nx)
check_data = topobathy_shift_plot[xplot < 100,:]

yy = np.nansum(np.isnan(check_data),axis=0 )
yy = yy[(yy < check_data.shape[0]) & (yy > 0)]
fig, ax = plt.subplots()
plt.hist(yy,bins=np.arange(0,700,25))


# ok, now find our WHERE (what data_set) those profiles are located...




rows_nonans = np.where(np.nansum(np.isnan(check_data),axis=0 ) == check_data.shape[0])[0]

# Isolate and plot times where full profile exists......
profiles_to_process = topobathy_shift_plot
# tkeep = np.sum(~np.isnan(profiles_to_process),axis=1 ) == nx
ikeep = np.where(np.sum(~np.isnan(profiles_to_process),axis=0) == nx)[0]
shorelines = profiles_to_process[:,ikeep]