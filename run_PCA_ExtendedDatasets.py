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
with open(picklefile_dir+'topobathy_reshape_indexKeeper.pickle','rb') as file:
    tt_unique,origin_set,dataset_index_fullspan,dataset_index_plot = pickle.load(file)
# with open(picklefile_dir+'topobathy_reshapeToNXbyNumUmiqueT.pickle','rb') as file:
#     _,_,_,_,_,_ = pickle.load(file)


# DEFINE DATASET FOR PCA
topobathy_check = np.empty(topobathy_shift_plot.shape)
topobathy_check[:] = topobathy_shift_plot[:]
nx = topobathy_check.shape[0]
nt = topobathy_check.shape[1]
dx = 0.1
xplot = dx*np.arange(nx)
check_data = topobathy_check[xplot < 100,:]
yy = np.nansum(np.isnan(check_data),axis=0 )
# yy = yy[(yy < check_data.shape[0]) & (yy > 0)]
yy = yy[(yy > 0)]
fig, ax = plt.subplots()
plt.hist(yy,bins=np.arange(0,1200,25))


# ok, now find our where in x-shore are the nans are located...
yy = np.nansum(np.isnan(check_data),axis=0 )
iiisnan = np.where(yy > 0)[0]
xcoor_wherenan = np.empty((check_data.shape[0],iiisnan.size))
xcoor_wherenan[:] = np.nan
for jj in np.arange(iiisnan.size):
    xcoor_wherenan[np.isnan(check_data[:,iiisnan[jj]]),jj] = 1
nx = check_data.shape[0]
xplot = dx*np.arange(nx)
yplot = np.nansum(xcoor_wherenan,axis=1)
fig, ax = plt.subplots()
ax.plot(xplot,yplot,'.')

# WHICH datasets are those profiles a part of...
num_datasets = dataset_index_plot.shape[0]

dataset_withnans = np.empty(0)
for nn in np.arange(num_datasets):
    testfor_iiisnan = np.isin(dataset_index_plot[nn,:],iiisnan)
    if sum(testfor_iiisnan) > 0:
        dataset_withnans = np.append(dataset_withnans,nn)
unique_baddatasets = np.unique(dataset_withnans[1:]).astype(int)

# Check if any gaps can be filled?
Nlook = 4*24
yy = np.nansum(np.isnan(check_data),axis=0)
topobathy_check_xshoreFill = np.empty(shape=check_data.shape)
topobathy_check_xshoreFill[:] = check_data[:]
for jj in np.arange(unique_baddatasets.size):
# for jji in np.floor(np.linspace(0,unique_baddatasets.size-1,10)):
    # jj = int(jji)
    iiprof_inset = dataset_index_plot[unique_baddatasets[jj],:].astype(int)
    ZZ = topobathy_check[:nx,iiprof_inset]
    ZZ_fill = np.empty(shape=ZZ.shape)
    ZZ_fill[:] = ZZ[:]
    # go through x-shore locations, interpolate across time if available
    for ii in np.arange(ZZ.shape[0]):
        xshore_slice = ZZ[ii,:]
        percent_avail = sum(~np.isnan(xshore_slice))/Nlook
        if (percent_avail >= 0.66) & (percent_avail < 1.0):
            tin = np.arange(0,Nlook)
            zin = xshore_slice
            tin = tin[~np.isnan(zin)]
            zin = zin[~np.isnan(zin)]
            zout = np.interp(np.arange(0,Nlook),tin,zin)
            # topobathy_xshoreinterpX2[ii,:,jj] = zout
            ZZ_fill[ii,:] = zout
    topobathy_check_xshoreFill[:nx,iiprof_inset] = ZZ_fill
    ## PLOT TO CONFIRM
    # tplot = np.arange(Nlook)
    # xplot = np.arange(nx)
    # XX, TT = np.meshgrid(xplot,tplot)
    # timescatter = np.reshape(TT, TT.size)
    # xscatter = np.reshape(XX, XX.size)
    # zscatter = np.reshape(ZZ_fill.T, ZZ_fill.size)
    # tt = timescatter[~np.isnan(zscatter)]
    # xx = xscatter[~np.isnan(zscatter)]
    # zz = zscatter[~np.isnan(zscatter)]
    # fig, ax = plt.subplots()
    # ph = ax.scatter(xx, tt, s=2, c=zz, cmap='viridis')
    # cbar = fig.colorbar(ph, ax=ax)
    # cbar.set_label('z[m]')
    # max_nan = np.nanmax(yy[iiprof_inset])
    # ax.set_title('Profile elevation - ML_Dataset '+str(unique_baddatasets[jj])+', '+str(max_nan))

# Check to see available now that some filling done....
check_data_filled = np.empty(shape=check_data.shape)
nx = check_data_filled.shape[0]
xplot = dx*np.arange(nx)
check_data_filled[:] = topobathy_check_xshoreFill[xplot < 100,:]
yy = np.nansum(np.isnan(check_data_filled),axis=0 )
iiisnan = np.where(yy > 0)[0]
xcoor_wherenan = np.empty((check_data_filled.shape[0],iiisnan.size))
xcoor_wherenan[:] = np.nan
xcoor_wherenotnan = np.empty((check_data_filled.shape[0],iiisnan.size))
xcoor_wherenotnan[:] = np.nan
for jj in np.arange(iiisnan.size):
    xcoor_wherenan[np.isnan(check_data_filled[:,iiisnan[jj]]),jj] = 1
    xcoor_wherenotnan[~np.isnan(check_data_filled[:, iiisnan[jj]]), jj] = 1
fig, ax = plt.subplots()
ax.plot(xplot,np.nansum(xcoor_wherenan,axis=1),'.')
fig, ax = plt.subplots()
ax.plot(xplot,np.nansum(xcoor_wherenotnan,axis=1),'.')

# WHICH datasets are those profiles a part of...
num_datasets = dataset_index_plot.shape[0]
dataset_withnans = np.empty(0)
for nn in np.arange(num_datasets):
    testfor_iiisnan = np.isin(dataset_index_plot[nn,:],iiisnan)
    if sum(testfor_iiisnan) > 0:
        dataset_withnans = np.append(dataset_withnans,nn)
unique_baddatasets_retest = np.unique(dataset_withnans[1:]).astype(int)

# so, there should be no nans in the datasets that are NOT unique_baddatasets_retest
numnanstotal = np.empty(num_datasets,)
numnanstotal[:] = np.nan
for jj in np.arange(num_datasets):
    if ~np.isin(jj,unique_baddatasets_retest):
        iiprof_inset = dataset_index_plot[jj, :].astype(int)
        ZZ = topobathy_check_xshoreFill[:, iiprof_inset]
        numnanstotal[jj] = np.nansum(np.isnan(ZZ))
fig, ax = plt.subplots()
ax.plot(np.unique(numnanstotal),'.')
# identify WHICH datasets meet no-nan criterion...
dataset_passFinalCheck = np.empty(shape=numnanstotal.shape)
dataset_passFinalCheck[:] = np.nan
dataset_passFinalCheck[numnanstotal == 0] = 1
# verify that all the profiles in topobathy_check_xshoreFill for corresponding datasets are NOTNAN
iiDS_passFinalCheck = np.where(dataset_passFinalCheck == 1)[0]
irow_finalcheck = np.empty(0)
for jj in np.arange(iiDS_passFinalCheck.size):
    irow_finalcheck = np.append(irow_finalcheck,dataset_index_plot[iiDS_passFinalCheck[jj],:])
iirow_finalcheck = np.unique(irow_finalcheck[1:]).astype(int)
ZZ = topobathy_check_xshoreFill[:,iirow_finalcheck]

# with open(picklefile_dir+'topobathy_finalCheckBeforePCA.pickle','wb') as file:
#     pickle.dump([topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck], file)
# with open(picklefile_dir+'topobathy_finalCheckBeforePCA.pickle','rb') as file:
#     topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)



############################# NORMALIZE PROFILES FOR PCA #############################


profiles_to_process = np.empty(shape=topobathy_check_xshoreFill.shape)
profiles_to_process[:] = topobathy_check_xshoreFill
rows_nonans = np.where(np.nansum(~np.isnan(profiles_to_process),axis=0 ) == profiles_to_process.shape[0])[0]

# Using rows_nonans would be ALL the rows where the data is available, but the ML_datasets will only see rows/profiles iirow_finalcheck
iikeep = iirow_finalcheck
# iikeep = rows_nonans
data = profiles_to_process[:,iikeep]
dataMean = np.mean(data,axis=1) # this will give you an average for each cross-shore transect
dataStd = np.std(data,axis=1)
dataNormT = (data.T - dataMean.T) / dataStd.T
dataNorm = dataNormT.T
nx = data.shape[0]
dx = 0.1
fig, ax = plt.subplots()
xplot = dx*np.arange(nx)
ax.plot(xplot,data,linewidth=0.5,alpha=0.5)
ax.plot(xplot,dataMean,'k')
ax.plot(xplot,dataMean+dataStd,'k--')
ax.plot(xplot,dataMean-dataStd,'k--')
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.set_title('Profiles input to PCA')
fig, ax = plt.subplots()
ax.plot(dataNorm,linewidth=0.5,alpha=0.5)
# ax.plot(xplot,dataMean,'k')
# ax.plot(xplot,dataMean+dataStd,'k--')
# ax.plot(xplot,dataMean-dataStd,'k--')
ax.set_xlabel('x* [m]')
ax.set_ylabel('z* [-]')
ax.set_title('Normalized profiles input to PCA')


############################################ PCA ############################################


ipca = PCA(n_components=min(dataNorm.shape[0], dataNorm.shape[1]))
PCs = ipca.fit_transform(dataNorm.T)  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

fig, ax = plt.subplots(2,4)
time_PCA = tt_unique[iirow_finalcheck]
tplot = pd.to_datetime(time_PCA, unit='s', origin='unix')
nx = dataNorm.shape[0]
dx = 0.1
xplot = dx*np.arange(nx)
ax[0,0].scatter(tplot,PCs[:,0],4)
ax[0,0].set_title('Mode 1')
ax[1,0].plot(xplot,EOFs[0,:])
ax[0,1].scatter(tplot,PCs[:,1],4)
ax[0,1].set_title('Mode 2')
ax[1,1].plot(xplot,EOFs[1,:])
ax[0,2].scatter(tplot,PCs[:,2],4)
ax[0,2].set_title('Mode 3')
ax[1,2].plot(xplot,EOFs[2,:])
ax[0,3].scatter(tplot,PCs[:,3],4)
ax[0,3].set_title('Mode 4')
ax[1,3].plot(xplot,EOFs[3,:])