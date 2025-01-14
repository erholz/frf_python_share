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
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_12Jan2025/'
# with open(picklefile_dir+'topobathy_scale&shift.pickle','rb') as file:
#    topobathy_shift_plot,topobathy_scale_plot = pickle.load(file)
with open(picklefile_dir+'topobathy_reshape_indexKeeper.pickle','rb') as file:
    tt_unique,origin_set,dataset_index_fullspan,dataset_index_plot = pickle.load(file)
# with open(picklefile_dir+'topobathy_reshapeToNXbyNumUmiqueT.pickle','rb') as file:
#     _,_,_,_,_,_ = pickle.load(file)
with open(picklefile_dir+'topobathy_scale&shift_Zdunetoe_3p2m.pickle','rb') as file:
    topobathy_shift_plot,_ = pickle.load(file)
# with open(picklefile_dir + 'topobathy_scale&shift_ZMHW_0p36m.pickle', 'rb') as file:
#     topobathy_shift_plot, _ = pickle.load(file)


# DEFINE DATASET FOR PCA
topobathy_check = np.empty(topobathy_shift_plot.shape)
topobathy_check[:] = topobathy_shift_plot[:]
nx = topobathy_check.shape[0]
nt = topobathy_check.shape[1]
dx = 0.1
xplot = dx*np.arange(nx)
# Lmin = 50
Lmin = 75
check_data = topobathy_check[xplot <= Lmin,:]
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
check_data_filled[:] = topobathy_check_xshoreFill[xplot <= Lmin,:]
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
# with open(picklefile_dir+'topobathy_finalCheckBeforePCA_Zdunetoe_3p2m.pickle','wb') as file:
#     pickle.dump([topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck], file)
# with open(picklefile_dir+'topobathy_finalCheckBeforePCA_Zdunetoe_3p2m.pickle','rb') as file:
#     topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)
# with open(picklefile_dir+'topobathy_finalCheckBeforePCA_ZMHW_0p36m.pickle','wb') as file:
#     pickle.dump([topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck], file)

############################# MAKE NICE PLOTS OF DATA BEFORE PCA #############################

ZprePCA = topobathy_check_xshoreFill[:,iirow_finalcheck]

yplot1 = np.sum(~np.isnan(topobathy_check_xshoreFill),axis=1)#/tt_unique.size
yplot2 = np.sum(~np.isnan(ZprePCA),axis=1)
dx = 0.1
xplot = dx*np.arange(yplot2.size)
fig, ax = plt.subplots()
ax.plot(xplot,yplot1,'b',label='6) Shift origin to x*=0, then cross-time interp')
ax.plot(xplot,yplot2,'r--',label='7) Final selection for ML')
ax.set_ylim(0,43000)
ax.set_xlim(0,75)
ax.legend()
plt.grid()
ax.set_ylabel('Num. unique profiles')
ax.set_xlabel('x* [m]')

# get contours for plotting
mlw = -0.62
mwl = -0.13
zero = 0
mhw = 0.36
dune_toe = 3.22
cont_elev = np.array([mlw,mwl,mhw,dune_toe]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size ))


fig, ax = plt.subplots()
ax.plot(xplot,ZprePCA,color='0.5',linewidth=0.5,alpha=0.1)
profmean = np.nanmean(ZprePCA,axis=1)
profstd = np.nanstd(ZprePCA,axis=1)
ax.plot(xplot,profmean,'k')
ax.plot(xplot,profmean+profstd,'k:')
ax.plot(xplot,profmean-profstd,'k:')
plt.grid()
ax.set_xlim(0,75)
ax.set_ylabel('z [m]')
ax.set_xlabel('x* [m]')
ax.plot(xplot,cont_elev[0]+np.zeros(shape=xplot.shape),color=cmap[0, :],label='MLW')
ax.plot(xplot,cont_elev[1]+np.zeros(shape=xplot.shape),color=cmap[1, :],label='MWL')
ax.plot(xplot,cont_elev[2]+np.zeros(shape=xplot.shape),color=cmap[2, :],label='MHW')
ax.plot(xplot,cont_elev[3]+np.zeros(shape=xplot.shape),color=cmap[3, :],label='Dune toe')
ax.legend()



############################# NORMALIZE PROFILES FOR PCA #############################

# with open(picklefile_dir+'topobathy_finalCheckBeforePCA_Zdunetoe_3p2m.pickle','rb') as file:
#     topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)
# with open(picklefile_dir+'topobathy_finalCheckBeforePCA_ZMHW_0p36m.pickle','rb') as file:
#     topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)

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
ax.plot(xplot,dataNorm,linewidth=0.5,alpha=0.5)
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

fig, ax = plt.subplots()
xplot = np.arange(1,21)
ax.plot(xplot,APEV[0:20],'o')
plt.grid()
ax.plot([0,25],[95,95],'k')
ax.set_ylabel('cum. variance explained')
ax.set_xlabel('EOF')
ax.set_xlim(0,20)

fig, ax = plt.subplots(2,4)
time_PCA = tt_unique[iirow_finalcheck]
tplot = pd.to_datetime(time_PCA, unit='s', origin='unix')
nx = dataNorm.shape[0]
dx = 0.1
ccsize = 1
xplot = dx*np.arange(nx)
ax[0,0].scatter(tplot,PCs[:,0],ccsize)
ax[0,0].set_ylim(-75,130)
ax[0,0].set_title('Mode 1'+'\n Total Var. = '+str(round(APEV[0],1))+'%')
ax[1,0].plot(xplot,EOFs[0,:])
ax[1,0].set_ylim(-0.12,0.12)
ax[0,1].scatter(tplot,PCs[:,1],ccsize)
ax[0,1].set_ylim(-75,130)
ax[0,1].set_title('Mode 2'+'\n Total Var. = '+str(round(APEV[1],1))+'%')
ax[1,1].plot(xplot,EOFs[1,:])
ax[1,1].set_ylim(-0.12,0.12)
ax[0,2].scatter(tplot,PCs[:,2],ccsize)
ax[0,2].set_ylim(-75,130)
ax[0,2].set_title('Mode 3'+'\n Total Var. = '+str(round(APEV[2],1))+'%')
ax[1,2].plot(xplot,EOFs[2,:])
ax[1,2].set_ylim(-0.12,0.12)
ax[0,3].scatter(tplot,PCs[:,3],ccsize)
ax[0,3].set_ylim(-75,130)
ax[0,3].set_title('Mode 4'+'\n Total Var. = '+str(round(APEV[3],1))+'%')
ax[1,3].plot(xplot,EOFs[3,:])
ax[1,3].set_ylim(-0.12,0.12)

# Find contour position of input profiles to add to plot...
mwl = -0.13
zero = 0
mhw = 3.6
dune_toe = 3.22
cont_elev = np.array([mwl,mhw]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(data.T,time_PCA,xplot,cont_elev)
cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size ))
# for cc in np.arange(cont_elev.size):
#     ax.plot([0, 0] + cmean[cc], [0, 9999999999], label='z = ' + str(cont_elev[cc]) + ' m', color=cmap[cc, :])#, label='X_{c,MWL}')
# ax.plot([0, 0] + cmean[0], [0, 9999999999], color=cmap[0, :], label='$X_{c,MWL}$')
# ax.plot([0, 0] + cmean[1], [0, 9999999999], color=cmap[1, :], label='$X_{c,MHW}$')
for cc in np.arange(cont_elev.size):
    left, bottom, width, height = (cmean[cc] - cstd[cc], 0, cstd[cc] * 2, 9999999999)
    patch = plt.Rectangle((left, bottom), width, height, alpha=0.1, color=cmap[cc, :])
    ax.add_patch(patch)


# Can we re-create the profiles from the PCA?
reconstruct_profileNorm = np.empty(shape=dataNorm.shape)
reconstruct_profileNorm[:] = np.nan
for tt in np.arange(tplot.size):
    mode1 = EOFs[0,:]*PCs[tt,0]
    mode2 = EOFs[1,:]*PCs[tt,1]
    mode3 = EOFs[2, :] * PCs[tt, 2]
    mode4 = EOFs[3, :] * PCs[tt, 3]
    prof_tt = mode1 + mode2 + mode3 + mode4
    # prof_tt = mode1 + mode2
    reconstruct_profileNorm[:,tt] = prof_tt
reconstruct_profileT = reconstruct_profileNorm.T*dataStd.T + dataMean.T
reconstruct_profile = reconstruct_profileT.T

fig, ax = plt.subplots()
xplot = dx*np.arange(nx)
ax.plot(xplot,reconstruct_profileNorm,linewidth=0.5,alpha=0.5)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.set_title('Normalized profiles reconstructed from PCA')
fig, ax = plt.subplots()
xplot = dx*np.arange(nx)
ax.plot(xplot,reconstruct_profile,linewidth=0.5,alpha=0.5)
ax.plot(xplot,dataMean,'k')
ax.plot(xplot,dataMean+dataStd,'k--')
ax.plot(xplot,dataMean-dataStd,'k--')
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.set_title('Profiles reconstructed from PCA')

#
# with open(picklefile_dir+'topobathy_PCA_ZMHW_0p36m_Lmin_50m.pickle','wb') as file:
#     pickle.dump([dataNorm,dataMean,dataStd,PCs,EOFs,APEV,reconstruct_profileNorm,reconstruct_profile], file)
# with open(picklefile_dir+'topobathy_PCA_Zdunetoe_3p2m_Lmin_75.pickle','wb') as file:
#     pickle.dump([dataNorm,dataMean,dataStd,PCs,EOFs,APEV,reconstruct_profileNorm,reconstruct_profile], file)
# with open(picklefile_dir+'topobathy_PCA_ZMHW_0p36m_Lmin_50m.pickle','rb') as file:
#     dataNorm,dataMean,dataStd,PCs,EOFs,APEV,reconstruct_profileNorm,reconstruct_profile = pickle.load(file)




############################# Conservation of Mass --> Loss function? #############################
### For each dataset, calculate the change in volume (dVol) between time steps for OBS+ data and compare
###  with DV between time steps for the PCA-reconstructed data...


# Load dataset we are going to compare with...
# with open(picklefile_dir+'topobathy_finalCheckBeforePCA_Zdunetoe_3p2m.pickle','rb') as file:
#    topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)
with open(picklefile_dir+'topobathy_finalCheckBeforePCA_ZMHW_0p36m.pickle','rb') as file:
    topobathy_check_xshoreFill,dataset_passFinalCheck,iiDS_passFinalCheck,iirow_finalcheck = pickle.load(file)
with open(picklefile_dir+'topobathy_reshapeToNXbyNumUmiqueT.pickle','rb') as file:
    tt_unique,_,_,topobathy_xshoreInterp_plot,topobathy_extension_plot,topobathy_xshoreInterpX2_plot = pickle.load(file)

# First, isolate the data that ultimately goes into the PCA
num_datasets = iiDS_passFinalCheck.size
profiles_to_process = np.empty(shape=topobathy_check_xshoreFill.shape)
profiles_to_process[:] = topobathy_check_xshoreFill
iikeep = iirow_finalcheck
# iikeep = rows_nonans
# data = profiles_to_process[:,iikeep]
data = profiles_to_process[:,:]
dataset_profileIndeces = dataset_index_plot[dataset_passFinalCheck == 1,:]

# then go through each dataset and pull the profiles that correspond
dx = 0.1
num_profs_inset = dataset_profileIndeces.shape[1]
Vol_obsdata = np.empty((num_profs_inset,num_datasets))
dVol_obsdata = np.empty((num_profs_inset-1,num_datasets))
for nn in np.arange(num_datasets):
    iiprof_in_dataset = dataset_profileIndeces[nn,:].astype(int)
    prof_in_dataset = data[:,iiprof_in_dataset]
    Vol_setnn = np.empty(num_profs_inset,)
    for tt in np.arange(num_profs_inset):
        Vol_setnn[tt] = np.nansum(prof_in_dataset[:,tt]*dx)
    dVol_setnn = Vol_setnn[1:] - Vol_setnn[0:-1]
    Vol_obsdata[:,nn] = Vol_setnn
    dVol_obsdata[:,nn] = dVol_setnn
    # find where dVol is very high
    dVol_thresh = 5
    if sum(dVol_setnn > dVol_thresh) > 0:
        flag_prof_dVol_setnn = iiprof_in_dataset[np.where(dVol_setnn > dVol_thresh)]
        for jj in np.arange(flag_prof_dVol_setnn.size):
            fig, ax = plt.subplots()
            ax.plot(data[:,flag_prof_dVol_setnn[jj]])
            ax.plot(data[:, flag_prof_dVol_setnn[jj]+1])
            ax.plot(topobathy_xshoreInterp_plot[:,flag_prof_dVol_setnn[jj]])
            ax.plot(topobathy_xshoreInterp_plot[:, flag_prof_dVol_setnn[jj]+1])





fig, ax = plt.subplots()
# ax.plot(dVol_obsdata,'.')
plt.hist(np.resize(dVol_obsdata,(dVol_obsdata.size,)),bins=np.arange(-60,60,5))

# do the same for the pca_reconstructed profiles...
dx = 0.1
# remake the PCA_reconstruct array so that it is the same size as "data" above
PCAprofiles_sizedata = np.empty(shape=data.shape)
PCAprofiles_sizedata[:] = np.nan
PCAprofiles_sizedata[:,iikeep] = reconstruct_profile
Vol_pcaRecon = np.empty((num_profs_inset,num_datasets))
dVol_pcaRecon = np.empty((num_profs_inset-1,num_datasets))
for nn in np.arange(num_datasets):
    iiprof_in_dataset = dataset_profileIndeces[nn,:].astype(int)
    prof_in_dataset = PCAprofiles_sizedata[:,iiprof_in_dataset]
    Vol_setnn = np.empty(num_profs_inset,)
    for tt in np.arange(num_profs_inset):
        Vol_setnn[tt] = np.nansum(prof_in_dataset[:,tt]*dx)
    dVol_setnn = Vol_setnn[1:] - Vol_setnn[0:-1]
    Vol_pcaRecon[:,nn] = Vol_setnn
    dVol_pcaRecon[:,nn] = dVol_setnn
fig, ax = plt.subplots()
# ax.plot(dVol_obsdata,'.')
dVol_obsdata_plot = np.resize(dVol_obsdata,(dVol_obsdata.size,))
dVol_obsdata_mean = np.mean(dVol_obsdata_plot)
dVol_obsdata_std = np.std(dVol_obsdata_plot)
dVol_pcaRecon_plot = np.resize(dVol_pcaRecon,(dVol_pcaRecon.size,))
dVol_pcaRecon_mean = np.mean(dVol_pcaRecon_plot)
dVol_pcaRecon_std = np.std(dVol_pcaRecon_plot)
plt.hist(dVol_obsdata_plot,density=True,bins=np.arange(-60,60,.1),alpha=0.5,label='observed, PCA input')
plt.hist(dVol_pcaRecon_plot,density=True,bins=np.arange(-60,60,.1),alpha=0.5,label='constructed from PCs')
ax.plot([0,0]+dVol_obsdata_mean,[0, 0.65],'c')
ax.plot([0,0]+dVol_obsdata_mean+dVol_obsdata_mean,[0, 0.65],'c--')
ax.plot([0,0]+dVol_obsdata_mean-dVol_obsdata_mean,[0, 0.65],'c--')
ax.plot([0,0]+dVol_obsdata_mean+2*dVol_obsdata_mean,[0, 0.65],'c:')
ax.plot([0,0]+dVol_obsdata_mean-2*dVol_obsdata_mean,[0, 0.65],'c:')
ax.plot([0,0]+dVol_pcaRecon_mean,[0, 0.65],'m')
ax.plot([0,0]+dVol_pcaRecon_mean+dVol_pcaRecon_std,[0, 0.65],'m--')
ax.plot([0,0]+dVol_pcaRecon_mean-dVol_pcaRecon_std,[0, 0.65],'m--')
ax.plot([0,0]+dVol_pcaRecon_mean+2*dVol_pcaRecon_std,[0, 0.65],'m:')
ax.plot([0,0]+dVol_pcaRecon_mean-2*dVol_pcaRecon_std,[0, 0.65],'m:')
ax.set_xlabel('dVol [m^3/m]')
ax.set_ylabel('pdf [-]')
ax.legend()
ax.set_xlim(-15,15)
ax.set_ylim(0, 0.65)
fig, ax = plt.subplots()
xplot = np.resize(dVol_obsdata,(dVol_obsdata.size,))
yplot = np.resize(dVol_pcaRecon,(dVol_pcaRecon.size,))
ax.plot(xplot,yplot,'.')
fig, ax = plt.subplots()
ax.plot(xplot,xplot-yplot,'.',alpha=0.01)
plt.grid()
ax.set_ylabel('Error = dVol_obs - dVol_PCA [m^3/m]')
ax.set_xlabel('dVol_obs [m^3/m]')
ax.set_ylim(-0.5,0.5)


