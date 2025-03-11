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



############################# LOAD DATA #############################


picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96.pickle', 'rb') as file:
#     time_fullspan,lidar_xFRF,profileIDs_ML_final,profileTimes_ML_final,hydro_MLinput_final,topobathy_ML_final = pickle.load(file)
unique_profIDs = np.unique(profileIDs_ML_final).astype(int)

# Use same length constraints as before
Lmin = 75
Xstart = 50
Xend = 190
iistart = np.where(abs(lidar_xFRF-Xstart) == np.nanmin(abs(lidar_xFRF-Xstart)))[0]
iiend = np.where(abs(lidar_xFRF-Xend) == np.nanmin(abs(lidar_xFRF-Xend)))[0]
Ztrimlength = topobathy_ML_final[np.arange(iistart,iiend),:]
xplot = lidar_xFRF[np.arange(iistart,iiend)]

Zcheck = Ztrimlength[:,unique_profIDs]
yplot = np.sum(np.isnan(Zcheck),axis=0)
tmp = topobathy_ML_final[np.arange(iistart,iiend),:]
badprof = unique_profIDs[yplot > 0]
plotcheck = tmp[:,unique_profIDs[yplot > 0]]
fig, ax = plt.subplots()
ax.plot(xplot,plotcheck)
badset = np.empty((0,)).astype(int)
for jj in np.arange(badprof.size):
    sets_containing_profjj = np.where(np.nansum(np.isin(profileIDs_ML_final,badprof[jj]),axis=1))[0]
    new_sets = sets_containing_profjj[~np.isin(sets_containing_profjj,badset)]
    badset = np.append(badset,new_sets)

setid = np.arange(profileIDs_ML_final.shape[0])
goodset = setid[~np.isin(setid,badset)]
goodprofs = np.unique(profileIDs_ML_final[goodset,:]).astype(int)
goodtimes = np.unique(profileTimes_ML_final[goodset,:]).astype(int)
ZprePCA = Ztrimlength[:,goodprofs]

# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PrePCA.pickle', 'wb') as file:
#     pickle.dump([xplot,ZprePCA,goodset,goodprofs,goodtimes],file)



############################# MAKE NICE PLOTS OF DATA BEFORE PCA #############################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96.pickle', 'rb') as file:
#     time_fullspan,lidar_xFRF,profileIDs_ML_final,profileTimes_ML_final,hydro_MLinput_final,topobathy_ML_final = pickle.load(file)
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PrePCA.pickle', 'rb') as file:
#     xplot,ZprePCA,goodset,goodprofs,goodtimes = pickle.load(file)

yplot2 = np.sum(~np.isnan(ZprePCA),axis=1)
dx = 0.1
# xplot = dx*np.arange(yplot2.size)
fig, ax = plt.subplots()
ax.plot(xplot,yplot2,'r.',label='7) Final selection for ML')
# ax.set_ylim(0,43000)
# ax.set_xlim(0,75)
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
ax.set_xlim(xplot[0],xplot[-1])
ax.set_ylabel('z [m]')
ax.set_xlabel('x* [m]')
ax.plot(xplot,cont_elev[0]+np.zeros(shape=xplot.shape),color=cmap[0, :],label='MLW')
ax.plot(xplot,cont_elev[1]+np.zeros(shape=xplot.shape),color=cmap[1, :],label='MWL')
ax.plot(xplot,cont_elev[2]+np.zeros(shape=xplot.shape),color=cmap[2, :],label='MHW')
ax.plot(xplot,cont_elev[3]+np.zeros(shape=xplot.shape),color=cmap[3, :],label='Dune toe')
ax.legend()



############################# NORMALIZE PROFILES FOR PCA #############################

# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PrePCA.pickle', 'rb') as file:
#     xplot,ZprePCA,goodset,goodprofs,goodtimes = pickle.load(file)

profiles_to_process = np.empty(shape=ZprePCA.shape)
profiles_to_process[:] = ZprePCA
rows_nonans = np.where(np.nansum(~np.isnan(profiles_to_process),axis=0 ) == profiles_to_process.shape[0])[0]

# Using rows_nonans would be ALL the rows where the data is available, but the ML_datasets will only see rows/profiles iirow_finalcheck
# iikeep = iirow_finalcheck
iikeep = rows_nonans
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
xplot = np.arange(1,21).astype(int)
ax.plot(xplot,APEV[0:20])
ax.bar(xplot,APEV[0:20])
# plt.grid()
ax.plot([0,25],[95,95],'k')
ax.set_ylabel('cumulative variance')
ax.set_xlabel('EOF Mode')
ax.set_xticks(np.arange(21).astype(int))
ax.set_xlim(0.5,10.5)
ax.set_ylim(0,100)

fig, ax = plt.subplots(2,4)
time_PCA = goodtimes
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

fig, ax = plt.subplots(2,1)
ax[0].scatter(tplot,PCs[:,0],ccsize,marker='o',label='Mode 1')
ax[0].scatter(tplot,PCs[:,1],ccsize,marker='o',label='Mode 2')
ax[0].scatter(tplot,PCs[:,2],ccsize,marker='o',label='Mode 3')
ax[0].scatter(tplot,PCs[:,3],ccsize,marker='o',label='Mode 4')
# ax[0].set_xlabel('time')
ax[0].set_ylabel('amplitude')
ax[0].grid(axis="both")
ax[0].legend()
ax[1].plot(xplot,EOFs[0,:],label='Mode 1')
ax[1].plot(xplot,EOFs[1,:],label='Mode 2')
ax[1].plot(xplot,EOFs[2,:],label='Mode 3')
ax[1].plot(xplot,EOFs[3,:],label='Mode 4')
ax[1].set_xlabel('x* [m]')
ax[1].set_ylabel('EOF')
ax[1].grid(axis="both")
ax[1].set_xlim(min(xplot),max(xplot))
ax[1].legend()

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


# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PCA.pickle', 'wb') as file:
#     pickle.dump([dataNorm,dataMean,dataStd,PCs,EOFs,APEV,reconstruct_profileNorm,reconstruct_profile],file)




############################# Conservation of Mass --> Loss function? #############################
### For each dataset, calculate the change in volume (dVol) between time steps for OBS+ data and compare
###  with DV between time steps for the PCA-reconstructed data...


# Load dataset we are going to compare with...
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96.pickle', 'rb') as file:
#     time_fullspan,lidar_xFRF,profileIDs_ML_final,profileTimes_ML_final,hydro_MLinput_final,topobathy_ML_final = pickle.load(file)
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PrePCA.pickle', 'rb') as file:
#     xplot,ZprePCA,goodset,goodprofs,goodtimes = pickle.load(file)
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PCA.pickle', 'rb') as file:
#     dataNorm,dataMean,dataStd,PCs,EOFs,APEV,reconstruct_profileNorm,reconstruct_profile = pickle.load(file)

# First, isolate the data that ultimately goes into the PCA
num_datasets = goodset.size
profiles_to_process = np.empty(shape=topobathy_ML_final.shape)
profiles_to_process[:] = topobathy_ML_final
# Use same length constraints as before
Lmin = 75
Xstart = 50
Xend = 190
iistart = np.where(abs(lidar_xFRF-Xstart) == np.nanmin(abs(lidar_xFRF-Xstart)))[0]
iiend = np.where(abs(lidar_xFRF-Xend) == np.nanmin(abs(lidar_xFRF-Xend)))[0]
Ztrimlength = topobathy_ML_final[np.arange(iistart,iiend),:]
data = Ztrimlength[:]
# data = profiles_to_process[:,:]
dataset_profileIndeces = profileIDs_ML_final[goodset,:]

# then go through each dataset and pull the profiles that correspond
dx = 0.1
num_profs_inset = dataset_profileIndeces.shape[1]
Vol_obsdata = np.empty((num_profs_inset,num_datasets))
dVol_obsdata = np.empty((num_profs_inset-1,num_datasets))
numinset_dVolGTthresh = np.empty((num_datasets,))
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
    numinset_dVolGTthresh[nn] = np.sum(abs(dVol_setnn) > dVol_thresh)
    # if sum(dVol_setnn > dVol_thresh) > 0:
    #     flag_prof_dVol_setnn = iiprof_in_dataset[np.where(dVol_setnn > dVol_thresh)]
    #     for jj in np.arange(flag_prof_dVol_setnn.size):
    #         fig, ax = plt.subplots()
    #         ax.plot(data[:,flag_prof_dVol_setnn[jj]])
    #         ax.plot(data[:, flag_prof_dVol_setnn[jj]+1])
    #         ax.plot(topobathy_xshoreInterpX2_plot[:,flag_prof_dVol_setnn[jj]])
    #         ax.plot(topobathy_xshoreInterpX2_plot[:, flag_prof_dVol_setnn[jj]+1])
fig, ax = plt.subplots()
plt.hist(numinset_dVolGTthresh,bins=25)
ii_dVolThreshMet = (numinset_dVolGTthresh <= 1)



fig, ax = plt.subplots()
# ax.plot(dVol_obsdata,'.')
plt.hist(np.resize(dVol_obsdata,(dVol_obsdata.size,)),bins=np.arange(-10,10,0.1))

# do the same for the pca_reconstructed profiles...
dx = 0.1
# remake the PCA_reconstruct array so that it is the same size as "data" above
PCAprofiles_sizedata = np.empty(shape=data.shape)
PCAprofiles_sizedata[:] = np.nan
iikeep = goodprofs
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
dVol_obsdata_plot = np.resize(dVol_obsdata,(dVol_obsdata.size,))
dVol_obsdata_mean = np.mean(dVol_obsdata_plot)
dVol_obsdata_std = np.std(dVol_obsdata_plot)
dVol_pcaRecon_plot = np.resize(dVol_pcaRecon,(dVol_pcaRecon.size,))
dVol_pcaRecon_mean = np.mean(dVol_pcaRecon_plot)
dVol_pcaRecon_std = np.std(dVol_pcaRecon_plot)
fig, ax = plt.subplots()
plt.hist(dVol_obsdata_plot,density=True,bins=np.arange(-5.5,5.5,.1),alpha=0.5,label='observed, PCA input')
plt.hist(dVol_pcaRecon_plot,density=True,bins=np.arange(-5.5,5.5,.1),alpha=0.5,label='constructed from PCs')
ax.plot([0,0]+dVol_obsdata_mean,[0, 100],'c')
ax.plot([0,0]+dVol_obsdata_mean+dVol_obsdata_mean,[0, 100],'c--')
ax.plot([0,0]+dVol_obsdata_mean-dVol_obsdata_mean,[0, 100],'c--')
# ax.plot([0,0]+dVol_obsdata_mean+2*dVol_obsdata_mean,[0, 100],'c:')
# ax.plot([0,0]+dVol_obsdata_mean-2*dVol_obsdata_mean,[0, 100],'c:')
ax.plot([0,0]+dVol_pcaRecon_mean,[0, 100],'m')
ax.plot([0,0]+dVol_pcaRecon_mean+dVol_pcaRecon_std,[0,100],'m--')
ax.plot([0,0]+dVol_pcaRecon_mean-dVol_pcaRecon_std,[0, 100],'m--')
# ax.plot([0,0]+dVol_pcaRecon_mean+2*dVol_pcaRecon_std,[0, 100],'m:')
# ax.plot([0,0]+dVol_pcaRecon_mean-2*dVol_pcaRecon_std,[0, 100],'m:')
ax.set_xlabel('dVol [m^3/m]')
ax.set_ylabel('pdf [-]')
ax.legend()
# ax.set_xlim(-15,15)
ax.set_ylim(0, 2)
fig, ax = plt.subplots()
xplot = np.resize(dVol_obsdata,(dVol_obsdata.size,))
yplot = np.resize(dVol_pcaRecon,(dVol_pcaRecon.size,))
ax.plot(xplot,yplot,'.',alpha=0.01)
plt.grid()
fig, ax = plt.subplots()
ax.plot(xplot,xplot-yplot,'.',alpha=0.01)
plt.grid()
ax.set_ylabel('Error = dVol_obs - dVol_PCA [m^3/m]')
ax.set_xlabel('dVol_obs [m^3/m]')
ax.set_ylim(-0.5,0.5)



################ RERUN PCA WITH DATASETS WHERE DVOL THRESHOLD MET... ################

# verify that all the profiles in topobathy_check_xshoreFill for corresponding datasets are NOTNAN
iiDS_passDVolCheck = goodset[ii_dVolThreshMet]
iirow_dVolCheck = np.unique(dataset_profileIndeces[np.where(ii_dVolThreshMet)[0],:]).astype(int)
data_profIDs_dVolThreshMet = dataset_profileIndeces[np.where(ii_dVolThreshMet)[0],:].astype(int)
data_hydro = hydro_MLinput_final[iiDS_passDVolCheck,:,:]


ZZ = Ztrimlength[:,iirow_dVolCheck]
ZprePCA = Ztrimlength[:,iirow_dVolCheck]
cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size ))
xplot = lidar_xFRF[np.arange(iistart,iiend)]
profmean = np.nanmean(ZprePCA,axis=1)
profstd = np.nanstd(ZprePCA,axis=1)
fig, ax = plt.subplots()
ax.plot(xplot,ZprePCA,color='0.5',linewidth=0.5,alpha=0.1)
ax.plot(xplot,profmean,'k')
ax.plot(xplot,profmean+profstd,'k:')
ax.plot(xplot,profmean-profstd,'k:')
plt.grid()
ax.set_xlim(min(xplot),max(xplot))
ax.set_ylabel('z [m]')
ax.set_xlabel('x* [m]')
ax.plot(xplot,cont_elev[0]+np.zeros(shape=xplot.shape),color=cmap[0, :],label='MLW')
ax.plot(xplot,cont_elev[1]+np.zeros(shape=xplot.shape),color=cmap[1, :],label='MWL')
ax.plot(xplot,cont_elev[2]+np.zeros(shape=xplot.shape),color=cmap[2, :],label='MHW')
ax.plot(xplot,cont_elev[3]+np.zeros(shape=xplot.shape),color=cmap[3, :],label='Dune toe')
ax.legend()


# NORMALIZE PRE-PCA
data = ZprePCA[:]
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
ax.set_xlabel('x* [m]')
ax.set_ylabel('z* [-]')
ax.set_title('Normalized profiles input to PCA')


# RUN PCA
ipca = PCA(n_components=min(dataNorm.shape[0], dataNorm.shape[1]))
PCs = ipca.fit_transform(dataNorm.T)  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

fig, ax = plt.subplots()
xplot = np.arange(1,21).astype(int)
ax.plot(xplot,APEV[0:20])
ax.bar(xplot,APEV[0:20])
# plt.grid()
ax.plot([0,25],[95,95],'k')
ax.set_ylabel('cumulative variance')
ax.set_xlabel('EOF Mode')
ax.set_xticks(np.arange(21).astype(int))
ax.set_xlim(0.5,10.5)
ax.set_ylim(0,100)

fig, ax = plt.subplots(2,4)
time_PCA = time_fullspan[iirow_dVolCheck]
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

fig, ax = plt.subplots(2,1)
ax[0].scatter(tplot,PCs[:,0],ccsize,marker='o',label='Mode 1')
ax[0].scatter(tplot,PCs[:,1],ccsize,marker='o',label='Mode 2')
ax[0].scatter(tplot,PCs[:,2],ccsize,marker='o',label='Mode 3')
ax[0].scatter(tplot,PCs[:,3],ccsize,marker='o',label='Mode 4')
ax[0].scatter(tplot,PCs[:,4],ccsize,marker='o',label='Mode 5')
# ax[0].set_xlabel('time')
ax[0].set_ylabel('amplitude')
ax[0].grid(axis="both")
ax[0].legend()
ax[1].plot(xplot,EOFs[0,:],label='Mode 1')
ax[1].plot(xplot,EOFs[1,:],label='Mode 2')
ax[1].plot(xplot,EOFs[2,:],label='Mode 3')
ax[1].plot(xplot,EOFs[3,:],label='Mode 4')
ax[1].plot(xplot,EOFs[4,:],label='Mode 5')
ax[1].set_xlabel('x* [m]')
ax[1].set_ylabel('EOF')
ax[1].grid(axis="both")
ax[1].set_xlim(min(xplot),max(xplot))
ax[1].legend()

fig, ax = plt.subplots()
ax.plot(xplot,EOFs[0,:],label='Mode 1')
ax.plot(xplot,EOFs[1,:],label='Mode 2')
ax.plot(xplot,EOFs[2,:],label='Mode 3')
ax.plot(xplot,EOFs[3,:],label='Mode 4')
ax.plot(xplot,EOFs[4,:],label='Mode 5')
ax.set_xlabel('x* [m]')
ax.set_ylabel('EOF')
ax.grid(axis="both")
ax.set_xlim(min(xplot),max(xplot))
ax.legend()
fig, ax = plt.subplots(1,5)
ax[0].hist(PCs[:,0],bins=30,color='C0')
ax[1].hist(PCs[:,1],bins=30,color='C1')
ax[2].hist(PCs[:,2],bins=30,color='C2')
ax[3].hist(PCs[:,3],bins=30,color='C3')
ax[4].hist(PCs[:,4],bins=30,color='C4')
ax[0].set_ylim(0,650)
ax[1].set_ylim(0,650)
ax[2].set_ylim(0,650)
ax[3].set_ylim(0,650)
ax[4].set_ylim(0,650)
ax[0].set_xlim(-65,65)
ax[1].set_xlim(-65,65)
ax[2].set_xlim(-30,30)
ax[3].set_xlim(-30,30)
ax[4].set_xlim(-30,30)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[3].set_yticklabels([])
ax[4].set_yticklabels([])



## plot difference between PCA-reconstructed and observed profiles
reconstruct_profileNorm = np.empty(shape=dataNorm.shape)
reconstruct_profileNorm[:] = np.nan
for tt in np.arange(tplot.size):
    mode1 = EOFs[0,:]*PCs[tt,0]
    mode2 = EOFs[1,:]*PCs[tt,1]
    mode3 = EOFs[2, :] * PCs[tt, 2]
    mode4 = EOFs[3, :] * PCs[tt, 3]
    mode5 = EOFs[4, :] * PCs[tt, 4]
    prof_tt = mode1 + mode2 + mode3 + mode4 + mode5
    # prof_tt = mode1 + mode2
    reconstruct_profileNorm[:,tt] = prof_tt
reconstruct_profileT = reconstruct_profileNorm.T*dataStd.T + dataMean.T
reconstruct_profile = reconstruct_profileT.T
fig, ax = plt.subplots()
elev_plot = data-reconstruct_profile
ax.plot(xplot,elev_plot)
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(elev_plot, elev_plot.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=1, c=zz, cmap='RdBu', vmin=-1,vmax=1)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')


# make PCs_fullspan, etc.
PCs_fullspan = np.empty(shape=(time_fullspan.size, xplot.size))*np.nan
dataNorm_fullspan = np.empty(shape=(xplot.size,time_fullspan.size))*np.nan
reconstruct_profNorm_fullspan = np.empty(shape=(xplot.size,time_fullspan.size))*np.nan
reconstruct_prof_fullspan = np.empty(shape=(xplot.size,time_fullspan.size))*np.nan
for jj in np.arange(iiDS_passDVolCheck.size):
    ii_fullspan = profileIDs_ML_final[iiDS_passDVolCheck[jj],:].astype(int)
    ii_PCs = np.where(np.isin(iirow_dVolCheck,ii_fullspan))[0]
    PCs_fullspan[ii_fullspan,:] = PCs[ii_PCs,:]
    dataNorm_fullspan[:,ii_fullspan] = dataNorm[:,ii_PCs]
    reconstruct_profNorm_fullspan[:,ii_fullspan] = reconstruct_profileNorm[:,ii_PCs]
    reconstruct_prof_fullspan[:,ii_fullspan] = reconstruct_profile[:,ii_PCs]

# ## SAVE
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook96_PCApostDVol.pickle', 'wb') as file:
#     pickle.dump([xplot,time_fullspan,dataNorm_fullspan,dataMean,dataStd,PCs_fullspan,EOFs,APEV,data_profIDs_dVolThreshMet,
#                  reconstruct_profNorm_fullspan,reconstruct_prof_fullspan,data_hydro],file)
