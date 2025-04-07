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



################## LOAD DATA ##################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'

with open(picklefile_dir+'preppedHydroTopobathy.pickle', 'rb') as file:
   lidar_xFRF, time_fullspan, topobathy_fullspan_gapfilled, xplot_shift, topobathy_fullspan_gapfilled_shift, _ = pickle.load(file)

################## ISOLATE PROFILES FOR PCA ##################

topobathy_fullspan = topobathy_fullspan_gapfilled_shift[:]

Lmin = 140
dx = 0.1
proflength_fullspan = np.empty(shape=time_fullspan.shape)*np.nan
for tt in np.arange(time_fullspan.size):
   if sum(~np.isnan(topobathy_fullspan[:,tt])) > 25:
      proflength_fullspan[tt] = xplot_shift[(np.max(np.where(~np.isnan(topobathy_fullspan_gapfilled_shift[:,tt]))[0]))]
ii_minlengthmet = (proflength_fullspan >= Lmin)

numx = int(Lmin/dx)
z_prePCA = topobathy_fullspan[:numx,ii_minlengthmet]


################## NORM FOR PCA ##################

data = z_prePCA[:]
dataMean = np.mean(data,axis=1) # this will give you an average for each cross-shore transect
dataStd = np.std(data,axis=1)
dataNormT = (data.T - dataMean.T) / dataStd.T
dataNorm = dataNormT.T
nx = data.shape[0]
dx = 0.1
# fig, ax = plt.subplots()
# xplot = dx*np.arange(nx)
# ax.plot(xplot,data,linewidth=0.5,alpha=0.5)
# ax.plot(xplot,dataMean,'k')
# ax.plot(xplot,dataMean+dataStd,'k--')
# ax.plot(xplot,dataMean-dataStd,'k--')
# ax.set_xlabel('x* [m]')
# ax.set_ylabel('z [m]')
# ax.set_title('Profiles input to PCA')
# fig, ax = plt.subplots()
# ax.plot(xplot,dataNorm,linewidth=0.5,alpha=0.5)
# ax.set_xlabel('x* [m]')
# ax.set_ylabel('z* [-]')
# ax.set_title('Normalized profiles input to PCA')


################## PCA ##################

ipca = PCA(n_components=min(dataNorm.shape[0], dataNorm.shape[1]))
PCs = ipca.fit_transform(dataNorm.T)  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

PCs_fullspan = np.empty((time_fullspan.size,numx))
PCs_fullspan[ii_minlengthmet,:] = PCs


################## DVOL FOR OBS AND PC-BASED TOPOBATHY ##################

# dVol, observed
vol_obs = np.sum(z_prePCA*dx,axis=0)
vol_obs_fullspan = np.empty(shape=time_fullspan.shape)*np.nan
vol_obs_fullspan[ii_minlengthmet] = vol_obs
dvol_obs_fullspan = vol_obs_fullspan[1:]-vol_obs_fullspan[:-1]

# dvol, PC-derived
reconstruct_profile = np.empty(shape=z_prePCA.shape)
for jj in np.arange(sum(ii_minlengthmet)):
   mode1 = EOFs[0, :] * PCs[jj,0]
   mode2 = EOFs[1, :] * PCs[jj,1]
   mode3 = EOFs[2, :] * PCs[jj,2]
   mode4 = EOFs[3, :] * PCs[jj,3]
   mode5 = EOFs[4, :] * PCs[jj,4]
   mode6 = EOFs[5, :] * PCs[jj,5]
   mode7 = EOFs[6, :] * PCs[jj,6]
   mode8 = EOFs[7, :] * PCs[jj,7]
   profjj_norm = mode1 + mode2 + mode3 + mode4 + mode5 + mode6 + mode7 + mode8
   profjj_T = profjj_norm.T*dataStd.T + dataMean.T
   reconstruct_profile[:,jj] = profjj_T.T
vol_pca = np.sum(reconstruct_profile*dx,axis=0)
vol_pca_fullspan = np.empty(shape=vol_obs_fullspan.shape)*np.nan
vol_pca_fullspan[ii_minlengthmet] = vol_pca
dvol_pca_fullspan = vol_pca_fullspan[1:]-vol_pca_fullspan[:-1]
# fig, ax = plt.subplots()
# ax.plot(dvol_obs_fullspan,'o')
# ax.plot(dvol_pca_fullspan,'.')
fig, ax = plt.subplots()
ax.plot(dvol_obs_fullspan-dvol_pca_fullspan)


################## RERUN PCA ONLY WHERE DVOL THRESH MET ##################

dVol_thresh = 5
ii_dVolthreshmet = abs(dvol_obs_fullspan) < dVol_thresh
ii_volthreshmet = np.append([True],ii_dVolthreshmet)

data = topobathy_fullspan[:numx,ii_minlengthmet & ii_volthreshmet]
dataMean = np.mean(data,axis=1) # this will give you an average for each cross-shore transect
dataStd = np.std(data,axis=1)
dataNormT = (data.T - dataMean.T) / dataStd.T
dataNorm = dataNormT.T
dataNorm_fullspan = np.empty((numx,time_fullspan.size))*np.nan
dataNorm_fullspan[:,ii_minlengthmet & ii_volthreshmet] = dataNorm

ipca = PCA(n_components=min(dataNorm.shape[0], dataNorm.shape[1]))
PCs = ipca.fit_transform(dataNorm.T)  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

PCs_fullspan = np.empty((time_fullspan.size,numx))*np.nan
PCs_fullspan[ii_minlengthmet & ii_volthreshmet,:] = PCs


################## PLOTS ##################

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

time_PCA = time_fullspan[ii_minlengthmet & ii_volthreshmet]
tplot = pd.to_datetime(time_PCA, unit='s', origin='unix')
xplot = np.arange(numx)*dx
ccsize = 3
fig, ax = plt.subplots(2,1)
for jj in np.arange(8):
   ax[0].scatter(tplot,PCs[:,jj],ccsize,marker='o',label='Mode '+str(jj+1))
   ax[1].plot(xplot, EOFs[jj, :], label='Mode '+str(jj+1))
ax[0].set_ylabel('amplitude')
ax[0].grid(axis="both")
ax[0].legend()
ax[1].set_xlabel('x* [m]')
ax[1].set_ylabel('EOF')
ax[1].grid(axis="both")
ax[1].set_xlim(min(xplot),max(xplot))
ax[1].legend()

fig, ax = plt.subplots(1,8)
fig.set_size_inches(15,2.5)
for jj in np.arange(8):
   ax[jj].hist(PCs[:,jj],bins=30,color='C'+str(jj))
   ax[jj].set_ylim(0,6000)
   ax[jj].set_xlim(-80,80)
   if jj > 0:
      ax[jj].set_yticklabels([])


########## SAVE PCA RESULTS ##########

with open(picklefile_dir+'PCAoutput.pickle', 'wb') as file:
   pickle.dump([xplot, time_fullspan, dataNorm_fullspan, dataMean,dataStd,PCs_fullspan,EOFs,APEV],file)

