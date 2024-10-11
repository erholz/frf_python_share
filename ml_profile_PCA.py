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


# Load temporally aligned data - need to add lidarelev_fullspan
picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
full_path = 'C:/Users/rdchlerh/Desktop/FRF_data/dune_lidar/lidar_transect/FRF-geomorphology_elevationTransects_duneLidarTransect_201510.nc'
_, _, _, _, _, lidar_xFRF, lidar_yFRF = (getlocal_lidar(full_path))


# Open all the permutations of Zsmooth & AvgSlope for full dataset
picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
with open(picklefile_dir+'elev&slp_processed_fullspan.pickle', 'rb') as file:
    zsmooth_fullspan, avgslope_fullspan = pickle.load(file)
with open(picklefile_dir+'elev&slp_process+shift_fullspan.pickle', 'rb') as file:
    zsmooth_fullspan_shift, avgslope_fullspan_shift = pickle.load(file)
with open(picklefile_dir + 'elev&slp_process+scale_fullspan.pickle', 'rb') as file:
    zsmooth_fullspan_scale, avgslope_fullspan_scale = pickle.load(file)
with open(picklefile_dir + 'contour_fullspan.pickle', 'rb') as file:
    xc_fullspan,dXcdt_fullspan = pickle.load(file)

# Define mean water level (mwl), mean high water (mhw), and dune toe
mwl = -0.13
mhw = 3.6
dune_toe = 3.22

# DEFINE DATASET FOR PCA
check_data = zsmooth_fullspan_scale
rowsnonans_scaled = np.where(np.sum(np.isnan(check_data),axis=0 ) == 2000)[0]

# Isolate and plot times where full profile exists......
profiles_to_process = zsmooth_fullspan_scale
nx = zsmooth_fullspan_scale.shape[0]
# tkeep = np.sum(~np.isnan(profiles_to_process),axis=1 ) == nx
ikeep = np.where(np.sum(~np.isnan(profiles_to_process),axis=0) == nx)[0]
shorelines = profiles_to_process[:,ikeep]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_fullspan)
# ax.set_ylim(-0.5,2.5)
ax.set_title('Profiles, all')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zsmooth_fullspan[:,ikeep])
# ax.set_ylim(-0.5,2.5)
ax.set_title('Profiles, smoothed')
fig, ax = plt.subplots()
xinterp = np.arange(nx)
ax.plot(xinterp,shorelines)
plt.grid(which='both', axis='both')
# ax.set_ylim(-0.5,2.5)
ax.set_title('Profiles, smoothed & scaled')



# FROM DYLAN
profiles_to_process = zsmooth_fullspan_scale.T
tkeep = np.where(np.sum(~np.isnan(profiles_to_process),axis=1 ) == profiles_to_process.shape[1])[0]
data = profiles_to_process[tkeep,:]
dataMean = np.mean(data,axis=0) # this will give you an average for each cross-shore transect
dataStd = np.std(data,axis=0)
dataNorm = (data[:,:] - dataMean) / dataStd
tmp1 = np.max(dataNorm,axis=1)
tmp2 = np.min(dataNorm,axis=1)
ikeep = np.where((tmp1<3.5)&(tmp2>-3.5))[0]

fig, ax = plt.subplots()
# ax.plot(dataNorm[ikeep,:].T)
xplot = np.linspace(0,1,nx)
ax.plot(xplot,profiles_to_process[:,:].T)
# ax.set_ylim(-5,5)
# ax.set_title('Normalized equi-length profiles')
ax.set_title('Scaled profiles')
ax.set_xlabel('x/b_w [-]')
ax.set_ylabel('z [m, NAVD88]')
ax.plot([0, 1],[mhw, mhw],'k--')
ax.plot([0, 1],[mwl, mwl],'k--')
ax.set_xlim(0,1)



fig, ax = plt.subplots()
ax.plot(dataNorm[ikeep,:].T)
# ax.plot(xinterp,dataNorm[:,:].T)
# ax.set_ylim(-5,5)
ax.set_title('Normalized equi-length profiles')

# principal components analysis
ipca = PCA(n_components=min(dataNorm[ikeep,:].shape[0], dataNorm[ikeep,:].shape[1]))
PCs = ipca.fit_transform(dataNorm[ikeep,:])  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

fig, ax = plt.subplots(2,4)
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
xplot = xinterp
ax[0,0].scatter(tplot[ikeep],PCs[:,0],4)
ax[0,0].set_title('Mode 1')
ax[1,0].plot(xplot,EOFs[0,:])
ax[0,1].scatter(tplot[ikeep],PCs[:,1],4)
ax[0,1].set_title('Mode 2')
ax[1,1].plot(xplot,EOFs[1,:])
ax[0,2].scatter(tplot[ikeep],PCs[:,2],4)
ax[0,2].set_title('Mode 3')
ax[1,2].plot(xplot,EOFs[2,:])
ax[0,3].scatter(tplot[ikeep],PCs[:,3],4)
ax[0,3].set_title('Mode 4')
ax[1,3].plot(xplot,EOFs[3,:])

# PLOT PCAs in x-y space, compare with (active) profile width
profile_width = xc_fullspan[0,:] - xc_fullspan[-1,:]
fig, ax = plt.subplots()
mode1_alltimes = PCs[:,0]
mode2_alltimes = PCs[:,1]
mode3_alltimes = PCs[:,2]
ax.scatter(profile_width[ikeep],mode1_alltimes,4,alpha=0.1)
plt.grid(which='major', axis='both')
fig, ax = plt.subplots(3,1)
ax[0].scatter(tplot[ikeep],mode1_alltimes,4)
ax[1].scatter(tplot[ikeep],mode2_alltimes,4)
ax[2].scatter(tplot[ikeep],mode3_alltimes,4)


# Rescale profiles used for PCA:
scaled_PCAprofiles = data
fig, ax = plt.subplots()
xplot = np.linspace(0,1,nx)
a = profile_width[ikeep].T
b = np.tile(xplot,(a.size,1))
xplot_scaled = a.T*b.T
zplot = np.flip(scaled_PCAprofiles.T)
ax.plot(xplot_scaled,zplot[:,ikeep])
rescaled_PCAprofiles = zplot[:,ikeep]
rescaled_xplot = xplot_scaled
rescaled_width = a
fig, ax = plt.subplots()
ax.hist(rescaled_width)

fig, ax = plt.subplots()
xplot = np.linspace(0,1,nx)
zplot = np.flip(scaled_PCAprofiles.T)
ax.plot(xplot,zplot)

fig, ax = plt.subplots()
xplot = PCs[:,0]
yplot = PCs[:,1]
# cplot = np.arange(xplot.size)
cplot = profile_width[tkeep[ikeep]]
# cmap = plt.cm.rainbow(np.linspace(0, 1, xplot.size))
# ax.set_prop_cycle('color', cmap)
ph = ax.scatter(xplot,yplot,4,cplot,cmap='plasma')
ax.set_xlabel('Mode 1')
ax.set_ylabel('Mode 2')
cbar = plt.colorbar(ph)
ph.set_clim(30,75)
ax.set_ylim(-100,100)
# ax.set_xlim(-100,100)
ax.set_xlim(-110,110)
plt.grid(which='both', axis='both')
cbar.set_label('beach width [m]')


fig, ax = plt.subplots(2,1)
xplot = np.linspace(0,1,nx)
tmp = (PCs[:,1] > 30) & (PCs[:,2] < -15)
idemo = np.where(tmp)[0]
idemo4 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')


fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] < -20) & (PCs[:,2] >= 10) & (PCs[:,2] < 15)
idemo = np.where(tmp)[0]
idemo2 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')
fig, ax = plt.subplots()
ax.scatter(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo],1,c='tab:orange',alpha=0.1)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.plot([0,75],[mhw,mhw],'k--')
ax.plot([0,75],[mwl,mwl],'k--')
ax.set_xlim(0,75)


fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] < -16) & (PCs[:,2] > 15)
idemo = np.where(tmp)[0]
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')
fig, ax = plt.subplots()
ax.scatter(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo],1,c='tab:blue',alpha=0.1)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.plot([0,75],[mhw,mhw],'k--')
ax.plot([0,75],[mwl,mwl],'k--')
ax.set_xlim(0,75)



fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] > 25) & (PCs[:,2] > 25)
idemo = np.where(tmp)[0]
idemo1 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')



fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] < -25) & (PCs[:,2] < -25)
idemo = np.where(tmp)[0]
idemo3 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
fig, ax = plt.subplots()
ax.scatter(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo],1,c='tab:green',alpha=0.1)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.plot([0,75],[mhw,mhw],'k--')
ax.plot([0,75],[mwl,mwl],'k--')
ax.set_xlim(0,75)



fig, ax = plt.subplots(2,2)
cplot1 = np.tile(PCs[idemo1,0],(rescaled_xplot.shape[0],))
ph1 = ax[0,1].scatter(rescaled_xplot[:,idemo1],rescaled_PCAprofiles[:,idemo1],4,cplot1,cmap='coolwarm')
# ph1 = ax[0,1].scatter(np.tile(xplot,(idemo1.size,1)).T,rescaled_PCAprofiles[:,idemo1],4,cplot1,cmap='coolwarm')
ph1.set_clim(-40, 40)

cplot2 = np.tile(PCs[idemo2,0],(rescaled_xplot.shape[0],))
ph2 = ax[0,0].scatter(rescaled_xplot[:,idemo2],rescaled_PCAprofiles[:,idemo2],4,cplot2,cmap='coolwarm')
# ph2 = ax[0,0].scatter(np.tile(xplot,(idemo2.size,1)).T,rescaled_PCAprofiles[:,idemo2],4,cplot2,cmap='coolwarm')
ph2.set_clim(-40, 40)

cplot3 = np.tile(PCs[idemo3,0],(rescaled_xplot.shape[0],))
ph3 = ax[1,0].scatter(rescaled_xplot[:,idemo3],rescaled_PCAprofiles[:,idemo3],4,cplot3,cmap='coolwarm')
# ph3 = ax[1,0].scatter(np.tile(xplot,(idemo3.size,1)).T,rescaled_PCAprofiles[:,idemo3],4,cplot3,cmap='coolwarm')
ph3.set_clim(-40, 40)

cplot4 = np.tile(PCs[idemo4,0],(rescaled_xplot.shape[0],))
ph4 = ax[1,1].scatter(rescaled_xplot[:,idemo4],rescaled_PCAprofiles[:,idemo4],4,cplot4,cmap='coolwarm')
# ph4 = ax[1,1].scatter(np.tile(xplot,(idemo4.size,1)).T,rescaled_PCAprofiles[:,idemo4],4,cplot4,cmap='coolwarm')
# ch = plt.colorbar(ph4)
ph4.set_clim(-40, 40)

# Calculate correlation between CHANGE in PCs and environmental variables
# with open('elev_processed_elev&slopes_scaled.pickle', 'rb') as file:
#     scaled_profiles,_ = pickle.load(file)
# with open('elev_processed_base.pickle', 'rb') as file:
#     _,lidar_xFRF,profile_width,_,_,_ = pickle.load(file)
# with open('elev_processed_elev.pickle', 'rb') as file:
#     zsmooth_fullspan, _, _ = pickle.load(file)
# with open('IO_alignedintime.pickle', 'rb') as file:
#     time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,_,_,_,data_lidarwg110,_,_,_,_ = pickle.load(file)
# with open('beach_stats.pickle', 'rb') as file:
#     var_beachwid,var_beachvol,var_beachslp = pickle.load(file)

var_wave8m_Hs = data_wave8m[:,0]
var_wave8m_Tp = data_wave8m[:,1]
var_wave8m_dir = data_wave8m[:,2]
var_wave17m_Hs = data_wave17m[:,0]
var_wave17m_Tp = data_wave17m[:,1]
var_wave17m_dir = data_wave17m[:,2]
var_lidar_elev2p = data_lidar_elev2p[:].flatten()
var_tidegauge = data_tidegauge[:].flatten()
var_wg110_Hs = data_lidarwg110[:,0]
var_wg110_HsIN = data_lidarwg110[:,1]
var_wg110_HsIG = data_lidarwg110[:,2]
var_wg110_Tp = data_lidarwg110[:,3]
var_wg110_TmIG = data_lidarwg110[:,4]

## Calculate basic beach profile parameters
scaled_profiles = zsmooth_fullspan_scale
var_beachwid = profile_width
var_beachvol = np.empty(shape=var_beachwid.shape)
var_beachslp = np.empty(shape=var_beachwid.shape)
var_beachvol[:] = np.nan
var_beachslp[:] = np.nan
dx = 0.1
for ii in np.arange(time_fullspan.size):
    if sum(~np.isnan(zsmooth_fullspan[:,ii])) > 0:
        tmp = np.abs(zsmooth_fullspan[:,ii] - var_tidegauge[ii])
        if sum(~np.isnan(tmp)) > 0:
            ix_shoreline = np.where(tmp == np.nanmin(tmp))[0][0]
            iix = np.arange(-50,50)+ix_shoreline
            ztmp = zsmooth_fullspan[iix,ii]
            ztmp = ztmp[~np.isnan(ztmp)]
            slptmp = (ztmp[-1]-ztmp[0])/(dx*ztmp.size)
            var_beachslp[ii] = slptmp
            delx = var_beachwid[ii]/nx
            var_beachvol[ii] = np.sum(scaled_profiles[ii,:])*delx

# with open('beach_stats.pickle', 'wb') as file:
#     pickle.dump([var_beachwid,var_beachvol,var_beachslp], file)
varnames = ['beachwid','beachvol','beachslp','tidegauge','wave8m_Hs','wave8m_Tp','wave8m_dir','wave17m_Hs',
            'wave17m_Tp','wave17m_dir','wg110_HsIN','wg110_HsIG','wg110_Tp','wg110_TmIG','lidar_elev2p']
PC_fullspan = np.empty((4,time_fullspan.size))
PC_fullspan[:] = np.nan
PC_fullspan[0,[tkeep[ikeep]]] = PCs[:,0]
PC_fullspan[1,[tkeep[ikeep]]] = PCs[:,1]
PC_fullspan[2,[tkeep[ikeep]]] = PCs[:,2]
PC_fullspan[3,[tkeep[ikeep]]] = PCs[:,3]
fig, ax = plt.subplots()
ax.plot(time_fullspan,PC_fullspan[0,:],'o')
ax.plot(time_fullspan[tkeep[ikeep]],PCs[:,0],'o')

## Calculate correlation between del_PCs and beach/hydro variables
delPC_fullspan = np.empty((4,time_fullspan.size))
delPC_fullspan[:] = np.nan
delPC_fullspan[:,1:] = PC_fullspan[:,1:] - PC_fullspan[:,0:-1]
pcorrvals = np.empty((4,len(varnames)))
for ii in np.arange(len(varnames)):
    for jj in np.arange(4):
        exec('varx = var_' + varnames[ii])
        vary = delPC_fullspan[jj,:]
        ijok = ~np.isnan(vary) & ~np.isnan(varx)
        # pearsons = np.cov(varx[ijok],vary[ijok])/(np.std(varx[ijok])*np.std(vary[ijok]))
        # numerator = np.sum((varx[ijok] - np.mean(varx[ijok])) * (vary[ijok] - np.mean(vary[ijok])))
        # denominator = np.sqrt(np.sum(varx[ijok] - np.mean(varx[ijok])) ** 2) * np.sqrt(np.sum(vary[ijok] - np.mean(vary[ijok])) ** 2)
        pearsons = np.corrcoef(varx[ijok],vary[ijok])
        pcorrvals[jj,ii] = pearsons[0,1]
# with open('pearsons_values.pickle', 'wb') as file:
#     pickle.dump([pcorrvals], file)
fig, ax = plt.subplots()
iplot = (np.arange(len(varnames)) < 6) + (np.arange(len(varnames)) > 9)
sns.heatmap(pcorrvals[:,iplot].T, annot=True, linewidth=.5, fmt=".2f", cmap='RdBu', vmin=-0.5, vmax=0.5)
ax.set_xlabel('D/Dt of Mode [1/hr]')

fig, ax = plt.subplots()
ax.hist(var_beachslp,bins=50)