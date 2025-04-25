import pickle
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd  # to load the dataframe
import os
from datetime import datetime
from funcs.align_data_time import align_data_fullspan
from funcs.create_contours import *
from funcs.wavefuncs import *
from funcs.calculate_beachvol import *

matplotlib.rcParams.update({'font.size': 14})


# picklefile_dir = 'G:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_26Nov2024/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/frf_data_backup/processed/processed_20Feb2025/'
# picklefile_dir = './'
# with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
#     time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)

with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_Nlook60_PCApostDVol_shifted.pickle', 'rb') as file:
    xplot_shift, time_fullspan, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV, reconstruct_profNorm_fullspan, reconstruct_prof_fullspan, dataobs_shift_fullspan, dataobs_fullspan, data_profIDs_dVolThreshMet, data_hydro, datahydro_fullspan = pickle.load(file)
dataPCA_fullspan = reconstruct_prof_fullspan
dx = 0.1
xplot_shift = np.arange(reconstruct_prof_fullspan.shape[0])*dx
with open(picklefile_dir+'waves_8m&17m_2015_2024.pickle','rb') as file:
    [data_wave8m,data_wave17m,data_wave8m_filled] = pickle.load(file)
with open(picklefile_dir+'stormHs95_Over12Hours.pickle','rb') as f:
    output = pickle.load(f)
list(output)
storm_start = np.array(output['startTimeStormList'])
storm_end = np.array(output['endTimeStormList'])
storm_startWIS = np.array(output['startTimeStormListWIS'])
storm_endWIS = np.array(output['endTimeStormListWIS'])

with open(picklefile_dir + 'tidalAveragedMetrics.pickle', 'rb') as file:
    datload = pickle.load(file)
list(datload)
bathysurvey_elev = np.array(datload['smoothUpperTidalAverage'])
bathysurvey_times = np.array(datload['highTideTimes'])



# Plot climatology
fig, ax = plt.subplots(3,1)
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
Hs = data_wave8m_filled[:,0]
Tp = data_wave8m_filled[:,1]
Hs_94 = 2.1171875
ax[0].plot(tplot,Hs,linewidth=0.5,color='k')
# ax[0].grid()
ax[0].set_ylabel('$H_s$ [m]')
ax[0].set_xlim(min(tplot),max(tplot))
ax[0].set_xticklabels('')
ax[0].plot(tplot,np.zeros(shape=tplot.shape)+Hs_94,'r--')
ax[1].plot(tplot,Tp,linewidth=0.3,color='k')
# ax[1].grid()
ax[1].set_ylabel('$T_p$ [s]')
ax[1].set_xlim(min(tplot),max(tplot))
ax[1].set_xticklabels('')
ax[2].plot(tplot,(Hs**2)*Tp,linewidth=1,color='k')
# ax[2].grid()
ax[2].set_ylabel('$H_s^2T_p$ [m$^2$s]')
ax[2].set_xlim(min(tplot),max(tplot))
fig.set_size_inches(11.32,  4.8)
#### add storm periods
for nn in np.arange(storm_start.size):
    x1 = pd.to_datetime(storm_start[nn], unit='s', origin='unix')
    x2 = pd.to_datetime(storm_end[nn], unit='s', origin='unix')
    y1, y2 = (0,6)
    left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
    patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='c')
    ax[0].add_patch(patch)
    ax[0].set_ylim(y1,y2)
    y1, y2 = (0, 25)
    left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
    patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='c')
    ax[1].add_patch(patch)
    ax[1].set_ylim(y1, y2)
    y1, y2 = (0, 400)
    left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
    patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='c')
    ax[2].add_patch(patch)
    ax[2].set_ylim(y1, y2)

# Calculate beach volume and width
mlw = -0.62
mwl = -0.13
zero = 0
mhw = 0.36
dune_toe = 3.22
upper_lim = 5.95
cont_elev = np.array([mlw,mwl,zero,mhw,dune_toe,upper_lim]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts_pca, cmean, cstd = create_contours(dataPCA_fullspan.T,time_fullspan,xplot_shift,cont_elev)
beachVol_pca, beachVol_xc_pca, dBeachVol_dt_pca, total_beachVol_pca, total_dBeachVol_dt_pca, total_obsBeachWid_pca =  calculate_beachvol(dataPCA_fullspan.T,time_fullspan,xplot_shift,cont_elev,cont_ts_pca)
total_beachVol_pca[total_beachVol_pca == 0] = np.nan

fig, ax = plt.subplots()
#### add storm periods
ax.scatter(tplot[0],cont_ts_pca[2,0],3,color='tab:orange',linewidth=0.5)
for nn in np.arange(storm_start.size):
    x1 = pd.to_datetime(storm_start[nn], unit='s', origin='unix')
    x2 = pd.to_datetime(storm_end[nn], unit='s', origin='unix')
    y1, y2 = (20,85)
    left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
    patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='c')
    ax.add_patch(patch)
    ax.set_ylim(y1,y2)
ax.scatter(tplot,cont_ts_pca[2,:],3,color='tab:orange',linewidth=0.5)
ax.set_xlim(min(tplot),max(tplot))
# ax.grid()
ax.set_ylabel('$Xc_{z=0}$ [m]')
fig.set_size_inches(9.46,  1.75)
plt.tight_layout()

fig, ax = plt.subplots()
mhw_xc = cont_ts_pca[3,:]
mhw_xc_smoothed = np.convolve(mhw_xc,np.ones(12)/12,mode="same")
ax.plot(tplot,mhw_xc,'o-',color='tab:orange',linewidth=0.5)
ax.plot(tplot,mhw_xc_smoothed,'.-',color='k',linewidth=0.5)



# plot all profiles
fig, ax = plt.subplots()
ax.plot(xplot_shift,dataobs_shift_fullspan,color='0.5',alpha=0.1)
ymean = np.nanmean(dataobs_shift_fullspan,axis=1)
ystd = np.nanstd(dataobs_shift_fullspan,axis=1)
ax.plot(xplot_shift,ymean,'k')
ax.plot(xplot_shift,ymean+ystd,'k:')
ax.plot(xplot_shift,ymean-ystd,'k:')
ax.plot(xplot_shift,mhw+np.zeros(shape=xplot_shift.shape),'--',color='c',linewidth=2)
ax.plot([0,0]+cmean[2],[-4,8],color='tab:orange')
ax.plot([0,0]+(cmean[2]+cstd[2]),[-4,8],':',color='tab:orange')
ax.plot([0,0]+(cmean[2]-cstd[2]),[-4,8],':',color='tab:orange')
x1 = cmean[2]-cstd[2]
x2 = cmean[2]+cstd[2]
y1, y2 = (-4,8)
left, bottom, width, height = (x1, y1, x2 - x1, y2 - y1)
patch = plt.Rectangle((left, bottom), width, height, alpha=0.25, color='tab:orange')
ax.add_patch(patch)
ax.set_ylim(-4,7)
ax.set_xlim(0,130)
ax.grid()
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')

# plot one profile
ii = 255
fig, ax = plt.subplots()
ax.plot(xplot_shift,mhw+np.zeros(shape=xplot_shift.shape),'--',color='c',linewidth=2)
ax.plot(xplot_shift,dataobs_shift_fullspan[:,ii],color='0.5',linewidth=2)
ax.scatter(cont_ts_pca[2,ii],cont_elev[2],40,color='tab:orange')
ax.set_ylim(-4,7)
ax.set_xlim(0,130)
ax.grid()
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')

# plot histograms of cont_ts, WL, Hs, Tp, dir available for analysis
fig, ax = plt.subplots(2,3)
iiplot = ~np.isnan(cont_ts_pca[2,:])
yplot = cont_ts_pca[2,iiplot]
ax[1,1].hist(yplot,bins=30,density=False,color='tab:orange')
yplot = datahydro_fullspan[0,iiplot]
ax[0,0].hist(yplot,bins=30,density=False,color='0.6')
yplot = datahydro_fullspan[1,iiplot]
ax[0,1].hist(yplot,bins=30,density=False,color='0.6')
yplot = datahydro_fullspan[2,iiplot]
ax[0,2].hist(yplot,bins=30,density=False,color='0.6')
yplot = datahydro_fullspan[3,iiplot]
ax[1,0].hist(yplot,bins=30,density=False,color='0.6')
ax[0,0].set_title('$\overline{\eta}$ [m]')
ax[0,1].set_title('$H_s$ [m]')
ax[0,2].set_title('$T_p$ [s]')
ax[1,0].set_title('$\\theta_p$ (deg.)')
ax[1,1].set_title('$Xc_{MHW}$ [m]')
ax[0,0].set_ylim(0,4400)
ax[0,1].set_ylim(0,4400)
ax[0,2].set_ylim(0,4400)
ax[1,0].set_ylim(0,4400)
ax[1,1].set_ylim(0,4400)

xplot = np.arange(bathysurvey_elev.shape[1])*dx
XX, TT = np.meshgrid(xplot,bathysurvey_times)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(bathysurvey_elev, bathysurvey_elev.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(tt, xx, s=1, c=zz, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_ylabel('x [m]')
ax.set_ylim(min(xplot),max(xplot))
ax.set_xlim(min(bathysurvey_times),max(bathysurvey_times))
fig.set_size_inches([7.5,3.4])
plt.tight_layout()




