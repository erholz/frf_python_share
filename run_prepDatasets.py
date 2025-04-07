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
from scipy.interpolate import CubicSpline, interp1d
from funcs.interpgap import interpolate_with_max_gap



################## LOAD DATA ##################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'
with open(picklefile_dir+'blendedLidarBathy.pickle', 'rb') as file:
   lidar_xFRF, time_fullspan, topobathy_fullspan = pickle.load(file)
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
watlev = np.squeeze(data_tidegauge[:])
Hs8m = data_wave8m[:,0]
Tp8m = data_wave8m[:,1]
dir8m = data_wave8m[:,2]
Hs17m = data_wave17m[:,0]
Tp17m = data_wave17m[:,1]
dir17m = data_wave17m[:,2]


################## FILL REMAINING GAPS IN TIME ##################

maxgap = 4
xv = np.arange(time_fullspan.size)
yv = watlev[:]
iixx = ~np.isnan(yv)
watlev_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=True, target_x_is_sorted=True)
yv = Hs8m[:]
iixx = ~np.isnan(yv)
Hs8m_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=True, target_x_is_sorted=True)
yv = Tp8m[:]
iixx = ~np.isnan(yv)
Tp8m_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=True, target_x_is_sorted=True)
yv = dir8m[:]
iixx = ~np.isnan(yv)
dir8m_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=True, target_x_is_sorted=True)
yv = Hs17m[:]
iixx = ~np.isnan(yv)
Hs17m_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=True, target_x_is_sorted=True)
yv = Tp17m[:]
iixx = ~np.isnan(yv)
Tp17m_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=False, target_x_is_sorted=False)
yv = dir17m[:]
iixx = ~np.isnan(yv)
dir17m_fullspan = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap,orig_x_is_sorted=False, target_x_is_sorted=False)

topobathy_fullspan_gapfilled = np.empty(shape=topobathy_fullspan.shape)*np.nan
for jj in np.arange(lidar_xFRF.size):
    maxgap = 5
    xv = np.arange(time_fullspan.size)
    yv = topobathy_fullspan[jj,:]
    iixx = ~np.isnan(yv)
    yvnew = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap, orig_x_is_sorted=True, target_x_is_sorted=True)
    topobathy_fullspan_gapfilled[jj,:] = yvnew


################## FILL REMAINING GAPS IN SPACE ##################


for tt in np.arange(time_fullspan.size):
    maxgap = 5
    xv = np.arange(lidar_xFRF.size)
    yv = topobathy_fullspan_gapfilled[:,tt]
    iixx = ~np.isnan(yv)
    yvnew = interpolate_with_max_gap(xv[iixx], yv[iixx], xv, max_gap=maxgap, orig_x_is_sorted=True, target_x_is_sorted=True)
    topobathy_fullspan_gapfilled[:,tt] = yvnew


################## SHIFT PROFILES    ##################

xmin = 49
zc_shore = 6
xc_shore = np.empty((time_fullspan.size,))*np.nan
xc_sea = np.empty((time_fullspan.size,))*np.nan
dx = 0.1
xplot = np.arange(lidar_xFRF.size)*dx
for jj in np.arange(time_fullspan.size):
    prof_jj = topobathy_fullspan_gapfilled[:,jj]
    ix_inspan = np.where((prof_jj <= zc_shore) & (lidar_xFRF >= xmin))[0]
    if ix_inspan.size > 25:
        padding = 2
        itrim = np.arange(ix_inspan[0] - padding, xplot.size)
        xtmp = xplot[itrim]
        ztmp = prof_jj[itrim]
        xtmp = xtmp[~np.isnan(ztmp)]        # remove nans
        ztmp = ztmp[~np.isnan(ztmp)]        # remove nans
        if sum(~np.isnan(xtmp)) > 25:
            xc_shore[jj] = np.interp(zc_shore,ztmp[0:5],xtmp[0:5])
            xc_sea[jj] = np.nanmax(xtmp)
beachwid = xc_sea - xc_shore
numx = int(np.floor(np.nanmax(beachwid))/dx)
# numx = int(np.ceil(np.nanmin(beachwid))/dx)
xplot_shift = np.arange(numx)*dx
topobathy_fullspan_gapfilled_shift = np.empty((numx,time_fullspan.size))*np.nan
for jj in np.arange(time_fullspan.size):
    prof_jj = topobathy_fullspan_gapfilled[:, jj]
    ix_inspan = np.where((prof_jj <= zc_shore))[0]
    if ix_inspan.size > 25:
        padding = 2
        itrim = np.arange(ix_inspan[0] - padding, xplot.size)
        xtmp = xplot[itrim]
        ztmp = prof_jj[itrim]
        xtmp = xtmp[~np.isnan(ztmp)]  # remove nans
        ztmp = ztmp[~np.isnan(ztmp)]  # remove nans
        # xend = xc_shore[jj] + np.nanmin(beachwid)
        # xinterp = np.linspace(xc_shore[jj], xend, numx)
        xend = xtmp[-1]
        xinterp = np.linspace(xc_shore[jj], xend, int(np.floor(xend-xc_shore[jj])/dx))
        if sum(~np.isnan(xtmp)) > 25:
            zinterp = np.interp(xinterp, xtmp, ztmp)
            topobathy_fullspan_gapfilled_shift[:xinterp.size,jj] = zinterp


########## PLOT TO VERIFY CHANGES ##########

tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')

fig, ax = plt.subplots(4,1)
fig.set_size_inches(17.5,7)
ax[0].plot(tplot,watlev,'oc',alpha=0.1)
ax[0].plot(tplot,watlev_fullspan,'.',color='tab:blue')
ax[0].set_ylabel('$\overline{\eta}$')
ax[0].set_xticklabels([])
ax[1].plot(tplot,Hs8m,'oc',alpha=0.1)
ax[1].plot(tplot,Hs8m_fullspan,'.',color='tab:blue')
ax[1].plot(tplot,Hs17m,'o',color='yellow',alpha=0.1)
ax[1].plot(tplot,Hs17m_fullspan,'.',color='tab:red')
ax[1].set_ylabel('$H_s$')
ax[1].set_xticklabels([])
ax[2].plot(tplot,Tp8m,'oc',alpha=0.1)
ax[2].plot(tplot,Tp8m_fullspan,'.',color='tab:blue')
ax[2].plot(tplot,Tp17m,'o',color='yellow',alpha=0.1)
ax[2].plot(tplot,Tp17m_fullspan,'.',color='tab:red')
ax[2].set_ylabel('$T_p$')
ax[2].set_xticklabels([])
ax[3].plot(tplot,dir8m,'oc',alpha=0.1)
ax[3].plot(tplot,dir8m_fullspan,'.',color='tab:blue')
ax[3].plot(tplot,dir17m,'o',color='yellow',alpha=0.1)
ax[3].plot(tplot,dir17m_fullspan,'.',color='tab:red')
ax[3].set_ylabel('$\\theta_p$')

fig, ax = plt.subplots()
ax.plot(xplot_shift,topobathy_fullspan_gapfilled_shift)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.set_title('shifted profiles')

fig, ax = plt.subplots()
yplot1 = 100*np.sum(~np.isnan(topobathy_fullspan),axis=1)/time_fullspan.size
yplot2 = 100*np.sum(~np.isnan(topobathy_fullspan_gapfilled),axis=1)/time_fullspan.size
yplot3 = 100*np.sum(~np.isnan(topobathy_fullspan_gapfilled_shift),axis=1)/time_fullspan.size
ax.plot(lidar_xFRF-lidar_xFRF[0],yplot1,'o',label='pre-filled')
ax.plot(lidar_xFRF-lidar_xFRF[0],yplot2,'*',label='post-filled')
ax.plot(xplot_shift,yplot3,'.',label='post-filled, shifted')
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('Percent Not-Nan')


########## SAVE FILLED TOPO-BATHY AND HYDRO ##########

hydro_fullspan = np.empty((4,time_fullspan.size))
hydro_fullspan[0,:] = watlev_fullspan
hydro_fullspan[1,:] = Hs8m_fullspan
hydro_fullspan[2,:] = Tp8m_fullspan
hydro_fullspan[3,:] = dir8m_fullspan
with open(picklefile_dir+'preppedHydroTopobathy.pickle', 'wb') as file:
   pickle.dump([lidar_xFRF, time_fullspan, topobathy_fullspan_gapfilled, xplot_shift, topobathy_fullspan_gapfilled_shift, hydro_fullspan],file)
