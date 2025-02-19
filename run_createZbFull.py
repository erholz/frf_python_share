import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import time
import pickle
from scipy.optimize import curve_fit
from funcs.create_contours import *
from scipy.interpolate import splrep, BSpline, splev, CubicSpline
from funcs.interpgap import interpolate_with_max_gap



# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
picklefile_dir = 'D:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]


## FIRST, load the processed data from Dylan
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
picklefile_dir = 'D:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_26Nov2024/'
with open(picklefile_dir + 'tidalAveragedMetrics.pickle', 'rb') as file:
    datload = pickle.load(file)
list(datload)
bathysurvey_elev = np.array(datload['smoothUpperTidalAverage'])
bathysurvey_times = np.array(datload['highTideTimes'])

# PLOT surveys from Dylan
XX, TT = np.meshgrid(lidar_xFRF, bathysurvey_times)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(bathysurvey_elev, bathysurvey_elev.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')

# Find overlap in Dylan's elevation set and the raw initial profiles
nt = np.nanmax([bathysurvey_elev.T.shape[1], lidarelev_fullspan.shape[1]])
nx = lidar_xFRF.size
bathypresenc = np.empty((nx,nt))
bathypresenc[:] = np.nan
lidarpresenc = np.empty((nx,nt))
lidarpresenc[:] = np.nan
lidarpresenc[~np.isnan(lidarelev_fullspan)] = 1
tplot_lidar = pd.to_datetime(time_fullspan, unit='s', origin='unix')
tplot_bathy = bathysurvey_times.copy()
bathysurvey_fullspan = np.empty((nx,nt))
bathysurvey_fullspan[:] = np.nan
for tt in np.arange(bathysurvey_times.size):
    if tplot_bathy[tt].minute == 30:
        tplot_bathy[tt] = tplot_bathy[tt] + dt.timedelta(minutes=30)
    ttdelta_min = np.nanmin(abs(tplot_lidar - tplot_bathy[tt])).astype('timedelta64[h]')
    ttdiff_min = np.nanmin(abs(tplot_lidar - tplot_bathy[tt])).astype('timedelta64[h]') / np.timedelta64(1, 'h')
    if ttdiff_min < 0.25:
        ttnew = np.where(abs(tplot_lidar - tplot_bathy[tt]) == ttdelta_min)[0]
        iinotnan = np.where(~np.isnan(bathysurvey_elev[tt,:]))[0]
        bathypresenc[iinotnan,ttnew] = 0.5
        bathysurvey_fullspan[iinotnan,ttnew] = bathysurvey_elev[tt,iinotnan]
# Plot the overlap of these two datasets
elev_overlap = np.nansum(np.dstack((lidarpresenc,bathypresenc)),2)
elev_overlap[elev_overlap == 0] = np.nan
bathy_nolidar = np.empty((nx,nt))
bathy_nolidar[:] = np.nan
bathy_nolidar[elev_overlap == 0.5] = bathysurvey_fullspan[elev_overlap == 0.5]
## SCATTER PLOT SHOWING OVERLAP
XX, TT = np.meshgrid(lidar_xFRF, time_fullspan)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(elev_overlap.T, elev_overlap.T.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=2, c=zz, vmin=0, vmax=2, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('OVERLAP')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')
ax.set_title('Lidar = 1, BathySurvey = 0.5')
## PROFILE PLOT OF DLA'S DATA
fig, ax = plt.subplots()
zmean = np.nanmean(bathy_nolidar,axis=1)
zstd = np.nanstd(bathy_nolidar,axis=1)
ax.plot(lidar_xFRF,bathy_nolidar)
ax.plot(lidar_xFRF,zmean,'k')
ax.plot(lidar_xFRF,zmean+2.*zstd,'k:')
ax.plot(lidar_xFRF,zmean-2.*zstd,'k:')
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
thresh = zmean+2*zstd
iixx = (lidar_xFRF >= 146.97) & (lidar_xFRF <= 164)
thresh_iixx = thresh[iixx]
bathy_nolidar_clean = np.empty(shape=bathy_nolidar.shape)
bathy_nolidar_clean[:] = bathy_nolidar[:]
for tt in np.arange(time_fullspan.size):
    ztmp = bathy_nolidar[iixx,tt]
    ztmp[ztmp > thresh_iixx] = np.nan
    bathy_nolidar_clean[iixx,tt] = ztmp
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,bathy_nolidar_clean)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')

# ########## DONT COMBINE ##########
# bathylidar_combo = np.empty(shape=bathy_nolidar.shape)
# bathylidar_combo[:] = profile_fullspan[:]
# bathylidar_combo[~np.isnan(bathy_nolidar_clean)] = bathy_nolidar_clean[~np.isnan(bathy_nolidar_clean)]
# fig, ax = plt.subplots()
# ax.plot(lidar_xFRF,bathylidar_combo)
# ax.set_xlabel('xFRF [m]')
# ax.set_ylabel('z [m]')
# ax.set_title('Bathy-Lidar combined')


bathy_times = time_fullspan
bathy_elevation = bathy_nolidar_clean
bathy_x = lidar_xFRF
bathy_dates_unique = np.unique(bathy_times)
offshore_x = 200
xinterp = np.arange(45, offshore_x+1, 0.1)
zInterpSurvey = np.zeros([len(bathy_dates_unique), len(xinterp)])*np.nan

for ii in range(len(bathy_dates_unique)):
    # iifind_data = np.where((bathy_times == bathy_dates_unique[ii]))
    iifind_data = ii
    if np.size(iifind_data) == 1:
        z_data_tmp = bathy_elevation[:,iifind_data]
        if sum(~np.isnan(z_data_tmp)) > 5:
            x_data_tmp = bathy_x[~np.isnan(z_data_tmp)]
            z_data_tmp = z_data_tmp[~np.isnan(z_data_tmp)]
            zInterp = interpolate_with_max_gap(x_data_tmp, z_data_tmp, xinterp, max_gap=5, orig_x_is_sorted=False,
                                               target_x_is_sorted=False)
            zInterpSurvey[ii, :] = zInterp
    else:
        # zInterpSurvey[ii, :] = zInterp
        print('error for ii = '+str(ii))

zInterpAll = np.zeros([len(time_fullspan), len(xinterp)]) * np.nan
for ix in np.arange(len(xinterp)):
    tempz = zInterpSurvey[:, ix]
    iinotnan = np.where(np.isnan(tempz) == False)
    if sum(~np.isnan(tempz)) > 5:
        zInterpAll[:,ix]  = np.interp(time_fullspan, bathy_dates_unique[iinotnan], tempz[iinotnan])

## Ok, now plot interps
fig, ax = plt.subplots()
ax.plot(xinterp,zInterpSurvey.T)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Interp in space')
zmean = np.nanmean(zInterpSurvey,axis=0)
zstd = np.nanstd(zInterpSurvey,axis=0)
ax.plot(xinterp,zmean,'k')
ax.plot(xinterp,zmean+2.*zstd,'k:')
ax.plot(xinterp,zmean-2.*zstd,'k:')

fig, ax = plt.subplots()
ax.plot(xinterp,zInterpAll.T)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Interp in time')
zmean = np.nanmean(zInterpAll,axis=0)
zstd = np.nanstd(zInterpAll,axis=0)
ax.plot(xinterp,zmean,'k')
ax.plot(xinterp,zmean+2.*zstd,'k:')
ax.plot(xinterp,zmean-2.*zstd,'k:')




XX, TT = np.meshgrid(xinterp, time_fullspan)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(zInterpAll.T, zInterpAll.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=3, c=zz, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')
