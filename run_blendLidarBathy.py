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
from funcs.find_nangaps import find_nangaps


################## LOAD DATA ##################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'

with open(picklefile_dir+'cleanLidarProfiles.pickle','rb') as file:
    lidar_xFRF,time_fullspan,final_profile_fullspan_best = pickle.load(file)
lidarprofile_fullspan = final_profile_fullspan_best[:]
with open(picklefile_dir + 'tidalAveragedMetrics.pickle', 'rb') as file:
    datload = pickle.load(file)
list(datload)
bathysurvey_elev = np.array(datload['smoothUpperTidalAverage'])
bathysurvey_times = np.array(datload['highTideTimes'])


################## ISOLATE BATHY DATA FROM COMBINED ##################


# Find overlap in Dylan's elevation set and the lidar profiles
nt = np.nanmax([bathysurvey_elev.T.shape[1], lidarprofile_fullspan.shape[1]])
nx = lidar_xFRF.size
bathypresenc = np.empty((nx,nt))
bathypresenc[:] = np.nan
lidarpresenc = np.empty((nx,nt))
lidarpresenc[:] = np.nan
lidarpresenc[~np.isnan(lidarprofile_fullspan)] = 1
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
# XX, TT = np.meshgrid(lidar_xFRF, time_fullspan)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(elev_overlap.T, elev_overlap.T.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=2, c=zz, vmin=0, vmax=2, cmap='viridis')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('OVERLAP')
# ax.set_xlabel('x [m, FRF]')
# ax.set_ylabel('time')
# ax.set_title('Lidar = 1, BathySurvey = 0.5')
## TEST TO CONFIRM NO-OVERLAP
# testbathy = bathy_nolidar+lidarprofile_fullspan
# zscatter = np.reshape(testbathy.T, testbathy.T.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=2, c=zz, vmin=0, vmax=2, cmap='viridis')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('bathy + lidar')
## PROFILE PLOT OF DLA'S DATA
# fig, ax = plt.subplots()
zmean = np.nanmean(bathy_nolidar,axis=1)
zstd = np.nanstd(bathy_nolidar,axis=1)
# ax.plot(lidar_xFRF,bathy_nolidar)
# ax.plot(lidar_xFRF,zmean,'k')
# ax.plot(lidar_xFRF,zmean+2.*zstd,'k:')
# ax.plot(lidar_xFRF,zmean-2.*zstd,'k:')
# ax.set_xlabel('xFRF [m]')
# ax.set_ylabel('z [m]')
thresh = zmean+2*zstd
iixx = (lidar_xFRF >= 146.97) & (lidar_xFRF <= 164)
thresh_iixx = thresh[iixx]
bathy_nolidar_clean = np.empty(shape=bathy_nolidar.shape)
bathy_nolidar_clean[:] = bathy_nolidar[:]
for tt in np.arange(time_fullspan.size):
    ztmp = bathy_nolidar[iixx,tt]
    ztmp[ztmp > thresh_iixx] = np.nan
    bathy_nolidar_clean[iixx,tt] = ztmp
# fig, ax = plt.subplots()
# ax.plot(lidar_xFRF,bathy_nolidar_clean)
# ax.set_xlabel('xFRF [m]')
# ax.set_ylabel('z [m]')


################## INTERP BATHY DATA IN TIME AND SPACE ##################

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
        zInterpAll[:,ix] = np.interp(time_fullspan, bathy_dates_unique[iinotnan], tempz[iinotnan])

## Ok, now plot interps
# fig, ax = plt.subplots()
# ax.plot(xinterp,zInterpSurvey.T)
# ax.set_xlabel('xFRF [m]')
# ax.set_ylabel('z [m]')
# ax.set_title('Interp in space')
zmean = np.nanmean(zInterpSurvey,axis=0)
zstd = np.nanstd(zInterpSurvey,axis=0)
# ax.plot(xinterp,zmean,'k')
# ax.plot(xinterp,zmean+2.*zstd,'k:')
# ax.plot(xinterp,zmean-2.*zstd,'k:')
# fig, ax = plt.subplots()
# ax.plot(xinterp,zInterpAll.T)
# ax.set_xlabel('xFRF [m]')
# ax.set_ylabel('z [m]')
# ax.set_title('Interp in time')
zmean = np.nanmean(zInterpAll,axis=0)
zstd = np.nanstd(zInterpAll,axis=0)
# ax.plot(xinterp,zmean,'k')
# ax.plot(xinterp,zmean+2.*zstd,'k:')
# ax.plot(xinterp,zmean-2.*zstd,'k:')
# Plot as surface...
XX, TT = np.meshgrid(xinterp, time_fullspan)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(zInterpAll, zInterpAll.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=3, c=zz, cmap='viridis')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('z [m]')
# ax.set_xlabel('x [m, FRF]')
# ax.set_ylabel('time')



################## COMBINE LIDAR AND INTERP'ED BATHY ##################

ZbFull_addLidar = np.zeros(shape=lidarprofile_fullspan.shape)*np.nan
lidarinterp_fullspan = np.zeros(shape=lidarprofile_fullspan.shape)*np.nan
bathyinterp_fullspan = np.zeros(shape=lidarprofile_fullspan.shape)*np.nan
weighted_bathy_fullspan = np.zeros(shape=lidarprofile_fullspan.shape)*np.nan
weighted_lidar_fullspan = np.zeros(shape=lidarprofile_fullspan.shape)*np.nan
# for tt in np.arange(30):
for tt in np.arange(time_fullspan.size):

    iilidarnotnan = np.where(~np.isnan(lidarprofile_fullspan[:,tt]))[0]
    iibathynotnan = np.where(~np.isnan(zInterpAll[tt,:]))[0]
    if sum(iilidarnotnan) & sum(iibathynotnan) >= 10:
        lidarprof_tt = lidarprofile_fullspan[iilidarnotnan[0]:iilidarnotnan[-1],tt]
        lidarprof_xx = lidar_xFRF[iilidarnotnan[0]:iilidarnotnan[-1]]
        lidarprof_tt[lidarprof_tt == 0.0] = np.nan
        bathyprof_tt = zInterpAll[tt,iibathynotnan[0]:iibathynotnan[-1]]
        bathyprof_xx = xinterp[iibathynotnan[0]:iibathynotnan[-1]]
        bathyprof_tt[bathyprof_tt == 0.0] = np.nan

        if (np.sum(~np.isnan(lidarprof_tt)) > 5) & (np.sum(~np.isnan(bathyprof_tt)) > 5):
            # print(str(tt))
            # fig, ax = plt.subplots()
            # ax.plot(bathyprof_xx, bathyprof_tt)
            # ax.plot(lidarprof_xx,lidarprof_tt)

            # interp lidar to full profile
            tmplidar = np.zeros(shape=lidar_xFRF.shape) * np.nan
            iixx = np.where((lidar_xFRF > lidarprof_xx[0]) & (lidar_xFRF < lidarprof_xx[-1]))[0]
            xv = lidarprof_xx
            yv = lidarprof_tt
            # tmplidar[iixx] = np.interp(lidar_xFRF[iixx], xv[~np.isnan(yv)], yv[~np.isnan(yv)])
            tmplidar[iixx] = interpolate_with_max_gap(xv, yv, lidar_xFRF[iixx], max_gap=5,
                                                      orig_x_is_sorted=False, target_x_is_sorted=False)
            # create weight for lidar profile, then apply weight
            # weight_lidar = np.ones(shape=lidarprof_tt.shape)
            weight_lidar = np.ones(shape=tmplidar.shape)
            xend = lidarprof_xx[-1]
            interpdist = 5
            # iixx = np.where((lidarprof_xx >= xend-interpdist) & (lidarprof_xx <= xend))[0]
            iixx = np.where((lidar_xFRF > xend-interpdist) & (lidar_xFRF < xend))[0]
            weight_lidar[iixx] = np.linspace(1,0,len(iixx))
            # weight_lidar[np.isnan(lidarprof_tt)] = np.nan
            # weighted_lidar = lidarprof_tt * weight_lidar
            weighted_lidar = tmplidar * weight_lidar
            weighted_lidar[np.isnan(tmplidar)] = np.nan
            weighted_lidar_fullspan[0:len(weighted_lidar),tt] = weighted_lidar

            # interp bathy to full profile
            tmpbathy = np.zeros(shape=lidar_xFRF.shape) * np.nan
            iixx = np.where((lidar_xFRF > lidarprof_xx[0]) & (lidar_xFRF > bathyprof_xx[0]) & (lidar_xFRF < bathyprof_xx[-1]))[0]
            xv = bathyprof_xx
            yv = bathyprof_tt
            # tmpbathy[iixx] = np.interp(lidar_xFRF[iixx], xv[~np.isnan(yv)], yv[~np.isnan(yv)])
            tmpbathy[iixx] = interpolate_with_max_gap(xv, yv, lidar_xFRF[iixx], max_gap=5,
                                                      orig_x_is_sorted=False, target_x_is_sorted=False)
            # create weight for bathy profile, then apply weight
            weight_bathy = np.ones(shape=tmpbathy.shape)
            iixx = np.where((lidar_xFRF > xend-interpdist) & (lidar_xFRF < xend))[0]
            weight_bathy[iixx] = np.linspace(0, 1, len(iixx))
            weight_bathy[lidar_xFRF < xend-interpdist] = 0
            # weight_bathy[np.isnan(bathyprof_tt)] = np.nan
            weighted_bathy = tmpbathy * weight_bathy
            weighted_bathy[np.isnan(tmpbathy)] = np.nan
            weighted_bathy_fullspan[0:len(weighted_bathy),tt] = weighted_bathy

            # save blended profile
            ZbFull_addLidar[:,tt] = np.nansum(np.vstack((weighted_bathy,weighted_lidar)),axis=0)
            bathyinterp_fullspan[:,tt] = tmpbathy
            lidarinterp_fullspan[:, tt] = tmplidar
ZbFull_addLidar[ZbFull_addLidar == 0.000] = np.nan
topobathy_fullspan =ZbFull_addLidar[:]

################## PLOT FINAL RESULTS ##################

# PLOT surveys from Dylan
XX, TT = np.meshgrid(lidar_xFRF, bathysurvey_times)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(bathysurvey_elev, bathysurvey_elev.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
fig.set_size_inches(15,3.)
ph = ax.scatter(tt,xx, s=3, c=zz, cmap='rainbow',vmin=-4,vmax=8)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('diff [m]')
ax.set_title('tidally averaged topo-bathy')

tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(bathy_nolidar.T, bathy_nolidar.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
fig.set_size_inches(15,3.)
ph = ax.scatter(tt,xx, s=3, c=zz, cmap='rainbow',vmin=-4,vmax=8)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_title('bathy only')
zscatter = np.reshape(ZbFull_addLidar.T, ZbFull_addLidar.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
fig.set_size_inches(15,3.)
ph = ax.scatter(tt,xx, s=3, c=zz, cmap='rainbow',vmin=-4,vmax=8)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_title('bathy + lidar')

fig, ax = plt.subplots()
ax.plot(lidar_xFRF,bathy_nolidar)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('bathy only')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,ZbFull_addLidar)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Blend lidar and bathy')
zmean = np.nanmean(ZbFull_addLidar,axis=1)
zstd = np.nanstd(ZbFull_addLidar,axis=1)
ax.plot(lidar_xFRF,zmean,'k')
ax.plot(lidar_xFRF,zmean+2.*zstd,'k:')
ax.plot(lidar_xFRF,zmean-2.*zstd,'k:')


########## SAVE BLENDED TOPO-BATHY ##########

with open(picklefile_dir+'blendedLidarBathy.pickle', 'wb') as file:
   pickle.dump([lidar_xFRF, time_fullspan, topobathy_fullspan],file)