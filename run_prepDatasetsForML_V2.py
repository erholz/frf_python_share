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



## OPEN DICTS
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'data_poststorm_sliced.pickle','rb') as file:
    data_poststorm_all = pickle.load(file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'ZbFull_LidarBathyBlended.pickle', 'rb') as file:
#    _, _, ZbFull_LidarBathyBlended = pickle.load(file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'data_fullspan.pickle','rb') as file:
    data_fullspan = pickle.load(file)
    time_fullspan = data_fullspan["fullspan_time"]
    watlev_fullspan = data_fullspan["fullspan_tidegauge"]
    Hs8m_fullspan = data_fullspan["fullspan_Hs_8m"]
    Tp8m_fullspan = data_fullspan["fullspan_Tp_8m"]
    dir8m_fullspan = data_fullspan["fullspan_wavedir_8m"]
    Hs17m_fullspan = data_fullspan["fullspan_Hs_17m"]
    Tp17m_fullspan = data_fullspan["fullspan_Tp_17m"]
    dir17m_fullspan = data_fullspan["fullspan_wavedir_17m"]
    elev2p_fullspan = data_fullspan["fullspan_elev2p"]
    lidarwg_fullspan = data_fullspan["fullspan_lidargauge_110"]
    topobathy_fullspan = data_fullspan["fullspan_lidarbathy_blend_20Feb2025"]
    istormy_fullspan = data_fullspan["fullspan_istormy_1yes0no"]
    # data_fullspan["fullspan_lidarbathy_blend_20Feb2025"] = ZbFull_LidarBathyBlended
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'data_fullspan_addBlendedLidarBathy.pickle', 'wb') as file:
#    pickle.dump(data_fullspan, file)

########################## Add to "data_fullspan" the "Stormy" times to exclude ##########################

istormy_fullspan = np.ones(shape=time_fullspan.shape)
icusp_fullspan = np.zeros(shape=time_fullspan.shape)
for jj in np.arange(len(data_poststorm_all)):
    # get non-stormy times...
    timeslice = data_poststorm_all["data_poststorm" + str(jj)]["poststorm_time"]
    iijj = np.where(np.isin(time_fullspan,timeslice))[0]
    istormy_fullspan[iijj] = 0

data_fullspan["fullspan_istormy_1yes0no"] = istormy_fullspan
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'data_fullspan_addBlendedLidarBathy.pickle', 'wb') as file:
#    pickle.dump(data_fullspan, file)



########################## Find N-day series with adequate data coverage ##########################

Nlook = 4*24                   # look through Nlook hours to quantify data availability
profelev_numhrly_thresh = 0.75  # use to count percent of days (by x-locs) where THRESH % profile data available over Nlook
watlev_perctotal_thresh = 0.9
waves_perctotal_thresh = 0.9
lidarwg_perctotal_thresh = 0.9
set_starttime_dailybathythresh = np.empty(shape=time_fullspan.shape)
set_startii_dailybathythresh = np.empty(shape=time_fullspan.shape)
set_wavethreshmet = np.empty(shape=time_fullspan.shape)
set_watlevthreshmet = np.empty(shape=time_fullspan.shape)
set_lidarwgthreshmet = np.empty(shape=time_fullspan.shape)
set_profhrlythreshmet = np.empty((time_fullspan.size,lidar_xFRF.size))
set_nonstormythreshmet = np.empty(shape=time_fullspan.shape)

## Go through all times
for tt in np.arange(time_fullspan.size-Nlook):

    if (tt - Nlook) > 1:

            # isolate data for Nlook hrs
            ttlook = np.arange(tt,tt+Nlook)
            wavelook = Hs8m_fullspan[ttlook]
            watlevlook = watlev_fullspan[ttlook]
            lidarwglook = lidarwg_fullspan[ttlook,0]
            topobathy_look = topobathy_fullspan[:,ttlook]
            istormy_look = istormy_fullspan[ttlook]
            set_start = time_fullspan[ttlook[0]]
            set_end = time_fullspan[ttlook[-1]]
            set_start_ii = np.where(set_start == time_fullspan)[0]

            # calculate the percent available for hydro
            set_watlev_cover = np.nansum(~np.isnan(watlevlook))/Nlook
            set_wave_cover = np.nansum(~np.isnan(wavelook)) / Nlook
            set_lidarwg_cover = np.nansum(~np.isnan(lidarwglook)) / Nlook

            # day counter
            numdays = int(np.floor(Nlook/24))
            hrlyperc = np.empty(shape=(topobathy_look.shape[0],numdays))
            hrlyperc[:] = np.nan
            tmpii = 0
            for dd in np.arange(numdays):
                ztmp = topobathy_look[:,tmpii:tmpii+24]
                ytmp = np.nansum(~np.isnan(ztmp),axis=1)
                hrlyperc[ytmp/24 >= profelev_numhrly_thresh, dd] = 1
                tmpii = tmpii+24
            perchrly_threshmet = np.nansum(hrlyperc,axis=1)/numdays
            ytmp = np.zeros(shape=perchrly_threshmet.shape)
            ytmp[perchrly_threshmet == 1] = 1

            # IF SET MEETS PROFILE THRESH ANYWHERE, SAVE THAT TIME
            if sum(ytmp) > 1:
                # save the time where set profile data passes thresh
                set_starttime_dailybathythresh[tt] = set_start
                set_startii_dailybathythresh[tt] = set_start_ii
                # set the x-locs where condition met
                set_profhrlythreshmet[tt,:] = ytmp

                # save lidarwg condition (1/0)
                if set_lidarwg_cover >= lidarwg_perctotal_thresh:
                    threshmet = 1
                else:
                    threshmet = 0
                set_lidarwgthreshmet[tt] = threshmet
                # save wave condition (1/0)
                if set_wave_cover >= waves_perctotal_thresh:
                    threshmet = 1
                else:
                    threshmet = np.nan
                set_wavethreshmet[tt] = threshmet
                # save watlev condition (1/0)
                if set_watlev_cover >= watlev_perctotal_thresh:
                    threshmet = 1
                else:
                    threshmet = np.nan
                set_watlevthreshmet[tt] = threshmet
                # save storminess condtion (i.e., NOT STORMY is ok) (1/0)
                if sum(istormy_look) == 0:
                    threshmet = 1
                else:
                    threshmet = np.nan
                set_nonstormythreshmet[tt] = threshmet


# clean up initialized datasets...this was done for plotting in first version of this script
plot_profelev_cover_hrly = set_profhrlythreshmet[:, :]
plot_profelev_cover_hrly[plot_profelev_cover_hrly == 0] = np.nan
plot_watlevthreshmet = set_watlevthreshmet[:]
plot_watlevthreshmet[plot_watlevthreshmet == 0] = np.nan
plot_wavethreshmet = set_wavethreshmet[:]
plot_wavethreshmet[plot_wavethreshmet == 0] = np.nan
plot_lidarwgthreshmet = set_lidarwgthreshmet[:]
set_lidarwgthreshmet[set_lidarwgthreshmet == 0] = np.nan
plot_start_timekeep = set_starttime_dailybathythresh[:]
plot_start_iikeep = set_startii_dailybathythresh[:]


## Find the profiles where profiles are "good", long "enough", and we have watlev & wave data
dx = 0.1
length_profdatavail = np.nansum(plot_profelev_cover_hrly,axis=1)
length_criterion = (length_profdatavail >= 25/dx)       # profiles for ML analysis must have minimum length (for now)
wave_criterion = (plot_wavethreshmet == 1)
watlev_criterion = (plot_watlevthreshmet == 1)
nonstormy_criterion = (set_nonstormythreshmet == 1)
set_ii_prelimbathyhydrothreshmet = set_startii_dailybathythresh[length_criterion & wave_criterion & watlev_criterion & nonstormy_criterion]
set_time_prelimbathyhydrothreshmet = set_starttime_dailybathythresh[length_criterion & wave_criterion & watlev_criterion & nonstormy_criterion]

##### Save the set_id's that meet prelim conditions
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'set_id_tokeep_20Feb2025_Nlook'+str(Nlook)+'.pickle', 'wb') as file:
#     pickle.dump([set_ii_prelimbathyhydrothreshmet,set_time_prelimbathyhydrothreshmet],file)
# with open(picklefile_dir+'set_id_tokeep_20Feb2025.pickle', 'rb') as file:
#     set_ii_prelimbathyhydrothreshmet,set_time_prelimbathyhydrothreshmet = pickle.load(file)



########################## Interp sets of bathy in time to fill gaps ##########################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_26Nov2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'set_id_tokeep_20Feb2025_Nlook96.pickle', 'rb') as file:
#     set_ii_prelimbathyhydrothreshmet, set_time_prelimbathyhydrothreshmet = pickle.load(file)
with open(picklefile_dir+'data_fullspan_addBlendedLidarBathy.pickle', 'rb') as file:
   data_fullspan = pickle.load(file)
   time_fullspan = data_fullspan["fullspan_time"]
   topobathy_fullspan = data_fullspan["fullspan_lidarbathy_blend_20Feb2025"]

num_datasets = set_ii_prelimbathyhydrothreshmet.size
Nlook = 4*24
nx = lidar_xFRF.size

topobathy_prexshoreinterp = np.empty(shape=topobathy_fullspan.shape)
topobathy_prexshoreinterp[:] = np.nan
topobathy_postxshoreinterp = np.empty(shape=topobathy_fullspan.shape)
topobathy_postxshoreinterp[:] = np.nan
for jj in np.arange(num_datasets):

    setjj = set_ii_prelimbathyhydrothreshmet[jj].astype(int)
    ttlook = np.arange(setjj,setjj+Nlook)
    timeslice = time_fullspan[ttlook]
    topobathy = topobathy_fullspan[:,ttlook]
    topobathy_prexshoreinterp[:,ttlook] = topobathy[:]
    topobathy_postxshoreinterp[:,ttlook] = topobathy[:]

    for ii in np.arange(nx):
        xshore_slice = topobathy[ii,:]
        percent_avail = sum(~np.isnan(xshore_slice))/Nlook
        if percent_avail >= 0.66:
            tin = np.arange(0,Nlook)
            zin = xshore_slice
            tin = tin[~np.isnan(zin)]
            zin = zin[~np.isnan(zin)]
            zout = np.interp(np.arange(0,Nlook),tin,zin)
            topobathy_postxshoreinterp[ii,ttlook] = zout

fig, ax = plt.subplots()
ax.plot(topobathy_prexshoreinterp,'.')
# vplot = topobathy_prexshoreinterp[:,ttlook]
# XX, TT = np.meshgrid(lidar_xFRF, ttlook)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(vplot.T, vplot.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=1, c=zz, cmap='viridis')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('z [m]')
# ax.set_xlabel('x [m, FRF]')
# ax.set_ylabel('time')
fig, ax = plt.subplots()
ax.plot(topobathy_postxshoreinterp,'.')
# vplot = topobathy_postxshoreinterp[:,ttlook]
# XX, TT = np.meshgrid(lidar_xFRF, ttlook)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(vplot.T, vplot.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=1, c=zz, cmap='viridis')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('z [m]')
# ax.set_xlabel('x [m, FRF]')
# ax.set_ylabel('time')

# Compare pre- and post-interp methods
yplot1 = np.nansum(~np.isnan(topobathy_prexshoreinterp),axis=1)
yplot2 = np.nansum(~np.isnan(topobathy_postxshoreinterp),axis=1)
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot1,label='1. pre x-shore interp')
ax.plot(lidar_xFRF,yplot2,label='2. post x-shore interp')
ax.legend()
plt.grid()
ax.set_ylabel('num avail.')
ax.set_xlabel('xFRF [m]')

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'topobathy_blendedBathyLidar_xshoreInterp_20Feb2025.pickle', 'wb') as file:
#      pickle.dump([topobathy_prexshoreinterp, topobathy_postxshoreinterp],file)



########################## Fill hydro gaps ##########################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'set_id_tokeep_20Feb2025_Nlook96.pickle', 'rb') as file:
#     set_ii_prelimbathyhydrothreshmet, set_time_prelimbathyhydrothreshmet = pickle.load(file)
with open(picklefile_dir+'data_fullspan_addBlendedLidarBathy.pickle', 'rb') as file:
   data_fullspan = pickle.load(file)
   time_fullspan = data_fullspan["fullspan_time"]
   topobathy_fullspan = data_fullspan["fullspan_lidarbathy_blend_20Feb2025"]
   watlev_fullspan = data_fullspan["fullspan_tidegauge"]
   Hs8m_fullspan = data_fullspan["fullspan_Hs_8m"]
   Tp8m_fullspan = data_fullspan["fullspan_Tp_8m"]
   dir8m_fullspan = data_fullspan["fullspan_wavedir_8m"]

num_datasets = set_ii_prelimbathyhydrothreshmet.size
Nlook = 4*24
nanflag_hydro = np.zeros((num_datasets,4))
hydro_datasetsForML = np.empty((num_datasets,Nlook,4))
for jj in np.arange(num_datasets-1):

    setjj = set_ii_prelimbathyhydrothreshmet[jj].astype(int)
    ttlook = np.arange(setjj,setjj+Nlook)

    # get water level
    ds_watlev = np.squeeze(watlev_fullspan[ttlook])
    hydro_datasetsForML[jj,:,0] = ds_watlev
    nanflag_hydro[jj,0] = sum(np.isnan(ds_watlev))
    if (~np.isnan(ds_watlev[-1])) & (~np.isnan(ds_watlev[0])):
        xtmp = np.arange(ds_watlev.size)
        ytmp = ds_watlev
        xin = xtmp[~np.isnan(ytmp)]
        yin = ytmp[~np.isnan(ytmp)]
        cs = CubicSpline(xin, yin, bc_type='natural')
        ynew = cs(xtmp)
        hydro_datasetsForML[jj, :, 0] = ynew
        nanflag_hydro[jj, 0] = sum(np.isnan(ynew))
    # get wave height
    ds_Hs = np.squeeze(Hs8m_fullspan[ttlook])
    hydro_datasetsForML[jj, :, 1] = ds_Hs
    nanflag_hydro[jj, 1] = sum(np.isnan(ds_Hs))
    if (nanflag_hydro[jj, 1] > 0) & (nanflag_hydro[jj, 1] <= 5):
        if (~np.isnan(ds_Hs[-1])) & (~np.isnan(ds_Hs[0])):
            xtmp = np.arange(ds_Hs.size)
            ytmp = ds_Hs
            xin = xtmp[~np.isnan(ytmp)]
            yin = ytmp[~np.isnan(ytmp)]
            f = interp1d(xin, yin)
            ynew = f(xtmp)
            hydro_datasetsForML[jj, :, 1] = ynew
            nanflag_hydro[jj, 1] = sum(np.isnan(ynew))
    # get wave period
    ds_Tp = np.squeeze(Tp8m_fullspan[ttlook])
    hydro_datasetsForML[jj, :, 2] = ds_Tp
    nanflag_hydro[jj, 2] = sum(np.isnan(ds_Tp))
    if (nanflag_hydro[jj, 2] > 0) & (nanflag_hydro[jj, 2] <= 5):
        if (~np.isnan(ds_Tp[-1])) & (~np.isnan(ds_Tp[0])):
            xtmp = np.arange(ds_Tp.size)
            ytmp = ds_Tp
            xin = xtmp[~np.isnan(ytmp)]
            yin = ytmp[~np.isnan(ytmp)]
            f = interp1d(xin, yin)
            ynew = f(xtmp)
            hydro_datasetsForML[jj, :, 2] = ynew
            nanflag_hydro[jj, 2] = sum(np.isnan(ynew))
    # wave direction
    ds_wdir = np.squeeze(dir8m_fullspan[ttlook])
    hydro_datasetsForML[jj, :, 3] = ds_wdir
    nanflag_hydro[jj, 3] = sum(np.isnan(ds_wdir))
    if (nanflag_hydro[jj, 3] > 0) & (nanflag_hydro[jj, 3] <= 5):
        if (~np.isnan(ds_wdir[-1])) & (~np.isnan(ds_wdir[0])):
            xtmp = np.arange(ds_wdir.size)
            ytmp = ds_wdir
            xin = xtmp[~np.isnan(ytmp)]
            yin = ytmp[~np.isnan(ytmp)]
            f = interp1d(xin, yin)
            ynew = f(xtmp)
            hydro_datasetsForML[jj, :, 3] = ynew
            nanflag_hydro[jj, 3] = sum(np.isnan(ynew))

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir + 'hydro_datasetsForML_20Feb2025.pickle', 'wb') as file:
#     pickle.dump([hydro_datasetsForML,nanflag_hydro], file)



########################## Ok, check the profiles for adequate length ##########################

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir+'topobathy_blendedBathyLidar_xshoreInterp_20Feb2025.pickle', 'rb') as file:
#      _, topobathy_postxshoreinterp = pickle.load(file)
# with open(picklefile_dir+'set_id_tokeep_20Feb2025_Nlook96.pickle', 'rb') as file:
#     set_ii_prelimbathyhydrothreshmet, set_time_prelimbathyhydrothreshmet = pickle.load(file)

# how many sets could we do if we set length requirement to xshore-prof?
Lmin = 75
Xstart = 50
Xend = 190
iistart = np.where(abs(lidar_xFRF-Xstart) == np.nanmin(abs(lidar_xFRF-Xstart)))[0]
iiend = np.where(abs(lidar_xFRF-Xend) == np.nanmin(abs(lidar_xFRF-Xend)))[0]

num_datasets = set_ii_prelimbathyhydrothreshmet.size
Nlook = 4*24
numprofs_at_xstart = np.zeros(shape=set_ii_prelimbathyhydrothreshmet.shape)*np.nan
numprofs_at_xend = np.zeros(shape=set_ii_prelimbathyhydrothreshmet.shape)*np.nan
topobathy_nansPerSet = np.zeros(shape=set_ii_prelimbathyhydrothreshmet.shape)*np.nan
topobathy_fillSmallGaps = np.zeros(shape=topobathy_postxshoreinterp.shape)*np.nan
for jj in np.arange(num_datasets):

    setjj = set_ii_prelimbathyhydrothreshmet[jj].astype(int)
    ttlook = np.arange(setjj, setjj + Nlook)
    topobathy = topobathy_postxshoreinterp[:,ttlook]
    numprofs_at_xstart[jj] = np.sum(~np.isnan(topobathy[iistart,:]))
    numprofs_at_xend[jj] = np.sum(~np.isnan(topobathy[iiend,:]))
    # calculate the number of nans in each of the sets
    tmp_nantotal = np.nansum(np.isnan(topobathy[np.arange(iistart,iiend),:]))
    topobathy_nansPerSet[jj] = tmp_nantotal
    topobathy_filled = topobathy[:]
    if (tmp_nantotal > 0) & (tmp_nantotal <= 20):
        for ii in np.arange(iistart,iiend):
            xv = np.arange(Nlook)
            yv = topobathy[ii,:]
            topobathy_filled[ii,:] = interpolate_with_max_gap(xv, yv, yv, max_gap=5, orig_x_is_sorted=False, target_x_is_sorted=False)
        tmp_nantotal = np.nansum(np.isnan(topobathy_filled[np.arange(iistart, iiend), :]))
        topobathy_nansPerSet[jj] = tmp_nantotal
    topobathy_fillSmallGaps[:,ttlook] = topobathy_filled


fig, ax = plt.subplots()
ax.plot(numprofs_at_xstart,'x',label='# profiles at Xstart')
ax.plot(numprofs_at_xend,'.',label='# profiles at Xend')
plt.grid()
fig, ax = plt.subplots()
yplot = numprofs_at_xstart/Nlook + numprofs_at_xend/Nlook
ax.plot(yplot,'x',label='percent avail. at Xstart & Xend')
ax.plot(topobathy_nansPerSet,'.',label='num. nans per set')
plt.grid()


########################## Determine quality sets based on previous calcs ##########################

num_notopobathynans = np.nansum((topobathy_nansPerSet == 0))
num_nohydronans = np.nansum(np.nansum(nanflag_hydro,axis=1) == 0)
num_nonans_hydroANDbathy = np.nansum((np.nansum(nanflag_hydro,axis=1) == 0) & (topobathy_nansPerSet == 0))

print(str(num_notopobathynans)+' sets with no topo-bathy gaps for Nlook = '+str(Nlook)+' hrs')
print(str(num_nohydronans)+' sets with no hydro gaps for Nlook = '+str(Nlook)+' hrs')
print(str(num_nonans_hydroANDbathy)+' sets with no hydro OR topo-bathy gaps for Nlook = '+str(Nlook)+' hrs')

set_grab_final = (np.nansum(nanflag_hydro,axis=1) == 0) & (topobathy_nansPerSet == 0)
set_ii_final_bathyhydrothreshment = set_ii_prelimbathyhydrothreshmet[set_grab_final].astype(int)
set_time_final_bathyhydrothreshment = set_time_prelimbathyhydrothreshmet[set_grab_final]

hydro_MLinput_final = hydro_datasetsForML[set_grab_final,:,:]
profileIDs_ML_final = np.empty((sum(set_grab_final),Nlook))*np.nan
profileTimes_ML_final = np.empty((sum(set_grab_final),Nlook))*np.nan
for jj in np.arange(sum(set_grab_final)):
    profID1 = set_ii_final_bathyhydrothreshment[jj].astype(int)
    profIDs = np.arange(profID1,profID1+Nlook)
    profileIDs_ML_final[jj,:] = profIDs
    profileTimes_ML_final[jj,:] = time_fullspan[profIDs]
topobathy_ML_final = topobathy_fillSmallGaps[:]


picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
# with open(picklefile_dir + 'topobathyhydro_ML_final_20Feb2025_Nlook'+str(Nlook)+'.pickle', 'wb') as file:
#     pickle.dump([time_fullspan,lidar_xFRF,profileIDs_ML_final,profileTimes_ML_final,hydro_MLinput_final,topobathy_ML_final], file)