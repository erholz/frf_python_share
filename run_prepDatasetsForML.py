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



## OPEN DICTS
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
with open(picklefile_dir+'data_poststorm_sliced.pickle','rb') as file:
    data_poststorm_all = pickle.load(file)
with open(picklefile_dir+'data_fullspan.pickle','rb') as file:
    data_fullspan = pickle.load(file)
    time_fullspan = data_fullspan["fullspan_time"]
    elev_fullspan = data_fullspan["fullspan_bathylidar_10Dec24"]
    watlev_fullspan = data_fullspan["fullspan_tidegauge"]
    Hs8m_fullspan = data_fullspan["fullspan_Hs_8m"]
    Tp8m_fullspan = data_fullspan["fullspan_Tp_8m"]
    dir8m_fullspan = data_fullspan["fullspan_wavedir_8m"]
    Hs17m_fullspan = data_fullspan["fullspan_Hs_17m"]
    Tp17m_fullspan = data_fullspan["fullspan_Tp_17m"]
    dir17m_fullspan = data_fullspan["fullspan_wavedir_17m"]
    elev2p_fullspan = data_fullspan["fullspan_elev2p"]
    lidarwg_fullspan = data_fullspan["fullspan_lidargauge_110"]




## Load dicts and find N-day series with adequate data coverage
Nlook = 4*24                   # look through Nlook hours to quantify data availability
profelev_numhrly_thresh = 0.75  # use to count percent of days (by x-locs) where THRESH % profile data available over Nlook
watlev_perctotal_thresh = 0.9
waves_perctotal_thresh = 0.9
lidarwg_perctotal_thresh = 0.9
set_start_timekeep = np.empty(0)
set_start_iikeep = np.empty(0)
set_wavethreshmet = np.empty(0)
set_watlevthreshmet = np.empty(0)
set_lidarwgthreshmet = np.empty(0)
set_profhrlythreshmet = np.empty(lidar_xFRF.size,)

## Go through profiles
for jj in np.arange(len(data_poststorm_all)):
# for jj in np.arange(3):

    # get topobathy data
    timeslice = data_poststorm_all["data_poststorm" + str(jj)]["poststorm_time"]
    topobathy = data_poststorm_all["data_poststorm" + str(jj)]["poststorm_bathylidar_10Dec24"]

    # get other variables
    tidegauge = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_tidegauge"]
    Hs_8m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_Hs_8m"]
    lidar_wg = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_lidargauge_110"]
    timeslice = data_poststorm_all["data_poststorm" + str(jj)]["poststorm_time"]

    if (timeslice.size - Nlook) > 1:

        # now go through all times in post-storm_jj
        for tt in np.arange(timeslice.size - Nlook):

            # isolate data for Nlook hrs
            ttlook = np.arange(tt,tt+Nlook)
            wavelook = Hs_8m[ttlook]
            watlevlook = tidegauge[ttlook]
            lidarwglook = lidar_wg[ttlook,0]
            topobathy_look = topobathy[:,ttlook]
            set_start = timeslice[ttlook[0]]
            set_end = timeslice[ttlook[-1]]
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
                set_start_timekeep = np.append(set_start_timekeep, set_start)
                set_start_iikeep = np.append(set_start_iikeep, set_start_ii)
                # set the x-locs where condition met
                set_profhrlythreshmet = np.vstack((set_profhrlythreshmet,ytmp))
                # save lidarwg condition (1/0)
                if set_lidarwg_cover >= lidarwg_perctotal_thresh:
                    threshmet = 1
                else:
                    threshmet = 0
                set_lidarwgthreshmet = np.append(set_lidarwgthreshmet,threshmet)
                # save wave condition (1/0)
                if set_wave_cover >= waves_perctotal_thresh:
                    threshmet = 1
                else:
                    thresmet = np.nan
                set_wavethreshmet = np.append(set_wavethreshmet,threshmet)
                # save watlev condition (1/0)
                if set_watlev_cover >= watlev_perctotal_thresh:
                    threshmet = 1
                else:
                    thresmet = np.nan
                set_watlevthreshmet = np.append(set_watlevthreshmet,threshmet)


# clean up initialized datasets...
plot_profelev_cover_hrly = set_profhrlythreshmet[1:, :]
plot_profelev_cover_hrly[plot_profelev_cover_hrly == 0] = np.nan
plot_watlevthreshmet = set_watlevthreshmet[:]
plot_watlevthreshmet[plot_watlevthreshmet == 0] = np.nan
plot_wavethreshmet = set_wavethreshmet[:]
plot_wavethreshmet[plot_wavethreshmet == 0] = np.nan
plot_lidarwgthreshmet = set_lidarwgthreshmet[:]
set_lidarwgthreshmet[set_lidarwgthreshmet == 0] = np.nan
plot_start_timekeep = set_start_timekeep[:]
plot_start_iikeep = set_start_iikeep[:]

# # now do some plotting...
# fig = plt.figure()
# fig.set_size_inches(14.2, 7.1)
# scattersz = 3
# ax1 = fig.add_subplot(3, 1, 1)
# tplot = pd.to_datetime(plot_start_timekeep, unit='s', origin='unix')
# ax1.plot(tplot,1*plot_watlevthreshmet,'+',label='WL '+ str(100*watlev_perctotal_thresh)+'% avail')
# ax1.plot(tplot,2*plot_wavethreshmet,'.',label='waves '+ str(100*waves_perctotal_thresh)+'% avail')
# ax1.plot(tplot,3*plot_lidarwgthreshmet,'x',label='lidarwg '+ str(100*lidarwg_perctotal_thresh)+'% avail')
# ax1.legend()
# # xfmt = md.DateFormatter('%m/%y')
# # ax1.xaxis.set_major_formatter(xfmt)
#
# ax2 = fig.add_subplot(3, 1, 2)
# xplot = lidar_xFRF
# TT, XX = np.meshgrid(tplot,xplot)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(plot_profelev_cover_hrly.T, plot_profelev_cover_hrly.T.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# ph = ax2.scatter(tt, xx, s=scattersz, c=zz)
# ax2.set_ylabel('xFRF [m]')
#
# # subplot 3 , plot the profile coverage, with contour pos and ranges
# mwl = -0.13
# zero = 0
# mhw = 3.6
# dune_toe = 3.22
# cont_elev = np.array([mwl, zero, dune_toe, mhw])  # np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
# zinput = elev_fullspan[:,plot_start_iikeep.astype(int)]
# tinput = plot_start_timekeep
# cont_ts, cmean, cstd = create_contours(zinput.T, tinput, lidar_xFRF, cont_elev)
# ax3 = fig.add_subplot(3,1,3)
# yplot = np.nansum(plot_profelev_cover_hrly,axis=0)
# ax3.plot(lidar_xFRF, yplot, 'k')
# cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size))
# for cc in np.arange(cont_elev.size):
#     ax3.plot([0, 0] + cmean[cc], [0, 99999999], label='z = ' + str(cont_elev[cc]) + ' m', color=cmap[cc, :])
# for cc in np.arange(cont_elev.size):
#     left, bottom, width, height = (cmean[cc] - cstd[cc], 0, cstd[cc] * 2, 99999999)
#     patch = plt.Rectangle((left, bottom), width, height, alpha=0.1, color=cmap[cc, :])
#     ax3.add_patch(patch)
# ax3.set_xlabel('xFRF [m]')
# ax3.set_ylabel('num sets w/ consistent data')
# ax3.set_ylim(0,15000)

## Find the profiles where profiles are "good", long "enough", and we have watlev & wave data
dx = 0.1
length_profdatavail = np.nansum(plot_profelev_cover_hrly,axis=1)
length_criterion = (length_profdatavail >= 25/dx)       # profiles for ML analysis must have minimum length (for now)
wave_criterion = (plot_wavethreshmet == 1)
watlev_criterion = (plot_watlevthreshmet == 1)
set_id_tokeep = np.where( length_criterion & wave_criterion & watlev_criterion)[0]


picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_10Dec2024/'
# with open(picklefile_dir+'set_id_tokeep_14Dec2024.pickle', 'wb') as file:
#     pickle.dump([set_id_tokeep,plot_start_iikeep],file)
# with open(picklefile_dir+'set_id_tokeep_14Dec2024.pickle', 'rb') as file:
#     set_id_tokeep, plot_start_iikeep = pickle.load(file)

## save data from these times as unique dictionary
datasets_ML = {}
for jj in np.arange(set_id_tokeep.size):
    # initialize...
    outputname = 'dataset_' + str(jj)
    exec('datasets_ML["' + outputname + '"] = {}')
    # get indeces for fullspan...
    ii_start = int(plot_start_iikeep[set_id_tokeep[jj]])
    ii_end = ii_start + Nlook
    ii_foroutput = np.arange(ii_start,ii_end)

    # time, waterlevel, wave8m, wave17m, lidarwg_110, elev2p, etc
    exec('datasets_ML["' + outputname + '"]["set_timeslice"] = time_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_waterlevel"] = watlev_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_Hs8m"] = Hs8m_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_Tp8m"] = Tp8m_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_dir8m"] = dir8m_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_Hs17m"] = Hs17m_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_Tp17m"] = Tp17m_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_dir17m"] = dir17m_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_elev2p"] = elev2p_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_lidarwg"] = lidarwg_fullspan[ii_foroutput]')
    exec('datasets_ML["' + outputname + '"]["set_topobathy"] = elev_fullspan[:,ii_foroutput]')

# with open(picklefile_dir+'datasets_ML_14Dec2024.pickle', 'wb') as file:
#     pickle.dump(datasets_ML,file)














