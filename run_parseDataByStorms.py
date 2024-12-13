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


## Load temporally aligned data - need to add lidarelev_fullspan
# picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
# picklefile_dir = './'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
with open(picklefile_dir+'bathylidar_combo.pickle','rb') as file:
    lidar_xFRF,bathylidar_combo = pickle.load(file)
with open(picklefile_dir+'waves_8m&17m_2015_2024.pickle','rb') as file:
    [data_wave8m,data_wave17m,data_wave8m_filled] = pickle.load(file)
with open(picklefile_dir+'stormHs95_Over12Hours.pickle','rb') as f:
    output = pickle.load(f)
list(output)
storm_start = np.array(output['startTimeStormList'])
storm_end = np.array(output['endTimeStormList'])
storm_startWIS = np.array(output['startTimeStormListWIS'])
storm_endWIS = np.array(output['endTimeStormListWIS'])





# Load and process wave data
Hs_8m_fullspan = data_wave8m_filled[:,0]
Tp_8m_fullspan = data_wave8m_filled[:,1]
dir_8m_fullspan = data_wave8m_filled[:,2]
Hs_17m_fullspan = data_wave17m[:,0]
Tp_17m_fullspan = data_wave17m[:,1]
dir_17m_fullspan = data_wave17m[:,2]

fig, ax = plt.subplots()
ax.plot(time_fullspan,)


cuspfile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/cusp_presence/'
with open(cuspfile_dir+'cuspTimes.pickle', 'rb') as file:
    datload = pickle.load(file)
cusp_time = np.array(datload['timeCusps'])
cusp_presence = np.ones(shape=cusp_time.shape)

cusp_fullspan = np.empty(shape=time_fullspan.shape)
cusp_fullspan[:] = 0
for tt in np.arange(cusp_time.size):
    iiclose = np.where(abs(cusp_time[tt] - time_fullspan) == np.nanmin(abs(cusp_time[tt] - time_fullspan)))[0]
    if iiclose.size > 1:
        iiclose = iiclose[0]
    cusp_fullspan[iiclose] = 1

# Plot cusp times with storms --> shows cusps ~occur during non-stormy conditions
fig, ax = plt.subplots()
tplot = pd.to_datetime(cusp_time, unit='s', origin='unix')
ax.plot(cusp_time,cusp_presence,'x')
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
ax.plot(time_fullspan,cusp_fullspan,'+')
for jj in np.arange(len(storm_start)):
    yplot = [0,2]
    ax.plot([storm_start[jj],storm_start[jj]],yplot,'g')
    ax.plot([storm_end[jj],storm_end[jj]],yplot,'r')
for jj in np.arange(len(storm_startWIS)):
    yplot = [0,2]
    ax.plot([storm_startWIS[jj],storm_startWIS[jj]],yplot,'b:')
    ax.plot([storm_endWIS[jj],storm_endWIS[jj]],yplot,'m:')
ax.set_ylim((0.5,1.5))


## Remove storms outside of general time of interest
storm_start = storm_start[(storm_start >= time_fullspan[0]) & (storm_start < time_fullspan[-1])]
storm_end = storm_end[(storm_end > time_fullspan[0]) & (storm_end <= time_fullspan[-1])]
storm_startWIS = storm_startWIS[(storm_startWIS >= time_fullspan[0]) & (storm_startWIS < time_fullspan[-1])]
storm_endWIS = storm_endWIS[(storm_endWIS > time_fullspan[0]) & (storm_endWIS <= time_fullspan[-1])]

## Start by identifing times when FRF or WIS data says stormy
storm_flag = np.empty(shape=time_fullspan.shape)        # BINARY - stormy == 1, calm/non-stormy = nan
storm_flag[:] = 0
for jj in np.arange(len(storm_start)):
    tt_during_storm = (time_fullspan >= storm_start[jj]) & (time_fullspan <= storm_end[jj])
    storm_flag[tt_during_storm] = 1
for jj in np.arange(len(storm_startWIS)):
    tt_during_storm = (time_fullspan >= storm_startWIS[jj]) & (time_fullspan <= storm_endWIS[jj])
    storm_flag[tt_during_storm] = 1
# fig, ax = plt.subplots()
# ax.plot(time_fullspan,storm_flag,'o')
# for jj in np.arange(len(storm_start)):
#     yplot = [0,2]
#     ax.plot([storm_start[jj],storm_start[jj]],yplot,'g')
#     ax.plot([storm_end[jj],storm_end[jj]],yplot,'r')
# for jj in np.arange(len(storm_startWIS)):
#     yplot = [0,2]
#     ax.plot([storm_startWIS[jj],storm_startWIS[jj]],yplot,'c')
#     ax.plot([storm_endWIS[jj],storm_endWIS[jj]],yplot,'m')
# ax.set_title('1 == Stormy, 0 == Calm/Non-stormy')

storm_timeend_all = []
storm_timestart_all = []
storm_iiend_all = []
storm_iistart_all = []
storm_flag[storm_flag == 0] = -1
iicross = np.where(storm_flag[1:]*storm_flag[0:-1] < 0)[0]
for jj in np.arange(iicross.size):
    if (storm_flag[iicross[jj]] == -1) & (storm_flag[iicross[jj]+1] == 1):
        storm_timestart_all = np.append(storm_timestart_all,time_fullspan[iicross[jj]])
        storm_iistart_all = np.append(storm_iistart_all,int(iicross[jj]+1))
    elif (storm_flag[iicross[jj]] == 1) & (storm_flag[iicross[jj]+1] == -1):
        storm_timeend_all = np.append(storm_timeend_all, time_fullspan[iicross[jj]])
        storm_iiend_all = np.append(storm_iiend_all,int(iicross[jj]+1))
    else:
        print('help')
fig, ax = plt.subplots()
ax.plot(time_fullspan,storm_flag,'o')
for jj in np.arange(len(storm_timestart_all)):
    yplot = [-1,2]
    ax.plot([storm_timestart_all[jj],storm_timestart_all[jj]],yplot,'g')
for jj in np.arange(len(storm_timeend_all)):
    yplot = [-1,2]
    ax.plot([storm_timeend_all[jj],storm_timeend_all[jj]],yplot,'r')
ax.set_title('1 == Stormy, -1 == Calm/Non-stormy')

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'bathylidar_fill.pickle','rb') as file:
    lidar_xFRF,bathylidar_fill = pickle.load(file)

## Create dicts for each post-storm snippet of data
data_poststorm_all = {}
for jj in np.arange(storm_iiend_all.size):
    outputname = 'data_poststorm'+str(jj)
    exec('data_poststorm_all["' + outputname + '"] = {}')
    # exec(outputname + '= {}')
    if jj == storm_iiend_all.size-1:
        ii_foroutput = np.arange(int(storm_iiend_all[jj]),time_fullspan.size)
    else:
        ii_foroutput = np.arange(int(storm_iiend_all[jj]), int(storm_iistart_all[jj+1]))
    exec('data_poststorm_all["' + outputname + '"]["poststorm_time"] = time_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_Hs_8m"] = Hs_8m_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_Tp_8m"] = Tp_8m_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_wavedir_8m"] = dir_8m_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_Hs_17m"] = Hs_17m_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_Tp_17m"] = Tp_17m_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_wavedir_17m"] = dir_17m_fullspan[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_tidegauge"] = data_tidegauge[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_elev2p"] = data_lidar_elev2p[ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_lidargauge_110"] = data_lidarwg110[ii_foroutput]')
    # exec('data_poststorm_all["' + outputname + '"]["poststorm_bathylidar_10Dec24"] = bathylidar_combo[:,ii_foroutput]')
    exec('data_poststorm_all["' + outputname + '"]["poststorm_bathylidar_10Dec24"] = bathylidar_fill[:,ii_foroutput]')

# Save all the data in shareable dict file
data_fullspan = {}
data_fullspan["fullspan_time"] = time_fullspan
data_fullspan["fullspan_Hs_8m"] = Hs_8m_fullspan
data_fullspan["fullspan_Tp_8m"] = Tp_8m_fullspan
data_fullspan["fullspan_wavedir_8m"] = dir_8m_fullspan
data_fullspan["fullspan_Hs_17m"] = Hs_17m_fullspan
data_fullspan["fullspan_Tp_17m"] = Tp_17m_fullspan
data_fullspan["fullspan_wavedir_17m"] = dir_17m_fullspan
data_fullspan["fullspan_tidegauge"] = data_tidegauge
data_fullspan["fullspan_elev2p"] = data_lidar_elev2p
data_fullspan["fullspan_lidargauge_110"] = data_lidarwg110
data_fullspan["fullspan_bathylidar_10Dec24"] = bathylidar_fill


#
## SAVE DICTS
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# with open(picklefile_dir+'data_poststorm_sliced.pickle','wb') as file:
#     pickle.dump(data_poststorm_all, file)
# with open(picklefile_dir+'data_fullspan.pickle','wb') as file:
#     pickle.dump(data_fullspan, file)



################### NOW OPEN DICTS AND DO ANALYSIS ON AVAILABLE DATA ###################

## OPEN DICTS
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
with open(picklefile_dir+'data_poststorm_sliced.pickle','rb') as file:
    data_poststorm_all = pickle.load(file)
with open(picklefile_dir+'data_fullspan.pickle','rb') as file:
    data_fullspan = pickle.load(file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]

## Plot available data for each post-storm period
for jj in np.arange(len(data_poststorm_all)):

    # get topobathy data
    timeslice = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_time"]
    tplot = pd.to_datetime(timeslice, unit='s', origin='unix')
    topobathy = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_bathylidar_10Dec24"]
    topobathy_fracavail = np.sum(~np.isnan(topobathy),axis=1)/timeslice.size
    topobathy_fracavail[topobathy_fracavail <= 0.01] = np.nan

    # calculate contour position
    mwl = -0.13
    zero = 0
    mhw = 3.6
    dune_toe = 3.22
    cont_elev = np.array([mwl, zero, dune_toe, mhw])  # np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
    cont_ts, cmean, cstd = create_contours(topobathy.T, timeslice, lidar_xFRF, cont_elev)

    # get other variables
    tidegauge = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_tidegauge"]
    Hs_8m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_Hs_8m"]
    Tp_8m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_Tp_8m"]
    dir_8m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_wavedir_8m"]
    Hs_17m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_Hs_17m"]
    Tp_17m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_Tp_17m"]
    dir_17m = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_wavedir_17m"]
    lidar_wg = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_lidargauge_110"]
    lidar_elev2p = data_poststorm_all["data_poststorm"+str(jj)]["poststorm_elev2p"]

    # calculate beach slope near tidal elevation (~SWL)
    beachslope = np.empty(shape=timeslice.shape)
    beachslope[:] = np.nan
    dx = 0.1
    # fig, ax = plt.subplots()
    for tt in np.arange(timeslice.size):
        g = 9.81
        h = tidegauge[tt]
        T = Tp_8m[tt]
        if ~np.isnan(T) & ~np.isnan(h):
            hmatch, Tmatch, k, L, C, Cg, n = wavenumber(h, T, g)
            L0 = g * T * T / (2 * np.pi)
            ztmp = np.abs(topobathy[:, tt])
            ix_stillwater = np.where(abs(ztmp - h) == np.nanmin(abs(ztmp - h)))[0]
            if len(ix_stillwater) > 0:
                if isinstance(ix_stillwater, np.ndarray):
                    ix_stillwater = ix_stillwater[0]
                halfspan = np.floor(L / 2).astype(int)
                iix = ix_stillwater + np.arange(-halfspan, halfspan)
                profiix = topobathy[iix, tt]
                # ax.plot(lidar_xFRF,ztmp)
                # ax.plot(lidar_xFRF[iix],profiix,'k:')
                if sum(~np.isnan(profiix)) >= 10:
                    profiix = profiix[~np.isnan(profiix)]
                    beachslope[tt] = abs((profiix[-1] - profiix[0]) / (profiix.size * dx))
    # fig, ax = plt.subplots()
    # ax.plot(beachslope,'o')
    # ax.plot([0, timeslice.size], [1/25, 1/25])
    # ax.plot([0, timeslice.size], [1/50, 1/50])
    # ax.plot([0, timeslice.size], [1/100, 1/100])
    # ax.plot([0, timeslice.size], [1/200, 1/200])
    # ax.plot([0, timeslice.size], [1/250, 1/250])

    # create figure (2 panels)
    # fig, (ax1, ax2) = plt.subplots(2)
    fig = plt.figure()
    fig.set_size_inches(8.5,5)

    # first plot - topobathy data availale + contour avg positions
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(lidar_xFRF,topobathy_fracavail,'k')
    cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size ))
    for cc in np.arange(cont_elev.size):
        ax1.plot([0, 0] + cmean[cc], [0, 1], label='z = ' + str(cont_elev[cc]) + ' m',color=cmap[cc, :])
    for cc in np.arange(cont_elev.size):
        left, bottom, width, height = (cmean[cc] - cstd[cc], 0, cstd[cc] * 2, 1)
        patch = plt.Rectangle((left, bottom), width, height, alpha=0.1, color=cmap[cc, :])
        ax1.add_patch(patch)
    ax1.legend()
    ax1.set_xlabel('xFRF [m]')
    ax1.set_title(tplot[0])
    ax1.set_ylabel('fraction available [-]')
    ax1.set_xlim((40,150))

    # second figures, plot scatter(?) of available explanatory data...
    # bar - xc_MWL
    bar1 = ~np.isnan(cont_ts[0,:])
    # bar - xc_0
    bar2 = ~np.isnan(cont_ts[1, :])
    # bar - xc_MHW
    bar3 = ~np.isnan(cont_ts[2, :])
    # bar - xc_dunetoe
    bar4 = ~np.isnan(cont_ts[3, :])
    # bar - beach slope at WSE(t)
    bar5 = ~np.isnan(beachslope)
    # bar - Hs/Tp/dir at 8m
    tmp = (~np.isnan(Hs_8m)).astype(int) + (~np.isnan(Tp_8m)).astype(int) + (~np.isnan(dir_8m)).astype(int)
    bar6 = (tmp >= 2)
    # bar - Hs/Tp/dir at 17m
    tmp = (~np.isnan(Hs_17m)).astype(int) + (~np.isnan(Tp_17m)).astype(int) + (~np.isnan(dir_17m)).astype(int)
    bar7 = (tmp >= 2)
    # bar - Hs/Tp/dir at lidarWG
    bar8 = ~np.isnan(lidar_wg[:,0])
    # bar - elev2p from lidarWG
    bar9 = ~np.isnan(lidar_elev2p[:,0])
    # bar - waterlevel
    bar10 = ~np.isnan(tidegauge[:,0])

    ax2 = fig.add_subplot(2, 1, 2)
    XX, YY = np.meshgrid(timeslice, np.arange(12))
    ZZ = np.empty(shape=XX.shape)
    ZZ[:] = np.nan
    ZZ[1, bar1] = 1*2
    ZZ[2, bar2] = 2*2
    ZZ[3, bar3] = 3*2
    ZZ[4, bar4] = 4*2
    ZZ[5, bar5] = 5*2
    ZZ[6, bar6] = 6*2
    ZZ[7, bar7] = 7*2
    ZZ[8, bar8] = 8*2
    ZZ[9, bar9] = 9*2
    ZZ[10, bar10] = 10 * 2
    ZZmasked = np.ma.array(ZZ, mask=np.isnan(ZZ))
    # surf = ax2.pcolormesh(ZZ)
    surf = ax2.pcolormesh(tplot,np.arange(12), ZZmasked)
    xfmt = md.DateFormatter('%m/%d')
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.set_yticks(np.arange(1,11), ['xc_MWL','xc_0','xc_MHW','xc_dunetoe','beach_slp','waves_8m','waves_17m','WG_110','elev2p','tide'])

    # save figure
    figpath = 'C:/Users/rdchlerh/PycharmProjects/frf_python_share/figs/data/poststorm_timeslices/'
    fig.savefig(figpath+'poststorm_'+str(jj)+'.png',dpi=300)  # save the figure to file
    plt.close(fig)




## Load dicts and find N-day series with adequate data coverage
Nlook = 4*24                   # look through Nlook hours to quantify data availability
profelev_numdaily_thresh = 2    # use to count percent of days (by x-locs) where atleast THRESH profile available during a day
profelev_numtidal_tresh = 2     # use to count percent of tidal cycles (by x-locs) where atleast THRESH profile available during a tidal cycle
profelev_perctotal_tresh = 0.5  # use to count x-locs where THRESH % profile data available over Nlook
profelev_numhrly_thresh = 0.75  # use to count percent of days (by x-locs) where THRESH % profile data available over Nlook

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
        # initialize the counting arrays
        set_start = np.empty(timeslice.size-Nlook)
        set_end = np.empty(timeslice.size-Nlook)
        set_watlev_cover = np.empty(timeslice.size-Nlook)
        set_wave_cover = np.empty(timeslice.size-Nlook)
        set_lidarwg_cover = np.empty(timeslice.size-Nlook)
        set_profelev_cover_total = np.empty(lidar_xFRF.size,)
        set_profelev_cover_daily = np.empty(lidar_xFRF.size,)
        set_profelev_cover_tidecyc = np.empty(lidar_xFRF.size,)
        set_profelev_cover_hrly = np.empty(lidar_xFRF.size,)

        # now go through all times in post-storm_jj
        for tt in np.arange(timeslice.size - Nlook):

            # isolate data for Nlook hrs
            ttlook = np.arange(tt,tt+Nlook)
            wavelook = Hs_8m[ttlook]
            watlevlook = tidegauge[ttlook]
            lidarwglook = lidar_wg[ttlook,0]
            topobathy_look = topobathy[:,ttlook]
            set_start[tt] = timeslice[ttlook[0]]
            set_end[tt] = timeslice[ttlook[-1]]

            # calculate the percent available for hydro
            set_watlev_cover[tt] = np.nansum(~np.isnan(watlevlook))/Nlook
            set_wave_cover[tt] = np.nansum(~np.isnan(wavelook)) / Nlook
            set_lidarwg_cover[tt] = np.nansum(~np.isnan(lidarwglook)) / Nlook

            # calculate total availability of topobathy over set time
            percavail_total = np.nansum(~np.isnan(topobathy_look),axis=1)/Nlook
            set_profelev_cover_total = np.vstack((set_profelev_cover_total,percavail_total))

            # day counter
            numdays = int(np.floor(Nlook/24))
            daycount = np.empty(shape=(topobathy_look.shape[0],numdays))
            daycount[:] = np.nan
            hrlyperc = np.empty(shape=(topobathy_look.shape[0],numdays))
            hrlyperc[:] = np.nan
            tmpii = 0
            for dd in np.arange(numdays):
                ztmp = topobathy_look[:,tmpii:tmpii+24]
                ytmp = np.nansum(~np.isnan(ztmp),axis=1)
                daycount[ytmp >= profelev_numdaily_thresh, dd] = 1
                hrlyperc[ytmp/24 >= profelev_numhrly_thresh, dd] = 1
                tmpii = tmpii+24
            percdaily_threshmet = np.nansum(daycount,axis=1)/numdays
            perchrly_threshmet = np.nansum(hrlyperc,axis=1)/numdays
            set_profelev_cover_daily = np.vstack((set_profelev_cover_daily,percdaily_threshmet))
            set_profelev_cover_hrly = np.vstack((set_profelev_cover_hrly,perchrly_threshmet))

            # tidal counter
            numtides = int(np.floor(Nlook / 12))
            tidecount = np.empty(shape=(topobathy_look.shape[0], numtides))
            tidecount[:] = np.nan
            tmpii = 0
            for dd in np.arange(numtides):
                ztmp = topobathy_look[:, tmpii:tmpii + 12]
                ytmp = np.nansum(~np.isnan(ztmp), axis=1)
                tidecount[ytmp >= profelev_numtidal_tresh, dd] = 1
                tmpii = tmpii + 12
            perctidal_threshmet = np.nansum(tidecount, axis=1)/numtides
            set_profelev_cover_tidecyc = np.vstack((set_profelev_cover_tidecyc, perctidal_threshmet))

        # clean up initialized datasets...
        plot_profelev_cover_total = set_profelev_cover_total[1:, :]
        plot_profelev_cover_daily = set_profelev_cover_daily[1:, :]
        plot_profelev_cover_tidecyc = set_profelev_cover_tidecyc[1:, :]
        plot_profelev_cover_hrly = set_profelev_cover_hrly[1:, :]
        plot_profelev_cover_total[plot_profelev_cover_total == 0] = np.nan
        plot_profelev_cover_daily[plot_profelev_cover_daily == 0] = np.nan
        plot_profelev_cover_tidecyc[plot_profelev_cover_tidecyc == 0] = np.nan
        plot_profelev_cover_hrly[plot_profelev_cover_hrly == 0] = np.nan


        # now do some plotting...
        fig = plt.figure()
        fig.set_size_inches(19, 9.5)
        scattersz = 3
        mycmap = plt.colormaps['rainbow'].resampled(5)

        # plot fraction hydro availability
        tplot = pd.to_datetime(set_start, unit='s', origin='unix')
        ax1 = fig.add_subplot(1, 5, 1)
        ax1.plot(set_watlev_cover,tplot,'+',label='WSE')
        ax1.plot(set_wave_cover,tplot,'.',label='waves')
        ax1.plot(set_lidarwg_cover,tplot,'x',label='lidar wg')
        yfmt = md.DateFormatter('%m/%d')
        format_str = '%m/%d'
        format_ = md.DateFormatter(format_str)
        ax1.yaxis.set_major_formatter(format_)
        ax1.yaxis.set_major_formatter(yfmt)
        fig.suptitle(str(tplot[0])+', Nlook = '+str(Nlook))
        ax1.legend()
        # ax1.set_xticks([])

        # plot scatter for "set_profelev_cover_total"
        xplot = lidar_xFRF
        XX, TT = np.meshgrid(xplot, tplot)
        timescatter = np.reshape(TT, TT.size)
        xscatter = np.reshape(XX, XX.size)
        zscatter = np.reshape(plot_profelev_cover_total, plot_profelev_cover_total.size)
        tt = timescatter[~np.isnan(zscatter)]
        xx = xscatter[~np.isnan(zscatter)]
        zz = zscatter[~np.isnan(zscatter)]
        ax2 = fig.add_subplot(1, 5, 2)
        ph = ax2.scatter(xx, tt, s=scattersz, c=zz, cmap=mycmap, vmin=0, vmax=1)
        # cbar = fig.colorbar(ph, ax=ax2)
        # cbar.set_label('frac avail')
        # ax2.yaxis.set_major_formatter(yfmt)
        ax2.set_xlabel('xFRF [m]')
        ax2.set_title('Total avail for each set \n of Nlook times')
        ax2.set_yticks([])

        # plot "plot_profelev_cover_daily"
        zscatter = np.reshape(plot_profelev_cover_daily, plot_profelev_cover_daily.size)
        tt = timescatter[~np.isnan(zscatter)]
        xx = xscatter[~np.isnan(zscatter)]
        zz = zscatter[~np.isnan(zscatter)]
        ax3 = fig.add_subplot(1, 5, 3)
        ph = ax3.scatter(xx, tt, s=scattersz, c=zz, cmap=mycmap, vmin=0, vmax=1)
        # cbar = fig.colorbar(ph, ax=ax3)
        # cbar.set_label('frac avail')
        # ax3.yaxis.set_major_formatter(yfmt)
        ax3.set_title('Frac of days w/ at least \n 2 returns/day in each set')
        ax3.set_yticks([])
        ax3.set_xlabel('xFRF [m]')

        # plot "plot_profelev_cover_tidecyc"
        zscatter = np.reshape(plot_profelev_cover_tidecyc, plot_profelev_cover_tidecyc.size)
        tt = timescatter[~np.isnan(zscatter)]
        xx = xscatter[~np.isnan(zscatter)]
        zz = zscatter[~np.isnan(zscatter)]
        ax4 = fig.add_subplot(1, 5, 4)
        ph = ax4.scatter(xx, tt, s=scattersz, c=zz, cmap=mycmap, vmin=0, vmax=1)
        # cbar = fig.colorbar(ph, ax=ax4)
        # cbar.set_label('frac avail')
        # ax4.yaxis.set_major_formatter(yfmt)
        ax4.set_title('Frac of tidecycs w/ at least \n 2 returns/cyc in each set')
        ax4.set_yticks([])
        ax4.set_xlabel('xFRF [m]')

        # plot "plot_profelev_cover_hrly"
        zscatter = np.reshape(plot_profelev_cover_hrly, plot_profelev_cover_hrly.size)
        tt = timescatter[~np.isnan(zscatter)]
        xx = xscatter[~np.isnan(zscatter)]
        zz = zscatter[~np.isnan(zscatter)]
        ax5 = fig.add_subplot(1, 5, 5)
        ph = ax5.scatter(xx, tt, s=scattersz, c=zz, cmap=mycmap, vmin=0, vmax=1)
        cbar = fig.colorbar(ph, ax=ax5)
        cbar.set_label('frac avail')
        # ax5.yaxis.set_major_formatter(yfmt)
        ax5.set_xlabel('xFRF [m]')
        ax5.set_yticks([])
        ax5.set_title('Frac of days w/ at least \n 75% returns/day in each set')

        # save figure
        figpath = 'C:/Users/rdchlerh/PycharmProjects/frf_python_share/figs/data/poststorm_timeslices_Nlook='+str(Nlook)+'/'
        fig.savefig(figpath + 'poststorm_' + str(jj) + '.png', dpi=300)  # save the figure to file
        plt.close(fig)


## ____________BELOW HERE IS OLD AND PROB WORTHLESS__________________________

# # Check all storm_START times
# check_keep = np.empty((len(storm_start),))
# check_keep[:] = np.nan
# storm_start_toss = []
# for jj in np.arange(len(storm_start)):
#     time_check = storm_start[jj]
#     if np.sum(time_check == time_fullspan) == 1:
#         iicheck = np.where(time_check == time_fullspan)[0]
#         storm_prior = (storm_flag[iicheck-1] == 1)
#         storm_after = (storm_flag[iicheck+1] == 1)
#         if storm_prior:
#             print('storm happening before storm_start['+str(jj)+'] (iicheck = '+str(iicheck[0])+')')
#             storm_start_toss = np.append(storm_start_toss, jj)
#         if ~storm_after:
#             print('storm stops immediately after storm_start[' + str(jj) + '] (iicheck = ' + str(iicheck[0]) + ')')
#             storm_start_toss = np.append(storm_start_toss, jj)
#         check_keep[jj] = 1
#     else:
#         iiclose = np.where(abs(time_fullspan - time_check) == np.nanmin(abs(time_fullspan - time_check)))[0]
#         if len(iiclose) > 0:
#             storm_prior = (storm_flag[iiclose[0]] == 1)
#             storm_after = (storm_flag[iiclose[1]] == 1)
#             check_keep[jj] = 1
#             if storm_prior:
#                 print('storm happening before storm_start[' + str(jj) + '] (iiclose = ' + str(iiclose) + ')')
#                 storm_start_toss = np.append(storm_start_toss, jj)
#             if ~storm_after:
#                 print('storm stops immediately after storm_start[' + str(jj) + '] (iiclose = ' + str(iiclose) + ')')
#                 storm_start_toss = np.append(storm_start_toss, jj)
# # Check all storm_END times
# check_keep = np.empty((len(storm_end),))
# check_keep[:] = np.nan
# storm_end_toss = []
# for jj in np.arange(len(storm_end)):
#     time_check = storm_end[jj]
#     if np.sum(time_check == time_fullspan) == 1:
#         iicheck = np.where(time_check == time_fullspan)[0]
#         calm_prior = (storm_flag[iicheck-1] == 0)
#         calm_after = (storm_flag[iicheck+1] == 0)
#         if calm_prior:
#             print('storm has ended before storm_end['+str(jj)+'] (iicheck = '+str(iicheck[0])+')')
#             storm_end_toss = np.append(storm_end_toss,jj)
#         if ~calm_after:
#             print('storm continues after storm_end[' + str(jj) + '] (iicheck = ' + str(iicheck[0]) + ')')
#             storm_end_toss = np.append(storm_end_toss, jj)
#         check_keep[jj] = 1
#     else:
#         iiclose = np.where(abs(time_fullspan - time_check) == np.nanmin(abs(time_fullspan - time_check)))[0]
#         if len(iiclose) > 0:
#             calm_prior = (storm_flag[iiclose[0]] == 0)
#             calm_after = (storm_flag[iiclose[1]] == 0)
#             check_keep[jj] = 1
#             if calm_prior:
#                 print('storm has ended before storm_end[' + str(jj) + '] (iicheck = ' + str(iiclose) + ')')
#                 storm_end_toss = np.append(storm_end_toss, jj)
#             if ~calm_after:
#                 print('storm continues after storm_end[' + str(jj) + '] (iicheck = ' + str(iiclose) + ')')
#                 storm_end_toss = np.append(storm_end_toss, jj)
#
# # Check all storm_START-WIS times
# check_keep = np.empty((len(storm_startWIS),))
# check_keep[:] = np.nan
# storm_startWIS_toss = []
# for jj in np.arange(len(storm_startWIS)):
#     time_check = storm_startWIS[jj]
#     if np.sum(time_check == time_fullspan) == 1:
#         iicheck = np.where(time_check == time_fullspan)[0]
#         storm_prior = (storm_flag[iicheck-1] == 1)
#         storm_after = (storm_flag[iicheck+1] == 1)
#         if storm_prior:
#             print('storm happening before storm_start['+str(jj)+'] (iicheck = '+str(iicheck[0])+')')
#             storm_startWIS_toss = np.append(storm_startWIS_toss, jj)
#         if ~storm_after:
#             print('storm stops immediately after storm_start[' + str(jj) + '] (iicheck = ' + str(iicheck[0]) + ')')
#             storm_startWIS_toss = np.append(storm_startWIS_toss, jj)
#         check_keep[jj] = 1
#     else:
#         iiclose = np.where(abs(time_fullspan - time_check) == np.nanmin(abs(time_fullspan - time_check)))[0]
#         if len(iiclose) > 0:
#             storm_prior = (storm_flag[iiclose[0]] == 1)
#             storm_after = (storm_flag[iiclose[1]] == 1)
#             check_keep[jj] = 1
#             if storm_prior:
#                 print('storm happening before storm_start[' + str(jj) + '] (iiclose = ' + str(iiclose) + ')')
#                 storm_startWIS_toss = np.append(storm_startWIS_toss, jj)
#             if ~storm_after:
#                 print('storm stops immediately after storm_start[' + str(jj) + '] (iiclose = ' + str(iiclose) + ')')
#                 storm_startWIS_toss = np.append(storm_startWIS_toss, jj)
#
# # Check all storm_END-WIS times
# check_keep = np.empty((len(storm_endWIS),))
# check_keep[:] = np.nan
# storm_endWIS_toss = []
# for jj in np.arange(len(storm_endWIS)):
#     time_check = storm_endWIS[jj]
#     if np.sum(time_check == time_fullspan) == 1:
#         iicheck = np.where(time_check == time_fullspan)[0]
#         calm_prior = (storm_flag[iicheck-1] == 0)
#         calm_after = (storm_flag[iicheck+1] == 0)
#         if calm_prior:
#             print('storm has ended before storm_end['+str(jj)+'] (iicheck = '+str(iicheck[0])+')')
#             storm_endWIS_toss = np.append(storm_endWIS_toss,jj)
#         if ~calm_after:
#             print('storm continues after storm_end[' + str(jj) + '] (iicheck = ' + str(iicheck[0]) + ')')
#             storm_endWIS_toss = np.append(storm_endWIS_toss, jj)
#         check_keep[jj] = 1
#     else:
#         iiclose = np.where(abs(time_fullspan - time_check) == np.nanmin(abs(time_fullspan - time_check)))[0]
#         if len(iiclose) > 0:
#             calm_prior = (storm_flag[iiclose[0]] == 0)
#             calm_after = (storm_flag[iiclose[1]] == 0)
#             check_keep[jj] = 1
#             if calm_prior:
#                 print('storm has ended before storm_end[' + str(jj) + '] (iicheck = ' + str(iiclose) + ')')
#                 storm_endWIS_toss = np.append(storm_endWIS_toss, jj)
#             if ~calm_after:
#                 print('storm continues after storm_end[' + str(jj) + '] (iicheck = ' + str(iiclose) + ')')
#                 storm_endWIS_toss = np.append(storm_endWIS_toss, jj)
#
# # remove erroneous times
# jjrange = np.arange(storm_start.size)
# jjtoss = np.unique(storm_start_toss)
# storm_start_clean = storm_start[~np.isin(jjrange,jjtoss)]
# jjrange = np.arange(storm_startWIS.size)
# jjtoss = np.unique(storm_startWIS_toss)
# storm_startWIS_clean = storm_startWIS[~np.isin(jjrange,jjtoss)]
# jjrange = np.arange(storm_end.size)
# jjtoss = np.unique(storm_end_toss)
# storm_end_clean = storm_end[~np.isin(jjrange,jjtoss)]
# jjrange = np.arange(storm_endWIS.size)
# jjtoss = np.unique(storm_endWIS_toss)
# storm_endWIS_clean = storm_endWIS[~np.isin(jjrange,jjtoss)]
#
# fig, ax = plt.subplots()
# ax.plot(time_fullspan,storm_flag,'o')
# for jj in np.arange(len(storm_start_clean)):
#     yplot = [0,2]
#     ax.plot([storm_start_clean[jj],storm_start_clean[jj]],yplot,'g')
# for jj in np.arange(len(storm_end_clean)):
#     yplot = [0,2]
#     ax.plot([storm_end_clean[jj],storm_end_clean[jj]],yplot,'r')
# for jj in np.arange(len(storm_startWIS_clean)):
#     yplot = [0,2]
#     ax.plot([storm_startWIS_clean[jj],storm_startWIS_clean[jj]],yplot,'c')
# for jj in np.arange(len(storm_endWIS_clean)):
#     yplot = [0,2]
#     ax.plot([storm_endWIS_clean[jj],storm_endWIS_clean[jj]],yplot,'m')
# ax.set_title('1 == Stormy, 0 == Calm/Non-stormy')
#
# storm_timeend_all = []
# storm_timestart_all = []
# storm_iiend_all = []
# storm_iistart_all = []
# storm_flag[storm_flag == 0] = -1
# iicross = np.where(storm_flag[1:]*storm_flag[0:-1] < 0)[0]
# for jj in np.arange(iicross.size):
#     if (storm_flag[iicross[jj]] == -1) & (storm_flag[iicross[jj]+1] == 1):
#         storm_timestart_all = np.append(storm_timestart_all,time_fullspan[iicross[jj]])
#         storm_iistart_all = np.append(storm_iistart_all,iicross[jj]+1)
#     elif (storm_flag[iicross[jj]] == 1) & (storm_flag[iicross[jj]+1] == -1):
#         storm_timeend_all = np.append(storm_timeend_all, time_fullspan[iicross[jj]])
#         storm_iiend_all = np.append(storm_iiend_all,iicross[jj]+1)
#     else:
#         print('help')
#
# # Ok, now combine and check for near-duplicates in combined start and end vecs
# storm_start_all = np.hstack((storm_start_clean,storm_startWIS_clean))
# storm_end_all = np.hstack((storm_end_clean,storm_endWIS_clean))
# start_nearestneighbor = np.empty(shape=storm_start_all.shape)
# start_nearestneighbor[:] = np.nan
# end_nearestneighbor = np.empty(shape=storm_end_all.shape)
# end_nearestneighbor[:] = np.nan
# for jj in np.arange(storm_start_all.size):
#     distance = abs(storm_start_all[jj]-storm_start_all)
#     start_nearestneighbor[jj] = np.nanmin(distance[distance > 0])/3600
# fig, ax = plt.subplots()
# ax.plot(start_nearestneighbor,'o')
# for jj in np.arange(storm_end_all.size):
#     distance = abs(storm_end_all[jj]-storm_end_all)
#     end_nearestneighbor[jj] = np.nanmin(distance[distance > 0])/3600
# fig, ax = plt.subplots()
# ax.plot(end_nearestneighbor,'o')
#
#
# fig, ax = plt.subplots()
# ax.plot(time_fullspan,storm_flag,'o')
# for jj in np.arange(len(storm_start_all)):
#     yplot = [0,2]
#     if start_nearestneighbor[jj] > 1:
#         ax.plot([storm_start_all[jj],storm_start_all[jj]],yplot,'g')
# for jj in np.arange(len(storm_end_all)):
#     yplot = [0,2]
#     if end_nearestneighbor[jj] > 1:
#         ax.plot([storm_end_all[jj],storm_end_all[jj]],yplot,'r')
# ax.set_title('1 == Stormy, 0 == Calm/Non-stormy')
#
# # Check all storm_START times
# check_keep = np.empty((len(storm_start),))
# check_keep[:] = np.nan
# for jj in np.arange(len(storm_start)):
#     time_check = storm_start[jj]
#     if np.sum(time_check == time_fullspan) == 1:
#         iicheck = np.where(time_check == time_fullspan)[0]
#         storm_prior = (storm_flag[iicheck-1] == 1)
#         storm_after = (storm_flag[iicheck+1] == 1)
#         if storm_prior:
#             print('storm happening before storm_start['+str(jj)+'] (iicheck = '+str(iicheck[0])+')')
#         if ~storm_after:
#             print('storm stops immediately after storm_start[' + str(jj) + '] (iicheck = ' + str(iicheck[0]) + ')')
#         check_keep[jj] = 1
#     else:
#         iiclose = np.where(abs(time_fullspan - time_check) == np.nanmin(abs(time_fullspan - time_check)))[0]
#         if len(iiclose) > 0:
#             storm_prior = (storm_flag[iiclose[0]] == 1)
#             storm_after = (storm_flag[iiclose[1]] == 1)
#             check_keep[jj] = 1
#             if storm_prior:
#                 print('storm happening before storm_start[' + str(jj) + '] (iiclose = ' + str(iiclose) + ')')
#             if ~storm_after:
#                 print('storm stops immediately after storm_start[' + str(jj) + '] (iiclose = ' + str(iiclose) + ')')
#
# fig, ax = plt.subplots()
# ax.plot(storm_start,np.ones(shape=storm_start.shape),'o')
# ax.plot(storm_startWIS,2*np.ones(shape=storm_startWIS.shape),'o')
# storm_start_all = storm_start
# storm_end_all = storm_end
# for jj in np.arange(len(storm_startWIS)):
#     # find the closest FRF storm start time to WIS storm start time [jj]
#     iiclose = np.where(abs(storm_startWIS[jj] - storm_start) == np.nanmin(abs(storm_startWIS[jj] - storm_start)))[0]
#     timediff = abs((storm_start[iiclose]-storm_startWIS[jj])/3600)
#     if timediff >= 24:
#         # if close, add to "all start times"
#         storm_start_all = np.append(storm_start_all,storm_startWIS)
#     # find the closest FRF storm end time to WIS storm end time [jj]
#     iiclose = np.where(abs(storm_endWIS[jj] - storm_end == np.nanmin(abs(storm_endWIS[jj] - storm_end))))[0]
#     timediff = abs((storm_end[iiclose] - storm_endWIS[jj]) / 3600)
#     if timediff >= 24:
#         # if close, add to "all end times"
#         storm_end_all = np.append(storm_end_all, storm_endWIS)