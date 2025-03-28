import pickle
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load the dataframe
from funcs.create_contours import *
from funcs.calculate_beachvol import *
import pickle



# picklefile_dir = 'C:/Users/rdchlerh/Desktop/frf_data_backup/processed/processed_20Feb2025/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_20Feb2025/'
with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_Nlook60_PCApostDVol_shifted.pickle', 'rb') as file:
    xplot_shift, time_fullspan, dataNorm_fullspan, dataMean, dataStd, PCs_fullspan, EOFs, APEV, reconstruct_profNorm_fullspan, reconstruct_prof_fullspan, dataobs_shift_fullspan, dataobs_fullspan, data_profIDs_dVolThreshMet, data_hydro, datahydro_fullspan = pickle.load(file)
with open(picklefile_dir+'stormy_times_fullspan.pickle','rb') as file:
   _,storm_flag,storm_timestart_all,storm_timeend_all = pickle.load(file)

# recreate observed datasets
dataobs_fullspan = dataobs_shift_fullspan
dataPCA_fullspan = reconstruct_prof_fullspan

# Calculate beach volume and width
dx = 0.1
mlw = -0.62
mwl = -0.13
zero = 0
mhw = 0.36
dune_toe = 3.22
upper_lim = 5.95
cont_elev = np.array([mlw,mwl,mhw,dune_toe,upper_lim]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts_obs, cmean, cstd = create_contours(dataobs_fullspan.T,time_fullspan,xplot_shift,cont_elev)
cont_ts_pca, cmean, cstd = create_contours(dataPCA_fullspan.T,time_fullspan,xplot_shift,cont_elev)

beachVol_obs, beachVol_xc_obs, dBeachVol_dt_obs, total_beachVol_obs, total_dBeachVol_dt_obs, total_obsBeachWid_obs =  calculate_beachvol(dataobs_fullspan.T,time_fullspan,xplot_shift,cont_elev,cont_ts_obs)
beachVol_pca, beachVol_xc_pca, dBeachVol_dt_pca, total_beachVol_pca, total_dBeachVol_dt_pca, total_obsBeachWid_pca =  calculate_beachvol(dataPCA_fullspan.T,time_fullspan,xplot_shift,cont_elev,cont_ts_pca)
total_beachVol_obs[total_beachVol_obs == 0] = np.nan
total_beachVol_pca[total_beachVol_pca == 0] = np.nan

# with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_obsBeachProfStats.pickle', 'wb') as file:
#     pickle.dump([cont_ts_obs, beachVol_obs, beachVol_xc_obs, dBeachVol_dt_obs, total_beachVol_obs, total_dBeachVol_dt_obs, total_obsBeachWid_obs],file)
# with open(picklefile_dir + 'topobathyhydro_ML_final_25Mar2025_pcaBeachProfStats.pickle', 'wb') as file:
#     pickle.dump([cont_ts_pca, beachVol_pca, beachVol_xc_pca, dBeachVol_dt_pca, total_beachVol_pca, total_dBeachVol_dt_pca, total_obsBeachWid_pca],file)

########### Find beach width and volume loss post-storm ###########

with open(picklefile_dir+'stormHs95_Over12Hours.pickle','rb') as f:
    output = pickle.load(f)
    wavePowerStormList = np.array(output['wavePowerStormList'])
    startTimeStormList = np.array(output['startTimeStormList'])
    endTimeStormList = np.array(output['endTimeStormList'])
    recoveryTimeStorm = (np.asarray(endTimeStormList)[1:] - np.asarray(startTimeStormList)[0:-1])

nstorms = len(startTimeStormList)-3

beachwid_prestorm_obs = np.empty((nstorms,))*np.nan
beachwid_prestorm_pca = np.empty((nstorms,))*np.nan
beachwid_poststorm_obs = np.empty((nstorms,))*np.nan
beachwid_poststorm_pca = np.empty((nstorms,))*np.nan
beachvol_prestorm_obs = np.empty((nstorms,))*np.nan
beachvol_prestorm_pca = np.empty((nstorms,))*np.nan
beachvol_poststorm_obs = np.empty((nstorms,))*np.nan
beachvol_poststorm_pca = np.empty((nstorms,))*np.nan
for jj in np.arange(nstorms):
    # pre-storm
    # ii_prestorm = np.where(np.isin(time_fullspan,startTimeStormList[jj]))[0].astype(int)
    ii_prestorm = np.where(abs(time_fullspan - startTimeStormList[jj])==np.min(abs(time_fullspan - startTimeStormList[jj])))[0]
    if len(ii_prestorm) > 0:
        if len(ii_prestorm) > 1:
            ii_prestorm = ii_prestorm[0]
        ii_rng = np.arange(ii_prestorm - 10,ii_prestorm + 10+1)
        beachwid_prestorm_obs[jj] = np.nanmean(total_obsBeachWid_obs[ii_rng])
        beachwid_prestorm_pca[jj] = np.nanmean(total_obsBeachWid_pca[ii_rng])
        beachvol_prestorm_obs[jj] = np.nanmean(total_beachVol_obs[ii_rng])
        beachvol_prestorm_pca[jj] = np.nanmean(total_beachVol_pca[ii_rng])
    # post-storm
    # ii_poststorm = np.where(np.isin(time_fullspan, endTimeStormList[jj]))[0]
    ii_poststorm = np.where(abs(time_fullspan - endTimeStormList[jj])==np.min(abs(time_fullspan - endTimeStormList[jj])))[0]
    if len(ii_poststorm) > 0:
        if len(ii_poststorm) > 1:
            ii_poststorm = ii_poststorm[0]
        ii_rng = np.arange(ii_poststorm - 10,ii_poststorm + 10+1)
        beachwid_poststorm_obs[jj] = np.nanmean(total_obsBeachWid_obs[ii_rng])
        beachwid_poststorm_pca[jj] = np.nanmean(total_obsBeachWid_pca[ii_rng])
        beachvol_poststorm_obs[jj] = np.nanmean(total_beachVol_obs[ii_rng])
        beachvol_poststorm_pca[jj] = np.nanmean(total_beachVol_pca[ii_rng])

# plot width change post-storm
fig, ax = plt.subplots(2,1)
ax[0].plot(beachwid_prestorm_obs,'xb',label='pre-sotrm, obs')
ax[0].plot(beachwid_prestorm_pca,'.b',label='pre-sotrm, pca')
ax[0].plot(beachwid_poststorm_obs,'xr',label='post-sotrm, obs')
ax[0].plot(beachwid_poststorm_pca,'.r',label='post-sotrm, pca')
ax[0].set_title('beach width change')
ax[0].set_ylabel('width')
ax[0].legend()
ax[1].plot(beachwid_prestorm_obs-beachwid_poststorm_obs,'ob',label='obs')
ax[1].plot(beachwid_prestorm_pca-beachwid_poststorm_pca,'or',label='pca')
ax[1].set_ylabel('dWidth')
ax[1].legend()

fig, ax = plt.subplots(2,1)
ax[0].plot(beachvol_prestorm_obs,'xb',label='pre-sotrm, obs')
ax[0].plot(beachvol_prestorm_pca,'.b',label='pre-sotrm, pca')
ax[0].plot(beachvol_poststorm_obs,'xr',label='post-sotrm, obs')
ax[0].plot(beachvol_poststorm_pca,'.r',label='post-sotrm, pca')
ax[0].legend()
ax[0].set_title('beach volume change')
ax[0].set_ylabel('Vol')
ax[1].plot(beachvol_prestorm_obs-beachwid_poststorm_obs,'ob',label='obs')
ax[1].plot(beachvol_prestorm_pca-beachwid_poststorm_pca,'or',label='pca')
ax[1].set_ylabel('dVol')
ax[0].legend()

fig, ax = plt.subplots()
dvol_obs = beachvol_prestorm_obs-beachvol_poststorm_obs
dvol_pca = beachvol_prestorm_pca-beachvol_poststorm_pca
ax.plot(wavePowerStormList[:nstorms],dvol_obs,'r<',label='obs')
ax.plot(wavePowerStormList[:nstorms],dvol_pca,'b>',label='pca')
ax.set_ylabel('dVol')
ax.set_xlabel('cumulative wave power')
ax.legend()

fig, ax = plt.subplots()
ax.plot(beachvol_prestorm_obs,dvol_obs,'r<',label='obs')
ax.plot(beachvol_prestorm_pca,dvol_pca,'b>',label='pca')
ax.set_ylabel('dVol')
ax.set_xlabel('pre-storm volume')
ax.legend()

fig, ax = plt.subplots()
ax.plot(beachwid_prestorm_obs,dvol_obs,'r<',label='obs')
ax.plot(beachwid_prestorm_pca,dvol_pca,'b>',label='pca')
ax.set_ylabel('dVol')
ax.set_xlabel('pre-storm width')
ax.legend()

fig, ax = plt.subplots()
dwidth_obs = beachwid_prestorm_obs-beachwid_poststorm_obs
dwidth_pca = beachwid_prestorm_pca-beachwid_poststorm_pca
ax.plot(beachwid_prestorm_obs,dwidth_obs,'r<',label='obs')
ax.plot(beachwid_prestorm_pca,dwidth_pca,'b>',label='pca')
ax.set_ylabel('dWidth')
ax.set_xlabel('pre-storm width')
ax.legend()

fig, ax = plt.subplots()
ax.plot(wavePowerStormList[:nstorms],dwidth_obs,'r<',label='obs')
ax.plot(wavePowerStormList[:nstorms],dwidth_pca,'b>',label='pca')
ax.set_ylabel('dWidth')
ax.set_xlabel('cumulative wave power')
ax.legend()


## Find times when dVol or dWid is POSITIVE (pre-storm > post-storm)
tmp = np.arange(nstorms)
ii_netloss = tmp[(dwidth_pca > 0) | (dvol_pca > 0)]
ii_netloss_vol = tmp[(dvol_pca > 0)]
ii_netloss_wid = tmp[(dwidth_pca > 0)]

ii_netgain = tmp[(dwidth_pca < 0) | (dvol_pca < 0)]


ii_netloss_vol = [ 15,  29,  32,  52,  72,  78,  89, 101, 106]
fig1, ax1 = plt.subplots()      # volume
# cmap = plt.cm.rainbow(np.linspace(0, 1, len(ii_netloss_vol)))
# ax1.set_prop_cycle('color',cmap)
vol_recovery_time = np.empty((nstorms,))*np.nan
for jj in ii_netloss_vol:
    ii_poststorm = np.where(abs(time_fullspan - endTimeStormList[jj])==np.min(abs(time_fullspan - endTimeStormList[jj])))[0]
    if len(ii_poststorm) > 1:
        ii_poststorm = ii_poststorm[0]
    ii_nextstorm = np.where(abs(time_fullspan - startTimeStormList[jj+1])==np.min(abs(time_fullspan - startTimeStormList[jj+1])))[0]
    if len(ii_nextstorm) > 1:
        ii_nextstorm = ii_nextstorm[0]
    # nnplot = np.arange(ii_poststorm,ii_poststorm+24*60)
    nnplot = np.arange(ii_poststorm,ii_nextstorm)
    xplot = (nnplot-ii_poststorm)/24
    yplot = total_beachVol_pca[nnplot] - beachvol_prestorm_pca[jj]
    nsmooth = 12
    ymean = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
    ts = pd.Series(yplot)
    ystd = ts.rolling(window=nsmooth, center=True).std()
    bad_id = (abs(yplot - ymean) >= 3 * ystd)
    yplot[bad_id] = np.nan
    ysmooth = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
    # ysmooth[ysmooth>100] = np.nan
    ax1.plot(xplot,ysmooth,linewidth=2)
    recover_time_approx = np.where(abs(ysmooth) == np.nanmin(abs(ysmooth)))[0]
    if abs(ysmooth[recover_time_approx]) < 100:
        itmp = np.arange(recover_time_approx-3,recover_time_approx+3)
        vol_recovery_time[jj] = np.interp(0,ysmooth[itmp],xplot[itmp])
ax1.grid()
ax1.set_xlabel('time')


fig2, ax2 = plt.subplots()      # width
cmap = plt.cm.rainbow(np.linspace(0, 1, len(ii_netloss_wid)))
ax2.set_prop_cycle('color',cmap)
for jj in ii_netloss_wid:
    ii_poststorm = np.where(abs(time_fullspan - endTimeStormList[jj])==np.min(abs(time_fullspan - endTimeStormList[jj])))[0]
    if len(ii_poststorm) > 1:
        ii_poststorm = ii_poststorm[0]
    ii_nextstorm = np.where(abs(time_fullspan - startTimeStormList[jj+1])==np.min(abs(time_fullspan - startTimeStormList[jj+1])))[0]
    if len(ii_nextstorm) > 1:
        ii_nextstorm = ii_nextstorm[0]
    # nnplot = np.arange(ii_poststorm,ii_poststorm+24*60)
    nnplot = np.arange(ii_poststorm,ii_nextstorm)
    xplot = (nnplot-ii_poststorm)/24
    yplot = total_obsBeachWid_pca[nnplot] - beachwid_prestorm_pca[jj]
    nsmooth = 12
    ymean = np.convolve(yplot, np.ones(nsmooth) / nsmooth, mode='same')
    ts = pd.Series(yplot)
    ystd = ts.rolling(window=nsmooth, center=True).std()
    bad_id = (abs(yplot - ymean) >= 3 * ystd)
    yplot[bad_id] = np.nan
    ax2.plot(xplot,yplot)
ax2.grid()
ax2.set_xlabel('time (days)')

fig, ax = plt.subplots()
ax.plot(wavePowerStormList[:nstorms],vol_recovery_time,'x')
fig, ax = plt.subplots()
ax.plot(dvol_pca,vol_recovery_time,'b+')
ax.plot(dvol_pca,recoveryTimeStorm[:nstorms]/(24*3600),'xr')