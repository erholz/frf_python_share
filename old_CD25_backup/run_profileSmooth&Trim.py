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
from astropy.convolution import convolve, interpolate_replace_nans, Gaussian1DKernel
from scipy.interpolate import splrep, BSpline, splev, CubicSpline
import seaborn as sns
from run_hydrocollect import run_getnoaatidewithpred_func
from datetime import datetime



# Load temporally aligned data - need to add lidarelev_fullspan
picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]
    # lidar_xFRF = pickle.load(file)
with open(picklefile_dir + 'IO_lidarhydro_aligned.pickle', 'rb') as file:
   lidarhydro_min_fullspan, lidarhydro_max_fullspan, lidarhydro_mean_fullspan = pickle.load(file)
with open(picklefile_dir + 'IO_lidarquality.pickle', 'rb') as file:
   lidarelevstd_fullspan, lidarmissing_fullspan = pickle.load(file)
with open(picklefile_dir + 'xc_fullspan_MWL_DuneToe_MHW.pickle', 'rb') as file:
   xc_fullspan, dXcdt_fullspan = pickle.load(file)
with open(picklefile_dir+'elev&slp_processed_fullspan.pickle', 'rb') as file:
    _, avgslope_fullspan = pickle.load(file)



def find_nangaps(zinput):
    if sum(np.isnan(zinput)) == 0:
        gapstart = np.nan
        gapend = np.nan
        gapsize = np.array([0])
        maxgap = np.array([0])
    elif sum(np.isnan(zinput)) == 1:
        gapstart = np.where(np.isnan(zinput))
        gapend = np.where(np.isnan(zinput))
        gapsize = np.array([1])
        maxgap = np.array([1])
    else:
        numcumnan = np.empty(shape=zinput.shape)
        numcumnan[:] = np.nan
        tmp = np.cumsum(np.isnan(zinput), axis=0)
        numcumnan[tmp > 0] = tmp[tmp > 0]
        uniq_numcumnan = np.unique(numcumnan)
        uniq_numcumnan = uniq_numcumnan[~np.isnan(uniq_numcumnan)]
        tmpgapstart = []
        tmpgapend = []
        for ij in np.arange(uniq_numcumnan.size):
            # If there is only ONE entry of a cumnan value, then we know it's a new nan value OR it's the end of the vector
            if sum((numcumnan == uniq_numcumnan[ij])) == 1:
                tmp = np.where(numcumnan == uniq_numcumnan[ij])[0]
                tmpgapstart = np.append(tmpgapstart,tmp[0])
                # if tmp is the END of the vector, also designate as tmpgapend
                if tmp == len(zinput)-1:
                    tmpgapend = np.append(tmpgapend,tmp[0])
            # If there are multiple entries of a cumnan value, then we know it switches from nan to not-nan
            elif sum((numcumnan == uniq_numcumnan[ij])) > 1:
                tmp = np.where(numcumnan == uniq_numcumnan[ij])[0]
                # the first value of tmp is where it switches from nan to not-nan, the last would be the first before the next nan (if it exists)
                tmpgapend = np.append(tmpgapend,tmp[0])
                # if there is more than 1 instance of cumnan but no preceding nan, then it is ALSO a starting nan
                if ~np.isnan(zinput[tmp[0]-1]):
                    tmpgapstart = np.append(tmpgapstart,tmp[0])
                # if it is the FIRST value, then it is ALSO a tmpgapstart
                if tmp[0] == 0:
                    tmpgapstart = np.append(tmpgapstart, tmp[0])
        # new revisions may create duplicates....
        tmpgapend = np.unique(tmpgapend)
        tmpgapstart = np.unique(tmpgapstart)
        gapend = tmpgapend[:]
        # if NO tmpgapstart have been found, then we have multiple single-nans
        if len(tmpgapstart) == 0:
            gapstart = gapend[:]
        else:
            gapstart = [tmpgapstart[0]]
            if len(tmpgapstart) > 0:
                # now, we need to figure out if this is in the beginning of the gap or the middle
                tmp1 = np.diff(tmpgapstart)     # tmp1 is GRADIENT in tmpgapstart (diff of 1 indicates middle of gap)
                tmp2 = tmpgapstart[1:]          # tmp2 is all the tmpgapstarts OTHER than the first
                # if there is only ONE gap of 3 nans, there will be no tmp1 not equal to 1...
                if sum(tmp1 != 1) > 0:
                    # tmpid = where numcumnan is equal to a gap that is not in the middle (tmp1 != 1)
                    tmpid = tmp2[tmp1 != 1]
                    gapstart = np.append(gapstart, tmpid)
            if len(gapend) > len(gapstart):
                for ij in np.arange(gapend.size):
                    # if there is a gapend that is surrounded by non-nans, then it is a single-nan gap
                    if gapend[ij] == len(zinput)-1:
                        if ~np.isnan(zinput[int(gapend[ij]) - 1]):
                            missinggapstart = gapend[ij]
                            gapstart = np.append(missinggapstart, gapstart)
                    else:
                        if ~np.isnan(zinput[int(gapend[ij])-1]) & ~np.isnan(zinput[int(gapend[ij])+1]):
                            missinggapstart = gapend[ij]
                            gapstart = np.append(missinggapstart, gapstart)
            if np.max(gapstart) > np.max(gapend):
                gapend = np.append(gapend, np.max(gapstart))
        gapend = np.unique(gapend)
        gapstart = np.unique(gapstart)
        gapend = np.array(sorted(gapend))
        gapstart = np.array(sorted(gapstart))
        gapsize = (gapend - gapstart) + 1
        maxgap = np.nanmax(gapsize)
    return gapstart, gapend, gapsize, maxgap

def smooth_profile(xprof,zprof,xtrim,nsmooth,thresh):
    zmean_tmp = np.convolve(zprof, np.ones(nsmooth)/nsmooth, mode='same')
    zmean = zmean_tmp[(xprof >= np.min(xtrim)) & (xprof <= np.max(xtrim)) ]
    ztrim = zprof[(xprof >= np.min(xtrim)) & (xprof <= np.max(xtrim)) ]
    ts = pd.Series(zprof)
    zstd_tmp = ts.rolling(window=nsmooth, center=True).std()
    zstd = zstd_tmp[(xprof >= np.min(xtrim)) & (xprof <= np.max(xtrim)) ]
    bad_id = (abs(ztrim - zmean) >= thresh*zstd)
    zbad = np.empty(shape=ztrim.shape)
    zgood = np.empty(shape=ztrim.shape)
    zbad[:] = np.nan
    zgood[:] = np.nan
    if sum(bad_id) > 0:
        zbad[bad_id] = ztrim[bad_id]
        zgood[~bad_id] = ztrim[~bad_id]
    else:
        zgood = ztrim[:]
    return zbad, zgood

# Calculate contours
mwl = -0.13
mhw = 3.6
dune_toe = 3.22
cont_elev = np.array([mwl, dune_toe, mhw]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(lidarelev_fullspan.T,time_fullspan,lidar_xFRF,cont_elev)
tmp = cont_ts[:,1:] - cont_ts[:,0:-1]
dXContdt = np.hstack((tmp,np.empty((cont_elev.size,1))))
dXContdt[:,-1] = np.nan
xc_fullspan = cont_ts[:]
dXcdt_fullspan = dXContdt[:]
picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
# with open(picklefile_dir+'xc_fullspan_MWL_DuneToe_MHW.pickle', 'wb') as file:
#     pickle.dump([xc_fullspan,dXcdt_fullspan], file)

# Plot the range of contour positions
fig, ax = plt.subplots(1,3)
ax[0].hist(xc_fullspan[0,:],density=True)
ax[0].set_title('MWL, N = '+str(sum(~np.isnan(xc_fullspan[0,:]))))
ax[0].plot([np.nanmean(xc_fullspan[0,:]),np.nanmean(xc_fullspan[0,:])],[0, 0.035],'k')
ax[0].set_ylim(0, 0.035)
ax[1].hist(xc_fullspan[1,:],density=True)
ax[1].set_title('dune toe, N = '+str(sum(~np.isnan(xc_fullspan[1,:]))))
ax[1].plot([np.nanmean(xc_fullspan[1,:]),np.nanmean(xc_fullspan[1,:])],[0, 0.35],'k')
ax[1].set_ylim(0, 0.35)
ax[2].hist(xc_fullspan[2,:],density=True)
ax[2].set_title('MHW, N = '+str(sum(~np.isnan(xc_fullspan[2,:]))))
ax[2].plot([np.nanmean(xc_fullspan[2,:]),np.nanmean(xc_fullspan[2,:])],[0, 0.35],'k')
ax[2].set_ylim(0, 0.35)

# Plot lidarelev before we do anything to it:
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(lidarelev_fullspan.T, lidarelev_fullspan.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_title('Profile elevation - Pre filtering/smoothing')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_fullspan)
ax.set_title('Profile elevation - Pre filtering/smoothing')

# Apply lidar-qaqc criterion to data:
lidarelev_QAQC_fullspan = np.empty(shape=lidarelev_fullspan.shape)
lidarelev_QAQC_fullspan[:] = np.nan
lidarelev_QAQC_fullspan[:] = lidarelev_fullspan[:]
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.60      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd_fullspan >= stdthresh) + (lidarmissing_fullspan > pmissthresh)
lidarelev_QAQC_fullspan[tmpii] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_QAQC_fullspan)
ax.set_title('Profile elevation - Post QAQC threshholds (std ~> 5cm & %miss ~>60)')
tmpii = ~np.isnan(lidarhydro_min_fullspan)
lidarelev_QAQC_fullspan[tmpii] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_QAQC_fullspan)
ax.set_title('Profile elevation - Post removal where min_wl observed')

# Go through time, analyze data availability per tidal cycle, eliminate stray portions of the cross-shore when no "match" available
gauge = '8651370'
datum = 'MSL'
start_year = 2015#1978
end_year = 2024
tideout = run_getnoaatidewithpred_func(gauge, datum, start_year, end_year)
dat = tideout['wl']
time = tideout['wltime']
time_tide_predUTC = np.asarray([(dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's') for dt64 in tideout['predtimeDateTime'][::5]])
time_tide_pred = np.asarray([datetime.utcfromtimestamp(utc) for utc in time_tide_predUTC])
tide_pred = tideout['pred'][::5]
pklocs, _ = sp.signal.find_peaks(tide_pred, height=0)
# Check that peaks and troughs overlap between NOAA pred and FRF obs
fig, ax = plt.subplots()
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
ax.plot(time_tide_pred,tide_pred)
ax.plot(time_tide_pred[pklocs],tide_pred[pklocs],'o')
# ax.plot(tplot,data_tidegauge)
# Ok, now go through each peak and find available data as a func of cross-shore distance
lidarelev_removeStrays_fullspan = np.empty(shape=lidarelev_fullspan.shape)
lidarelev_removeStrays_fullspan[:] = np.nan
for jj in np.arange(pklocs.size-1):
    tpeak1 = time_tide_pred[pklocs[jj]]
    tpeak2 = time_tide_pred[pklocs[jj+1]]
    # find the profiles between tpeak1 and tpeak2
    btwnpks = np.where((tplot >= tpeak1) & (tplot <= tpeak2))[0]
    if btwnpks.size > 0:
        # find the total number of not-nans along the cross-shore profile
        zbtwnpks = lidarelev_QAQC_fullspan[:,btwnpks]
        numnotnan = np.sum(~np.isnan(zbtwnpks),axis=1)
        # define which cross-shore IDs do not meet criterion
        ix_toss = numnotnan < 2
        # nan-out those cross-shore locations, re-assigned filtered profiles back to lidarelev_fullspan
        zbtwnpks[ix_toss,:] = np.nan
        lidarelev_removeStrays_fullspan[:, btwnpks] = zbtwnpks
# Plot post filtering...
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(lidarelev_removeStrays_fullspan.T, lidarelev_removeStrays_fullspan.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_title('Profile elevation - Post tide-based filtering')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_removeStrays_fullspan)
ax.set_title('Profile elevation - Post tide-based filtering')

# Ok, so remove isolate strays, dependent on cross-shore data availability
count = np.empty(shape=time_fullspan.shape)
count[:] = np.nan
lidarelev_removeStrays_fullspan_Xshore = np.empty(shape=lidarelev_removeStrays_fullspan.shape)
lidarelev_removeStrays_fullspan_Xshore[:] = np.nan
for tt in np.arange(time_fullspan.size):
    ztmp = lidarelev_removeStrays_fullspan[:, tt]
    # find the profile within 25 cm (vert) of hourly water level
    wltmp = data_tidegauge[tt]
    if sum(~np.isnan(ztmp)) > 10:
        ii_check = np.where(ztmp < wltmp + 0.30)
        zlower = ztmp[ii_check]
        count[tt] = zlower.size
        # ^^ go through ztmp and remove cross-shore strays
        numcheck = 6
        halfspan = numcheck/2
        for jj in np.arange(int(halfspan),zlower.size-int(halfspan)):
            ztmp_check = zlower[np.arange(jj-int(halfspan),jj+int(halfspan))]
            percavail = sum(~np.isnan(ztmp_check))/ztmp_check.size
            if percavail < 0.6:
                zlower[jj] = np.nan
        ztmp[ii_check] = zlower
        lidarelev_removeStrays_fullspan_Xshore[:, tt] = ztmp
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_removeStrays_fullspan_Xshore)
ax.set_title('Profile elevation - Cross-shore strays')



# Load profiles, remove locally high variability - use whole profile, but only if we have data between Xc[0] & Xc[-1]
zpass_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zbad_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zbad_fullspan[:] = np.nan
zpass_fullspan[:] = np.nan
maxgap_presmooth = np.empty((time_fullspan.size,))
maxgap_postsmooth = np.empty((time_fullspan.size,))
maxgap_fullspan = np.empty((time_fullspan.size,))
maxgap_presmooth[:] = np.nan
maxgap_postsmooth[:] = np.nan
maxgap_fullspan[:] = np.nan
for tt in np.arange(len(time_fullspan)):
    xc_shore = xc_fullspan[-1,tt]
    xc_sea = xc_fullspan[0,tt]
    ztmp = lidarelev_removeStrays_fullspan_Xshore[:, tt]
    # We can perform the smoothing if we have data even if XC not found
    near_xc_thresh = 0.5        # unit of [m], vertical
    mwl = -0.13
    mhw = 3.6
    if ((~np.isnan(xc_shore)) & (~np.isnan(xc_sea))) or ((np.nanmin(ztmp) <= mwl+near_xc_thresh) & (np.nanmax(ztmp) >= mhw-near_xc_thresh)):
        ztmp = lidarelev_removeStrays_fullspan_Xshore[:,tt]
        # Do some smoothing if we have enough data
        if sum(~np.isnan(ztmp)) > 10:
            # Isolate the non-nan portion for analysis
            ix_notnan = np.where(~np.isnan(ztmp))[0]
            approxlength = ix_notnan[-1] - ix_notnan[0]
            zinput = ztmp[np.arange(ix_notnan[0],ix_notnan[-1])]
            gapstart, gapend, gapsize, maxgap = find_nangaps(zinput)
            maxgap_presmooth[tt] = maxgap
            # only do smoothing routine if we have data between Xc[0] and Xc[-1] and there's >60% not-nans across prof
            maxgapthresh = 500
            if ((ix_notnan.size/approxlength) > 0.6) or (maxgap <= maxgapthresh):
                xtmp = lidar_xFRF[ix_notnan]
                zprof_init = ztmp[ix_notnan]
                ztmp = ztmp[ix_notnan]
                # First Smooth - remove extreme points
                Lsmooth1 = 2
                dx = 0.1
                thresh = 0.5
                nsmooth = int(np.ceil(Lsmooth1 / dx))
                zbad, zgood = smooth_profile(xtmp, ztmp, xtmp, nsmooth, thresh)
                zbad_fullspan[ix_notnan, tt] = zbad
                zpass_fullspan[ix_notnan,tt] = zgood
# Repeat the tidal-window filtering
zpass_tidefilter_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zpass_tidefilter_fullspan[:] = np.nan
for jj in np.arange(pklocs.size-1):
    tpeak1 = time_tide_pred[pklocs[jj]]
    tpeak2 = time_tide_pred[pklocs[jj+1]]
    # find the profiles between tpeak1 and tpeak2
    btwnpks = np.where((tplot >= tpeak1) & (tplot <= tpeak2))[0]
    if btwnpks.size > 0:
        # find the total number of not-nans along the cross-shore profile
        zbtwnpks = zpass_fullspan[:,btwnpks]
        numnotnan = np.sum(~np.isnan(zbtwnpks),axis=1)
        # define which cross-shore IDs do not meet criterion
        ix_toss = numnotnan < 2
        # nan-out those cross-shore locations, re-assigned filtered profiles back to lidarelev_fullspan
        zbtwnpks[ix_toss,:] = np.nan
        zpass_tidefilter_fullspan[:, btwnpks] = zbtwnpks


# FILTER - remove high-percent nan region at seaward edge of profile
zinput = zpass_tidefilter_fullspan[:]
percentnan_fullspan = np.empty(shape=lidarelev_fullspan.shape)
percentnan_fullspan[:] = np.nan
threshmet = np.empty(shape=time_fullspan.shape)
threshmet[:] = np.nan
thresh = 0.2
for tt in np.arange(len(time_fullspan)):
    span = 10
    halfspan = int(span/2)
    if np.sum(np.isnan(zinput[:,tt])) < lidar_xFRF.size:
        for jj in np.arange(halfspan,lidar_xFRF.size-halfspan):
            ztmp = zinput[np.arange(jj-halfspan,jj+halfspan),tt]
            percentnan = np.sum(np.isnan(ztmp))/ztmp.size
            percentnan_fullspan[jj,tt] = percentnan
        # find first time percent nan falls below threshhold from seaward side
        tmp = np.where(percentnan_fullspan[:,tt] < thresh)[0]
        if tmp.size > 0:
            threshmet[tt] = np.nanmax(tmp)
# plot the profiles from threshmet and onwards to see what the data is like...
zdiscard = np.empty(shape=zinput.shape)
zdiscard[:] = zinput[:]
zkeep = np.empty(shape=zinput.shape)
zkeep[:] = zinput[:]
for tt in np.arange(time_fullspan.size):
    if ~np.isnan(threshmet[tt]):
        zdiscard[np.arange(int(threshmet[tt])),tt] = np.nan
        zkeep[np.arange(int(threshmet[tt]),lidar_xFRF.size),tt] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zkeep)
ax.set_title('Remove high-nan region')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zdiscard)
ax.set_title('Data removed')
# plot as scatter
percentnan_fullspan[percentnan_fullspan == 1.00] = np.nan
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(percentnan_fullspan.T, percentnan_fullspan.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('% nan')
xplot = np.empty(shape=time_fullspan.shape)
xplot[:] = np.nan
for tt in np.arange(xplot.size):
    if ~np.isnan(threshmet[tt]):
        xplot[tt] = lidar_xFRF[int(threshmet[tt])]
ax.plot(xplot,tplot,'rx')


# NOW we can try the smoothing...
zsmooth_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zsmooth_fullspan[:] = np.nan
for tt in np.arange(len(time_fullspan)):
    ztmp = zkeep[:, tt]
    # Do some smoothing if we have enough data
    if sum(~np.isnan(ztmp)) > 10:
        ix_notnan = np.where(~np.isnan(ztmp))[0]
        ix_good = np.arange(ix_notnan[0], ix_notnan[-1])
        zgood = ztmp[ix_good]
        xtmp = lidar_xFRF[ix_good]
        gapstart, gapend, gapsize, maxgap = find_nangaps(zgood)
        maxgap_postsmooth[tt] = maxgap
        # Second smooth - refine if enough data exists
        delta = 10
        maxgapthresh = 200
        if (maxgap > maxgapthresh):
           # remove data where xFRF > 140
           ztmp[np.where(lidar_xFRF >= 140)] = np.nan
           ix_notnan = np.where(~np.isnan(ztmp))[0]
           ix_good = np.arange(ix_notnan[0], ix_notnan[-1])
           zgood = ztmp[ix_good]
           gapstart, gapend, gapsize, maxgap = find_nangaps(zgood)
           maxgap_postsmooth[tt] = maxgap
           gapstart, gapend, gapsize, maxgap = find_nangaps(zgood)
           maxgap_postsmooth[tt] = maxgap
        if (maxgap <= maxgapthresh):
            Lsmooth2 = 0.9
            dx = 0.1
            nsmooth = int(np.ceil(Lsmooth2 / dx))
            mykernel = np.ones(nsmooth)/nsmooth
            padding = np.zeros((nsmooth,))*2
            zinput = np.hstack((padding, zgood, padding))
            zconvolved = convolve(zinput, mykernel, boundary='extend')
            zsmooth_fullspan[ix_good[delta:-delta],tt] = zconvolved[nsmooth+delta:-(nsmooth+delta)]
            ztmp = zconvolved[nsmooth+delta:-(nsmooth+delta)]
            gapstart, gapend, gapsize, maxgap = find_nangaps(ztmp)
            maxgap_postsmooth[tt] = maxgap
            maxgap_fullspan[tt] = maxgap
# Plot the smoothed profiles
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zsmooth_fullspan)
ax.set_title('Profile elevation - Lsmooth2 = 0.9m')

# # Tried this, retaining for records, but not effective
# # Will a first-order 2nd-derivative (rate of slope change) indicate where we have weird flatness?
# first_diff = zsmooth_fullspan[1:,:] - zsmooth_fullspan[0:-1,:]
# second_diff = first_diff[1:,:] - first_diff[0:-1,:]
# fig, ax = plt.subplots()
# ax.plot(lidar_xFRF[2:],second_diff)






# Ok, now pull out slope data  to see if we can use it for filtering
z_slopeanalysis = np.empty(shape=zsmooth_fullspan.shape)
z_slopeanalysis[:] = zsmooth_fullspan[:]
avgslope_fullspan = np.empty(shape=lidarelev_fullspan.shape)
avgslope_fullspan[:] = np.nan
dx = 0.1
for tt in np.arange(len(time_fullspan)):
    ztmp = z_slopeanalysis[:,tt]
    ix_notnan = np.where(~np.isnan(ztmp))[0]
    # If profile is not all nans - calculate the average slope over nslopecheck
    if np.sum(~np.isnan(ztmp)) > 0:
        nslopecheck = 8
        delta = 2
        z_for_slopecheck = ztmp
        iterrange = ix_notnan[delta:-delta]
        for jj in np.arange(iterrange.size):
            tmpslope = np.empty(shape=(int(np.floor(nslopecheck / 2)),))
            tmpslope[:] = np.nan
            jj0 = iterrange[jj] - int(np.floor(nslopecheck * 0.75))
            for ii in np.arange(tmpslope.size):
                tmpslope_ii = (z_for_slopecheck[jj0] - z_for_slopecheck[jj0 + nslopecheck]) / (nslopecheck * dx)
                tmpslope[ii] = tmpslope_ii
                jj0 = jj0 + 1
            if sum(~np.isnan(tmpslope)) >= 2:
                avgslope_fullspan[iterrange[jj], tt] = np.nanmean(tmpslope)
# Can we remove the weird flatness based on average bed slope?
zscatter = np.reshape(avgslope_fullspan.T, avgslope_fullspan.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.075, vmax=0.075)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_title('Avg slope [m/m]')
# PLOT "flat" slope vs elevation
ii_flat = np.abs(avgslope_fullspan) < 0.01
xtmp = np.reshape(avgslope_fullspan[ii_flat], avgslope_fullspan[ii_flat].size)
ytmp = np.reshape(zsmooth_fullspan[ii_flat], zsmooth_fullspan[ii_flat].size)
fig, ax = plt.subplots()
ax.scatter(xtmp,ytmp,alpha=0.1)

# Ok, go through and find which profiles have an avg "flat" slope over the first few most seaward pts
flag_flattoe = np.empty(shape=time_fullspan.shape).astype(bool)
flag_flattoe[:] = False
for tt in np.arange(time_fullspan.size):
    slptmp = avgslope_fullspan[:,tt]
    if np.sum(~np.isnan(slptmp)) > 10:
        slptmp = slptmp[~np.isnan(slptmp)]
        ncheck = 4
        if np.mean(abs(slptmp[-10:])) < 0.02:
            flag_flattoe[tt] = True
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zsmooth_fullspan[:,flag_flattoe])
ax.set_title('Identified profiles with flat seaward edge')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zsmooth_fullspan[:,~flag_flattoe])
ax.set_title('Profiles without flat seaward edge')

# Try removing all pts where abs(slp) < 0.015 & elev < 0
ii_testremove = (abs(avgslope_fullspan) < 0.015 ) & (zsmooth_fullspan < 0)
testplot_pass = np.empty(shape=zsmooth_fullspan.shape)
testplot_pass[:] = zsmooth_fullspan[:]
testplot_pass[ii_testremove] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,testplot_pass)
ax.set_title('Passed filter')
ii_testremove = (abs(avgslope_fullspan) >= 0.015 ) & (zsmooth_fullspan > 0)
testplot_fail = np.empty(shape=zsmooth_fullspan.shape)
testplot_fail[:] = zsmooth_fullspan
testplot_fail[ii_testremove] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,testplot_fail)
ax.set_title('Identified as "flat" and below 0')

# find and filter based on percent-nans
zinput = np.empty(shape=zsmooth_fullspan.shape)
zinput[:] = testplot_pass[:]
percentnan_fullspan_round2 = np.empty(shape=lidarelev_fullspan.shape)
percentnan_fullspan_round2[:] = np.nan
thresh = 0.4
for tt in np.arange(len(time_fullspan)):
    span = 18
    halfspan = int(span/2)
    if np.sum(np.isnan(zinput[:,tt])) < lidar_xFRF.size:
        for jj in np.arange(halfspan,lidar_xFRF.size-halfspan):
            ztmp = zinput[np.arange(jj-halfspan,jj+halfspan),tt]
            percentnan = np.sum(np.isnan(ztmp))/ztmp.size
            percentnan_fullspan_round2[jj,tt] = percentnan
# find and remove where percent nan is less than threshhold and below z
ii_testremove = (percentnan_fullspan_round2 > thresh) & (zinput < 0)
yplot = np.empty(shape=zinput.shape)
yplot[:] = zinput[:]
yplot[ii_testremove] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot)
ax.set_title('Remove high-freq nan')
yplot[:] = zinput[:]
yplot[~ii_testremove] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot)
ax.set_title('Discarded values')

final_profile_fullspan = np.empty(shape=zsmooth_fullspan.shape)
yplot[:] = zinput[:]
yplot[ii_testremove] = np.nan
final_profile_fullspan[:] = yplot[:]
with open(picklefile_dir+'final_profile_28Oct2024.pickle','wb') as file:
    pickle.dump(final_profile_fullspan,file)
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,final_profile_fullspan)

# Plot data as a function of cross-shore distance
zplot_raw = np.sum(~np.isnan(lidarelev_fullspan),axis=1)/time_fullspan.size
zplot_final = np.sum(~np.isnan(final_profile_fullspan),axis=1)/time_fullspan.size
zplot_final[zplot_final == 0] = np.nan
zplot_raw[zplot_raw == 0] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zplot_raw,label='pre-filtering')
ax.plot(lidar_xFRF,zplot_final,label='post-filtering')
fig, ax = plt.subplots()
ax.hist(maxgap_fullspan,bins=np.arange(25))
ax.set_ylabel('Percent nan')
ax.set_xlabel('xFRF [m]')

# Plot the profiles that were in the original data that aren't anymore...
isprofile_raw = np.nansum(~np.isnan(lidarelev_fullspan),axis=0) > 0
isprofile_final = np.nansum(~np.isnan(final_profile_fullspan),axis=0) > 0
tmp = isprofile_raw & ~isprofile_final
ii_lostprofile = np.where(tmp)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_removeStrays_fullspan_Xshore[:,ii_lostprofile])
ax.plot(lidar_xFRF,final_profile_fullspan)

# Find avg slope and remove flat with low elevation
tossed_profiles = lidarelev_removeStrays_fullspan_Xshore[:,ii_lostprofile]
tossed_times = time_fullspan[ii_lostprofile]
z_slopeanalysis = np.empty(shape=tossed_profiles.shape)
z_slopeanalysis[:] = tossed_profiles[:]
avgslope_tossed = np.empty(shape=tossed_profiles.shape)
avgslope_tossed[:] = np.nan
dx = 0.1
for tt in np.arange(tossed_profiles.shape[1]):
    ztmp = z_slopeanalysis[:,tt]
    ix_notnan = np.where(~np.isnan(ztmp))[0]
    # If profile is not all nans - calculate the average slope over nslopecheck
    if np.sum(~np.isnan(ztmp)) > 0:
        nslopecheck = 8
        delta = 2
        z_for_slopecheck = ztmp
        iterrange = ix_notnan[delta:-delta]
        for jj in np.arange(iterrange.size):
            tmpslope = np.empty(shape=(int(np.floor(nslopecheck / 2)),))
            tmpslope[:] = np.nan
            jj0 = iterrange[jj] - int(np.floor(nslopecheck * 0.75))
            for ii in np.arange(tmpslope.size):
                tmpslope_ii = (z_for_slopecheck[jj0] - z_for_slopecheck[jj0 + nslopecheck]) / (nslopecheck * dx)
                tmpslope[ii] = tmpslope_ii
                jj0 = jj0 + 1
            if sum(~np.isnan(tmpslope)) >= 2:
                avgslope_tossed[iterrange[jj], tt] = np.nanmean(tmpslope)
# Plot what was removed and what was kept
ii_testremove = (abs(avgslope_tossed) < 0.02 ) & (z_slopeanalysis < 1)
# tmp = z_slopeanalysis[ii_testremove]
# fig, ax = plt.subplots()
# ax.hist(tmp,np.arange(0,8,.25))
yplot = np.empty(shape=z_slopeanalysis.shape)
yplot[:] = z_slopeanalysis[:]
yplot[ii_testremove] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot)
ax.set_title('Remove |slp| < 0.02')
yplot[:] = z_slopeanalysis[:]
yplot[~ii_testremove] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot)
ax.set_title('Discarded values')
tossed_profiles_step2 = np.empty(shape=tossed_profiles.shape)
tossed_profiles_step2[:] = np.nan
tossed_profiles_step2[:] = tossed_profiles[:]

# find peaks
fig, ax = plt.subplots()
peak_x = []
peak_z = []
peak_id = []
peak_start = []
peak_end = []
tossed_removepks = np.empty(tossed_profiles_step2.shape)
tossed_removepks[:] = tossed_profiles_step2[:]
for tt in np.arange(tossed_profiles.shape[1]):
    ztmp = tossed_removepks[:,tt]
    if np.sum(~np.isnan(ztmp)) > 0:
        # fig, ax = plt.subplots()
        # ax.plot(lidar_xFRF,ztmp)
        pklocs, pkprops = sp.signal.find_peaks(ztmp, prominence=0.5, width=0)
        # ax.plot(lidar_xFRF[pklocs],ztmp[pklocs],'kx')
        if pklocs.size > 0:
            peak_x = np.append(peak_x,lidar_xFRF[pklocs])
            peak_z = np.append(peak_z,ztmp[pklocs])
            peak_id = np.append(peak_id,tt)
            for pp in np.arange(pklocs.size):
                if ztmp[pklocs[pp]] < 6:
                    pkstart = np.round(pkprops['left_ips'][pp]).astype(int)
                    pkend = np.round(pkprops['right_ips'][pp]).astype(int)
                    pkwid = pkend - pkstart
                    pkwidover2 = np.round(pkwid/2).astype(int)
                    pkstrt = (pkstart-pkwidover2).astype(int)
                    pkend = (pkend+pkwidover2).astype(int)
                    peak_start = np.append(peak_start,pkstrt)
                    peak_end = np.append(peak_end,pkend)
                    ztmp[np.arange((pkstrt-pkwidover2).astype(int),(pkend+pkwidover2).astype(int))] = np.nan
            ax.plot(lidar_xFRF,ztmp)

# Plot
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,tossed_removepks[:,peak_id.astype(int)])
ax.plot(peak_x,peak_z,'kx')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,tossed_profiles_step2[:,peak_id.astype(int)])
ax.plot(peak_x,peak_z,'kx')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,tossed_profiles_step2)
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,tossed_removepks)

final_profile_fullspan_addtossed = np.empty(shape=final_profile_fullspan.shape)
final_profile_fullspan_addtossed[:] = final_profile_fullspan[:]
final_profile_fullspan_addtossed[:,ii_lostprofile] = tossed_removepks
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed)
with open(picklefile_dir+'final_profile_03Nov2024.pickle','wb') as file:
    pickle.dump(final_profile_fullspan_addtossed,file)


# What is the gap situation for the final combined data?
maxgap_fullspan_combined = np.empty(shape=maxgap_fullspan.shape)
gapstart_fullspan_combined = np.empty(shape=maxgap_fullspan.shape)
gapend_fullspan_combined = np.empty(shape=maxgap_fullspan.shape)
maxgap_fullspan_combined[:] = np.nan
gapstart_fullspan_combined[:] = np.nan
gapend_fullspan_combined[:] = np.nan
for tt in np.arange(time_fullspan.size):
    ztmp = final_profile_fullspan_addtossed[:,tt]
    if sum(~np.isnan(ztmp)) > 3:
        ix_notnan = np.where(~np.isnan(ztmp))[0]
        ztmp = ztmp[np.arange(ix_notnan[0],ix_notnan[-1])]
        gapstart, gapend, gapsize, maxgap = find_nangaps(ztmp)
        maxgap_fullspan_combined[tt] = maxgap
        if maxgap > 0:
            if len(gapstart) > 1:
                ii = max(np.where(gapsize == maxgap)[0])
                gapstart_fullspan_combined[tt] = gapstart[ii] + ix_notnan[0]
                gapend_fullspan_combined[tt] = gapend[ii] + ix_notnan[0]
            else:
                gapstart_fullspan_combined[tt] = gapstart[0] + ix_notnan[0]
                gapend_fullspan_combined[tt] = gapend[0] + ix_notnan[0]
fig, ax = plt.subplots()
ax.plot(maxgap_fullspan_combined,'o')
# plot where profiles maxgap > 200
iiplot = np.where(maxgap_fullspan_combined >= 200)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iiplot])
for tt in iiplot:
    itmp = gapstart_fullspan_combined[tt].astype(int)-1
    tmpx = lidar_xFRF[itmp]
    tmpy = final_profile_fullspan_addtossed[itmp,tt]
    ax.plot(tmpx,tmpy,'xk')
for tt in iiplot:
    iitoss = np.arange(gapend_fullspan_combined[tt].astype(int),lidar_xFRF.size)
    final_profile_fullspan_addtossed[iitoss,tt] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iiplot])
for tt in iiplot:
    ztmp = np.empty(shape=lidar_xFRF.shape)
    ztmp[:] = final_profile_fullspan_addtossed[:, tt]
    ix_notnan = np.where(~np.isnan(ztmp))[0]
    ztmp = ztmp[np.arange(ix_notnan[0], ix_notnan[-1])]
    xtmp = lidar_xFRF[np.arange(ix_notnan[0], ix_notnan[-1])]
    gapstart, gapend, gapsize, maxgap = find_nangaps(ztmp)
    # kernel = Gaussian1DKernel(3*maxgap)
    xin = xtmp[~np.isnan(ztmp)]
    yin = ztmp[~np.isnan(ztmp)]
    cs = CubicSpline(xin, yin, bc_type='natural')
    zspline_tmp = cs(xtmp)
    # ax.plot(xtmp,zspline_tmp,'o')
    final_profile_fullspan_addtossed[np.arange(ix_notnan[0], ix_notnan[-1]), tt] = zspline_tmp

# Now investigate profiles where max gap is between 50 and 200
iiplot = np.where((maxgap_fullspan_combined >= 100) & (maxgap_fullspan_combined < 200))[0]
for ii in np.arange(0,iiplot.size,10):
    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iiplot[ii:ii+10]])
    plt.legend(iiplot[ii:ii+10])

# I think we can just use a cubic spline for the rest of them...?
iiplot = np.where((maxgap_fullspan_combined < 200))[0]
for tt in iiplot:
    ztmp = np.empty(shape=lidar_xFRF.shape)
    ztmp[:] = final_profile_fullspan_addtossed[:, tt]
    ix_notnan = np.where(~np.isnan(ztmp))[0]
    ztmp = ztmp[np.arange(ix_notnan[0], ix_notnan[-1])]
    xtmp = lidar_xFRF[np.arange(ix_notnan[0], ix_notnan[-1])]
    xin = xtmp[~np.isnan(ztmp)]
    yin = ztmp[~np.isnan(ztmp)]
    cs = CubicSpline(xin, yin, bc_type='natural')
    zspline_tmp = cs(xtmp)
    final_profile_fullspan_addtossed[np.arange(ix_notnan[2], ix_notnan[-3]), tt] = zspline_tmp

fig, ax = plt.subplots()
vals = np.empty(shape=final_profile_fullspan_addtossed.shape)
vals[:] = final_profile_fullspan_addtossed[:]
final_profile_fullspan_addtossed[(vals > 10)] = np.nan
final_profile_fullspan_addtossed[(vals < -1.3)] = np.nan
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed)
prof_mean = np.nanmean(final_profile_fullspan_addtossed,axis=1)
prof_std = np.nanstd(final_profile_fullspan_addtossed,axis=1)
ax.plot(lidar_xFRF,prof_mean,'k')
ax.plot(lidar_xFRF,prof_mean+3*prof_std,'k:')
ax.plot(lidar_xFRF,prof_mean-3*prof_std,'k:')
lowerlim = prof_mean-3*prof_std-0.25
upperlim = prof_mean+3*prof_std+0.25
for tt in np.arange(time_fullspan.size):
    tmp = vals[:,tt]
    if sum(~np.isnan(tmp)) > 4:
        iitmp = np.where((tmp < lowerlim) | (tmp > upperlim))[0]
        final_profile_fullspan_addtossed[iitmp,tt] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed)

# Do first-order slope
simpleslp = (final_profile_fullspan_addtossed[0:-1,:] - final_profile_fullspan_addtossed[1:,:])/dx
# iiflag = np.where(simpleslp > 5)
# fig, ax = plt.subplots()
# ax.plot(lidar_xFRF,final_profile_fullspan_addtossed)
XX, TT = np.meshgrid(lidar_xFRF[1:], time_fullspan)
xplot = np.reshape(XX, XX.size)
splot = np.reshape(simpleslp.T, simpleslp.size)
zplot = np.reshape(final_profile_fullspan_addtossed[1:,:].T, final_profile_fullspan_addtossed[1:,:].size)
xplot = xplot[splot < -2]
zplot = zplot[splot < -2]
# fig, ax = plt.subplots()
ax.plot(xplot,zplot,'*k')

iibadslp = np.abs(simpleslp) > 2
iibeach = final_profile_fullspan_addtossed[1:,:] < dune_toe+0.25
iirowcol = np.column_stack(np.where(np.logical_and(iibadslp,iibeach)))
iicol_badslp = iirowcol[:,1]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iirowcol[:,1]])
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iirowcol[:,1]],'o')
# define this weird set for spline fix
profile_splinefix = final_profile_fullspan_addtossed[:,iirowcol[:,1]]
slp_splinefix = simpleslp[:,iirowcol[:,1]]

fig, ax = plt.subplots()
ax.plot(lidar_xFRF[1:],simpleslp[:,iirowcol[:,1]],'o')
ax.set_ylim(-2,2)

# Ok, remove points where slope meets criterion per xshore-section (see FRF_Waves for visual)
ztmp_check = np.empty(shape=profile_splinefix.shape)
ztmp_check[:] = profile_splinefix[:]
slp_check = np.empty(shape=slp_splinefix.shape)
slp_check[:] = slp_splinefix[:]
# Section 1 - x = 80-105, upperlim = 0.3, lowerlim = -0.175
upperlim = 0.3
lowerlim = -0.175
xx_inspan = np.where((lidar_xFRF >= 80) & (lidar_xFRF <= 105))[0]
xplot = lidar_xFRF
yplot = np.empty(shape=ztmp_check.shape)
yplot[:] = np.nan
iirowcol = np.column_stack(np.where(slp_check[xx_inspan,:] > upperlim))
yplot[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]] = ztmp_check[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]]
iirow_upper1 = iirowcol[:,0]+1+xx_inspan[0]
iicol_upper1 = iicol_badslp[iirowcol[:,1]]
iirowcol = np.column_stack(np.where(slp_check[xx_inspan,:] < lowerlim))
yplot[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]] = ztmp_check[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]]
iirow_lower1 = iirowcol[:,0]+1+xx_inspan[0]
iicol_lower1 = iicol_badslp[iirowcol[:,1]]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,profile_splinefix)
ax.plot(xplot,yplot,'k*')
# Section 2 - x = 105+, upperlim = 0.175, lowerlim = -0.5
upperlim = 0.175
lowerlim = -0.5
xx_inspan = np.where((lidar_xFRF >= 105))[0][0:-1]
xplot = lidar_xFRF
yplot = np.empty(shape=ztmp_check.shape)
yplot[:] = np.nan
iirowcol = np.column_stack(np.where(slp_check[xx_inspan,:] > upperlim))
yplot[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]] = ztmp_check[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]]
iirow_upper2 = iirowcol[:,0]+1+xx_inspan[0]
iicol_upper2 = iicol_badslp[iirowcol[:,1]]
iirowcol = np.column_stack(np.where(slp_check[xx_inspan,:] < lowerlim))
yplot[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]] = ztmp_check[iirowcol[:,0]+1+xx_inspan[0],iirowcol[:,1]]
iirow_lower2 = iirowcol[:,0]+1+xx_inspan[0]
iicol_lower2 = iicol_badslp[iirowcol[:,1]]
ax.plot(xplot,yplot,'k^')

# let's check to make sure that we remove the correct values in the full profile set
ztmp_check = np.empty(shape=final_profile_fullspan_addtossed.shape)
ztmp_check[:] = np.nan
ztmp_check[:] = final_profile_fullspan_addtossed[:]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,ztmp_check)
ztmp_check[iirow_upper1,iicol_upper1] = np.nan
ztmp_check[iirow_upper2,iicol_upper2] = np.nan
ztmp_check[iirow_lower1,iicol_lower1] = np.nan
ztmp_check[iirow_lower2,iicol_lower2] = np.nan
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,ztmp_check)
final_profile_fullspan_best = np.empty(shape=final_profile_fullspan_addtossed.shape)
final_profile_fullspan_best[:] = np.nan
final_profile_fullspan_best[:] = ztmp_check[:]

# SAVE CRITICAL
with open(picklefile_dir+'final_profile_13Nov2024.pickle','wb') as file:
    pickle.dump(final_profile_fullspan_best,file)

# What does the data look like?
fig, ax = plt.subplots()
yplot = 100*np.nansum(~np.isnan(lidarelev_fullspan),axis=1)/time_fullspan.size
ax.plot(lidar_xFRF,yplot,label='raw')
yplot = 100*np.nansum(~np.isnan(final_profile_fullspan_best),axis=1)/time_fullspan.size
ax.plot(lidar_xFRF,yplot,label='processed')
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('Percent available')

## Shift the profiles to the dune_toe contour
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'final_profile_13Nov2024.pickle', 'rb') as file:
    final_profile_fullspan_best = pickle.load(file)

# get the contour position
dune_toe = 3.22
cont_elev = np.array([dune_toe]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(final_profile_fullspan_best.T,time_fullspan,lidar_xFRF,cont_elev)
xc_fullspan = cont_ts[0,:]
fig, ax = plt.subplots()
ax.plot(time_fullspan, xc_fullspan,'o')

finalprofile_13Nov2024_shift = np.empty(shape=final_profile_fullspan_best.shape)
finalprofile_13Nov2024_shift[:] = np.nan
zsmooth_fullspan = np.empty(shape=final_profile_fullspan_best.shape)
zsmooth_fullspan[:] = final_profile_fullspan_best[:]
for tt in np.arange(len(time_fullspan)):
    xc_shore = xc_fullspan[tt]
    if (~np.isnan(xc_shore)):
        # first, map to *_shift vectors
        ix_inspan = np.where((lidar_xFRF >= xc_shore))[0]
        padding = 2
        itrim = np.arange(ix_inspan[0] - padding, lidar_xFRF.size)
        xtmp = lidar_xFRF[itrim]
        ztmp = zsmooth_fullspan[itrim, tt]
        xtmp = xtmp[~np.isnan(ztmp)]        # remove nans
        ztmp = ztmp[~np.isnan(ztmp)]        # remove nans
        xinterp = np.linspace(xc_shore, np.nanmax(xtmp), xtmp.size-(padding-1))
        zinterp = np.interp(xinterp, xtmp, ztmp)
        ztrim_FromXCshore = zinterp
        finalprofile_13Nov2024_shift[0:itrim.size, tt] = zsmooth_fullspan[itrim, tt]
        # finalprofile_13Nov2024_shift[0:ztrim_FromXCshore.size,tt] = ztrim_FromXCshore

fig, ax = plt.subplots()
ax.plot(lidar_xFRF, finalprofile_13Nov2024_shift)

final_profile_fullspan_best_shift = np.empty(shape=finalprofile_13Nov2024_shift.shape)
final_profile_fullspan_best_shift[:] = finalprofile_13Nov2024_shift[:]


## Shift the profiles to the dune_toe contour
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'final_profile_13Nov2024_shift.pickle','wb') as file:
    pickle.dump(final_profile_fullspan_best_shift,file)



















# # Ok, now pull out slope data  to see if we can use it for filtering
# z_slopeanalysis = np.empty(shape=zsmooth_fullspan.shape)
# z_slopeanalysis[:] = zsmooth_fullspan[:]
# avgslope_fullspan = np.empty(shape=lidarelev_fullspan.shape)
# avgslope_fullspan[:] = np.nan
# dx = 0.1
# for tt in np.arange(len(time_fullspan)):
#     ztmp = z_slopeanalysis[:,tt]
#     ix_notnan = np.where(~np.isnan(ztmp))[0]
#     # If profile is not all nans - calculate the average slope over nslopecheck
#     if np.sum(~np.isnan(ztmp)) > 0:
#         nslopecheck = 8
#         delta = 2
#         z_for_slopecheck = ztmp
#         iterrange = ix_notnan[delta:-delta]
#         for jj in np.arange(iterrange.size):
#             tmpslope = np.empty(shape=(int(np.floor(nslopecheck / 2)),))
#             tmpslope[:] = np.nan
#             jj0 = iterrange[jj] - int(np.floor(nslopecheck * 0.75))
#             for ii in np.arange(tmpslope.size):
#                 tmpslope_ii = (z_for_slopecheck[jj0] - z_for_slopecheck[jj0 + nslopecheck]) / (nslopecheck * dx)
#                 tmpslope[ii] = tmpslope_ii
#                 jj0 = jj0 + 1
#             if sum(~np.isnan(tmpslope)) >= 2:
#                 avgslope_fullspan[iterrange[jj], tt] = np.nanmean(tmpslope)


# find and plot profiles where any xshore value is > 3m from the mean xshore val
flag_splineerr = np.empty(shape=time_fullspan.shape)
flag_splineerr[:] = np.nan
for tt in np.arange(time_fullspan.size):
    ztmp = final_profile_fullspan_addtossed[:,tt]
    if sum(~np.isnan(ztmp)) > 5:
        tmp = abs(ztmp - prof_mean)
        if sum(tmp > 3) > 0:
            flag_splineerr[tt] = 1
flag_splineerr[np.isnan(flag_splineerr)] = 0

fig, ax = plt.subplots()
iiplot = np.where(flag_splineerr == 1)
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iiplot[0]],'o')

fig, ax = plt.subplots()
iiplot = np.where(flag_splineerr != 1)
ax.plot(lidar_xFRF,final_profile_fullspan_addtossed[:,iiplot[0]])

# fig, ax = plt.subplots()
fig = plt.figure()
ax = plt.axes(projection='3d')
for tt in np.arange(time_fullspan.size):
    ztmp = zsmooth_fullspan[:,tt]
    if sum(~np.isnan(ztmp)) > 0:
        tdummy = tt*np.ones(shape=ztmp.shape)
        ax.plot3D(lidar_xFRF, ztmp, tdummy)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_zlabel('time')
ax.view_init(-90, 0)


fig, ax = plt.subplots()
istart=24765
tt = np.arange(istart,istart+8)
ax.plot(lidar_xFRF, zsmooth_fullspan[:,tt])
ax.set_title('Lsmooth = '+str(Lsmooth1)+', thresh = '+str(thresh))
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
fig, ax = plt.subplots()
plt.hist(maxgap_fullspan)
ax.set_title('Max gap size post smoothing')


# plot profiles between 20-28 March 2023
tplot = pd.to_datetime(tossed_times, unit='s', origin='unix')
tt_check = (tplot >= '2023-03-20') & (tplot <= '2023-03-28')
tmp = z_slopeanalysis[:]
tmp[ii_testremove] = np.nan
yplot = tmp[:,tt_check]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot)
ax.set_title('March 20-28 2023')

# Calculate the moving average and calculate the difference between the profile and mean
zmean_tossed = np.empty(shape=tossed_profiles.shape)
zmean_tossed[:] = np.nan
zstd_tossed = np.empty(shape=tossed_profiles.shape)
zstd_tossed[:] = np.nan
nsmooth = 15
for tt in np.arange(tossed_profiles.shape[1]):
    ztmp = tossed_profiles_step2[:,tt]
    ix_notnan = np.where(~np.isnan(ztmp))[0]
    # If profile is not all nans - calculate the average slope over nslopecheck
    if np.sum(~np.isnan(ztmp)) > 0:
        zmean_tmp = np.convolve(ztmp, np.ones(nsmooth) / nsmooth, mode='same')
        ts = pd.Series(ztmp)
        zstd_tmp = ts.rolling(window=nsmooth, center=True).std()
        zmean_tossed[np.arange(2,ztmp.size-2),tt] = zmean_tmp[np.arange(2,ztmp.size-2)]
        zstd_tossed[np.arange(2, ztmp.size - 2), tt] = zstd_tmp[np.arange(2,ztmp.size-2)]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zmean_tossed)
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zstd_tossed)





# # PLOT/VERIFY - zsmooth (elevations) and avgslope for full profile
# tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
# XX, TT = np.meshgrid(lidar_xFRF, tplot)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(zsmooth_fullspan.T, zsmooth_fullspan.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('z (smoothed) [m]')
# ax.set_title('Profile elevation (smoothed)')
# zscatter = np.reshape(avgslope_fullspan.T, avgslope_fullspan.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('avg. slope [m/m]')
# ax.set_title('Avg. Slope for profile')

# Now, take fullspan datasets of elevation and slope and shift to same starting point (dune-toe)
zsmooth_fullspan_shift = np.empty(shape=avgslope_fullspan.shape)
zsmooth_fullspan_shift[:] = np.nan
avgslope_fullspan_shift = np.empty(shape=avgslope_fullspan.shape)
avgslope_fullspan_shift[:] = np.nan
nx = 2000
zsmooth_fullspan_scale = np.empty(shape=(nx,time_fullspan.size))
zsmooth_fullspan_scale[:] = np.nan
avgslope_fullspan_scale = np.empty(shape=(nx,time_fullspan.size))
avgslope_fullspan_scale[:] = np.nan
for tt in np.arange(len(time_fullspan)):
    xc_shore = xc_fullspan[-1, tt]
    xc_sea = xc_fullspan[0, tt]
    if (~np.isnan(xc_shore)) & (~np.isnan(xc_sea)):
        # first, map to *_shift vectors
        ix_inspan = np.where((lidar_xFRF >= xc_shore) & (lidar_xFRF <= xc_sea))[0]
        padding = 2
        itrim = np.arange(ix_inspan[0] - padding, lidar_xFRF.size)
        xtmp = lidar_xFRF[itrim]
        ztmp = zsmooth_fullspan[itrim, tt]
        slptmp = avgslope_fullspan[itrim, tt]
        xtmp = xtmp[~np.isnan(ztmp)]        # remove nans
        slptmp = slptmp[~np.isnan(ztmp)]    # remove nans
        ztmp = ztmp[~np.isnan(ztmp)]        # remove nans
        xinterp = np.linspace(xc_shore, np.nanmax(xtmp), xtmp.size-(padding-1))
        zinterp = np.interp(xinterp, xtmp, ztmp)
        slpinterp = np.interp(xinterp, xtmp, slptmp)
        ztrim_FromXCshore = zinterp
        zsmooth_fullspan_shift[0:ztrim_FromXCshore.size,tt] = ztrim_FromXCshore
        avgslope_FromXCshore = slpinterp
        avgslope_fullspan_shift[0:avgslope_FromXCshore.size,tt] = avgslope_FromXCshore
        # then, map to *_scale vectors
        padding = 2
        itrim = np.arange(ix_inspan[0]-padding, ix_inspan[-1]+padding+1)
        xtmp = lidar_xFRF[itrim]
        ztmp = zsmooth_fullspan[itrim,tt]
        slptmp = avgslope_fullspan[itrim,tt]
        # remove nans
        xtmp = xtmp[~np.isnan(ztmp)]
        slptmp = slptmp[~np.isnan(ztmp)]
        ztmp = ztmp[~np.isnan(ztmp)]
        if np.sum(~np.isnan(ztmp)) > 0:
            # create scaled profile (constrain length to be equal between XC_sea and XC_shore0
            xinterp = np.linspace(xc_shore,xc_sea,nx)
            zinterp = np.interp(xinterp, xtmp, ztmp)
            ztrim_BetweenXCs = zinterp
            zsmooth_fullspan_scale[:,tt] = ztrim_BetweenXCs
            slpinterp = np.interp(xinterp, xtmp, slptmp)
            slptrim_BetweenXCs = slpinterp
            avgslope_fullspan_scale[:,tt] = slptrim_BetweenXCs

# # PLOT/VERIFY - zsmooth (elevations) and avgslope for scaled
# tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
# xplot = np.arange(nx)
# XX, TT = np.meshgrid(xplot, tplot)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(zsmooth_fullspan_scale.T, zsmooth_fullspan_scale.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('z (smoothed) [m]')
# ax.set_title('Profile elevation (smoothed) - Scaled for XC_shore to XC_sea')
# fig, ax = plt.subplots()
# zscatter = np.reshape(avgslope_fullspan_scale.T, avgslope_fullspan_scale.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('avg. slope [m/m]')
# ax.set_title('Avg. Slope for profile - Scaled')
#
# # PLOT/VERIFY - zsmooth (elevations) and avgslope for shifted
# tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
# xplot = np.arange(zsmooth_fullspan_shift.shape[0])
# XX, TT = np.meshgrid(xplot, tplot)
# timescatter = np.reshape(TT, TT.size)
# xscatter = np.reshape(XX, XX.size)
# zscatter = np.reshape(zsmooth_fullspan_shift.T, zsmooth_fullspan_shift.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('z (smoothed) [m]')
# ax.set_title('Profile elevation (smoothed) - Shifted XCshore~0m')
# fig, ax = plt.subplots()
# zscatter = np.reshape(avgslope_fullspan_shift.T, avgslope_fullspan_shift.size)
# tt = timescatter[~np.isnan(zscatter)]
# xx = xscatter[~np.isnan(zscatter)]
# zz = zscatter[~np.isnan(zscatter)]
# fig, ax = plt.subplots()
# ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('avg. slope [m/m]')
# ax.set_title('Avg. Slope for profile - Shifted')


# SAVE all the permutations of Zsmooth & AvgSlope for full dataset
picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_backup/'
with open(picklefile_dir+'elev&slp_processed_fullspan.pickle', 'wb') as file:
    pickle.dump([zsmooth_fullspan,avgslope_fullspan], file)
with open(picklefile_dir+'elev&slp_process+shift_fullspan.pickle', 'wb') as file:
    pickle.dump([zsmooth_fullspan_shift, avgslope_fullspan_shift], file)
with open(picklefile_dir + 'elev&slp_process+scale_fullspan.pickle', 'wb') as file:
    pickle.dump([zsmooth_fullspan_scale, avgslope_fullspan_scale], file)
with open(picklefile_dir + 'contour_fullspan.pickle', 'wb') as file:
    pickle.dump([xc_fullspan,dXcdt_fullspan], file)

## What do the profiles look like where they have all-nans in the Zsmooth but not in the raw lidar?
# Primary reason for unequal number of profiles comes from lack of data at the *edge* of the scan (not the middle)
allnan_zsmooth = np.where(np.sum(np.isnan(zsmooth_fullspan),axis=0 ) == 1551)[0]
allnan_raw = np.where(np.sum(np.isnan(lidarelev_fullspan),axis=0 ) == 1551)[0]
allnan_plot = allnan_zsmooth[~np.isin(allnan_zsmooth,allnan_raw)]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_fullspan[:,allnan_plot])
