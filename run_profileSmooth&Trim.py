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
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,xc_fullspan,dXcdt_fullspan,lidarelev_fullspan = pickle.load(file)
full_path = 'C:/Users/rdchlerh/Desktop/FRF_data/dune_lidar/lidar_transect/FRF-geomorphology_elevationTransects_duneLidarTransect_201510.nc'
_, _, _, _, _, lidar_xFRF, lidar_yFRF = (getlocal_lidar(full_path))

def find_nangaps(zinput):
    if sum(np.isnan(zinput)) == 0:
        gapstart = np.nan
        gapend = np.nan
        gapsize = 0
        maxgap = 0
    elif sum(np.isnan(zinput)) == 1:
        gapstart = np.where(np.isnan(zinput))
        gapend = np.where(np.isnan(zinput))
        gapsize = 1
        maxgap = 1
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
            # If there is only ONE entry of a cumnan value, then we know it's a new nan value
            if sum((numcumnan == uniq_numcumnan[ij])) == 1:
                tmp = np.where(numcumnan == uniq_numcumnan[ij])[0]
                tmpgapstart = np.append(tmpgapstart,tmp[0])
            # If there are multiple entries of a cumnan value, then we know it switches from nan to not-nan
            elif sum((numcumnan == uniq_numcumnan[ij])) > 1:
                tmp = np.where(numcumnan == uniq_numcumnan[ij])[0]
                # the first value of tmp is where it switches from nan to not-nan, the last would be the first before the next nan (if it exists)
                tmpgapend = np.append(tmpgapend,tmp[0])
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
                    if ~np.isnan(zinput[int(gapend[ij])-1]) & ~np.isnan(zinput[int(gapend[ij])+1]):
                        missinggapstart = gapend[ij]
                        gapstart = np.append(missinggapstart, gapstart)
            if np.max(gapstart) > np.max(gapend):
                gapend = np.append(gapend, np.max(gapstart))
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
with open(picklefile_dir+'xc_fullspan_MWL_DuneToe_MHW.pickle', 'wb') as file:
    pickle.dump([xc_fullspan,dXcdt_fullspan], file)

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

# Load profiles, remove locally high variability - use whole profile, but only if we have data between Xc[0] & Xc[-1]
zsmooth_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zpass_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zbad_fullspan = np.empty(shape=lidarelev_fullspan.shape)
avgslope_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zsmooth_fullspan[:] = np.nan
zbad_fullspan[:] = np.nan
zpass_fullspan[:] = np.nan
avgslope_fullspan[:] = np.nan
maxgap_presmooth = np.empty((time_fullspan.size,))
maxgap_postsmooth = np.empty((time_fullspan.size,))
maxgap_fullspan = np.empty((time_fullspan.size,))
maxgap_presmooth[:] = np.nan
maxgap_postsmooth[:] = np.nan
maxgap_fullspan[:] = np.nan
smooth_to_nan = np.empty((time_fullspan.size,))
smooth_to_nan[:] = np.nan
for tt in np.arange(len(time_fullspan)):
    xc_shore = xc_fullspan[-1,tt]
    xc_sea = xc_fullspan[0,tt]
    ztmp = lidarelev_fullspan[:, tt]
    # We can perform the smoothing if we have data even if XC not found
    near_xc_thresh = 0.5        # unit of [m], vertical
    if ((~np.isnan(xc_shore)) & (~np.isnan(xc_sea))) or ((np.nanmin(ztmp) <= mwl+near_xc_thresh) & (np.nanmax(ztmp) >= dune_toe-near_xc_thresh)):
        ztmp = lidarelev_fullspan[:,tt]
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
            Lsmooth = 1.5
            dx = 0.1
            nsmooth = int(np.ceil(Lsmooth / dx))
            zbad, zgood = smooth_profile(xtmp, ztmp, xtmp, nsmooth, 1.5)
            zbad_fullspan[ix_notnan, tt] = zbad
            zpass_fullspan[ix_notnan,tt] = zgood
            # What are the gaps between the contours?
            ztmp = zgood
            tmp = (xtmp >= xc_shore) & (xtmp <= xc_sea)
            ix_inspan = np.where(tmp)[0]
            gapstart, gapend, gapsize, maxgap = find_nangaps(ztmp[ix_inspan])
            maxgap_postsmooth[tt] = maxgap
            # Second smooth - refine if enough data exists
            delta = 2
            if (maxgap <= maxgapthresh):
                Lsmooth = 0.5
                dx = 0.1
                nsmooth = int(np.ceil(Lsmooth / dx))
                mykernel = np.ones(nsmooth)/nsmooth
                padding = np.zeros((nsmooth,))*2
                zinput = np.hstack((padding, ztmp, padding))
                zconvolved = convolve(zinput, mykernel, boundary='extend')
                zsmooth_fullspan[ix_notnan[delta:-delta],tt] = zconvolved[nsmooth+delta:-(nsmooth+delta)]
                ztmp = zconvolved[nsmooth+delta:-(nsmooth+delta)]
                gapstart, gapend, gapsize, maxgap = find_nangaps(ztmp)
                maxgap_fullspan[tt] = maxgap




# Ok, now pull out slope data for the same coverage as zsmooth_fullspan
avgslope_fullspan = np.empty(shape=lidarelev_fullspan.shape)
avgslope_fullspan[:] = np.nan
for tt in np.arange(len(time_fullspan)):
    ztmp = zsmooth_fullspan[:,tt]
    ix_notnan = np.where(~np.isnan(ztmp))[0]
    # If profile is not all nans - calculate the average slope over nslopecheck
    if np.sum(~np.isnan(ztmp)) > 0:
        nslopecheck = 12
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


# fig, ax = plt.subplots()
# ax.plot(xplot,zsmooth_fullspan_shift)
# ax.set_title('Profile elevation (smoothed) - Shifted XCshore~0m')

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
