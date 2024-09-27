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
with open('IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,xc_fullspan,dXcdt_fullspan,lidarelev_fullspan = pickle.load(file)
full_path = 'C:/Users/rdchlerh/Desktop/FRF_data/dune_lidar/lidar_transect/FRF-geomorphology_elevationTransects_duneLidarTransect_201510.nc'
qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF = (
            getlocal_lidar(full_path))

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
                        gapstart = np.append(gapstart, missinggapstart)
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


# Load profiles, do remove locally high variability - use whole profile, but only if we have data between Xc[0] & Xc[-1]
zsmooth_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zpass_fullspan = np.empty(shape=lidarelev_fullspan.shape)
zbad_fullspan = np.empty(shape=lidarelev_fullspan.shape)
avgslope_fullspan = np.empty(shape=lidarelev_fullspan.shape)
avgslope_withinXCs = np.empty(shape=lidarelev_fullspan.shape)
avgslope_beyondXCsea = np.empty(shape=lidarelev_fullspan.shape)
zsmooth_fullspan[:] = np.nan
zbad_fullspan[:] = np.nan
zpass_fullspan[:] = np.nan
avgslope_fullspan[:] = np.nan
avgslope_withinXCs[:] = np.nan
avgslope_beyondXCsea[:] = np.nan
flag = np.zeros((time_fullspan.size,)).astype(bool)
smallgap = np.zeros((time_fullspan.size,)).astype(bool)
nownan = np.zeros((time_fullspan.size,)).astype(bool)
maxgap_fullspan = np.empty((time_fullspan.size,))
maxgap_fullspan[:] = np.nan
data_between_xc = np.zeros((time_fullspan.size,)).astype(bool)
for tt in np.arange(len(time_fullspan)):
# for tt in np.arange(0,100):
    xc_shore = xc_fullspan[-1,tt]
    xc_sea = xc_fullspan[0,tt]
    if (~np.isnan(xc_shore)) & (~np.isnan(xc_sea)):
        ztmp = lidarelev_fullspan[:,tt]
        ix_notnan = np.where(~np.isnan(ztmp))[0]
        approxlength = ix_notnan[-1] - ix_notnan[0]
        # zinput = ztmp[np.arange(ix_notnan[0],ix_notnan[-1])]
        # gapstart, gapend, gapsize, maxgap = find_nangaps(zinput)
        # maxgap_fullspan[tt] = maxgap
        data_between_xc[tt] = 1
        # only do smoothing routine if we have data between Xc[0] and Xc[-1] and there's >60% not-nans across prof
        if (ix_notnan.size/approxlength) > 0.6:
            xtmp = lidar_xFRF[ix_notnan]
            ztmp = ztmp[ix_notnan]
            # First Smooth - remove extreme points
            Lsmooth = 1.5
            dx = 0.1
            nsmooth = int(np.ceil(Lsmooth / dx))
            zbad, zgood = smooth_profile(xtmp, ztmp, xtmp, nsmooth, 1.5)
            ztmp = zgood
            zbad_fullspan[ix_notnan, tt] = zbad
            # # Second Smooth??
            # Lsmooth = 0.5
            # dx = 0.1
            # nsmooth = int(np.ceil(Lsmooth / dx))
            # zbad, zgood = smooth_profile(xtmp, ztmp, xtmp, nsmooth, 0.5)
            zpass_fullspan[ix_notnan,tt] = zgood
            gapstart, gapend, gapsize, maxgap = find_nangaps(ztmp)
            maxgap_fullspan[tt] = maxgap
            delta = 2
            if (maxgap <= 10):
                Lsmooth = 0.5
                dx = 0.1
                nsmooth = int(np.ceil(Lsmooth / dx))
                mykernel = np.ones(nsmooth)/nsmooth
                padding = np.zeros((nsmooth,))*2
                zinput = np.hstack((padding, ztmp, padding))
                zconvolved = convolve(zinput, mykernel, boundary='extend')
                zsmooth_fullspan[ix_notnan[delta:-delta],tt] = zconvolved[nsmooth+delta:-(nsmooth+delta)]
                xsmooth = lidar_xFRF[ix_notnan[delta:-delta]]
                smallgap[tt] = 1
                z_for_slopecheck = zsmooth_fullspan[:,tt]
            else:
                z_for_slopecheck = zpass_fullspan[:,tt]
            # calculate the average slope over nslopecheck
            nslopecheck = 12
            iterrange = ix_notnan[delta:-delta]
            for jj in np.arange(iterrange.size):
                tmpslope = np.empty(shape=(int(np.floor(nslopecheck / 2)),))
                tmpslope[:] = np.nan
                jj0 = iterrange[jj] - int(np.floor(nslopecheck * 0.75))
                for ii in np.arange(tmpslope.size):
                    tmpslope_ii = (z_for_slopecheck[jj0] - z_for_slopecheck[jj0+nslopecheck])/(nslopecheck*dx)
                    tmpslope[ii] = tmpslope_ii
                    jj0 = jj0+1
                if sum(~np.isnan(tmpslope)) >= 2:
                    avgslope_fullspan[iterrange[jj],tt] = np.nanmean(tmpslope)
            # shift the slope data, sort to between within xc_shore-xc_sea and seaward of xc_sea
            tmpdiff = lidar_xFRF - xc_sea
            ii_sea = np.where(tmpdiff == np.nanmin(tmpdiff[tmpdiff > 0]))[0] + 1
            tmpdiff = lidar_xFRF - xc_shore
            ii_shore = np.where(tmpdiff == np.nanmin(tmpdiff[tmpdiff > 0]))[0] - 1
            avgslope_withinXCs[np.arange(ii_shore,ii_sea),tt] = avgslope_fullspan[np.arange(ii_shore,ii_sea),tt]
            avgslope_beyondXCsea[np.arange(ii_sea,lidar_xFRF.size), tt] = avgslope_fullspan[np.arange(ii_sea,lidar_xFRF.size), tt]
            if sum(~np.isnan(zgood)) > 0:
                flag[tt] = 1
            else:
                nownan[tt] = 1

# Plot the range of maxgap sizes
var = maxgap_fullspan[~np.isnan(maxgap_fullspan)]
var = var[var < 50]
fig, ax = plt.subplots()
ax.hist(var, density=True)
ax.set_title('max gapsize, N = ' + str(sum(~np.isnan(var))))
ax.plot([np.nanmean(var), np.nanmean(var)], [0, 0.2], 'k')

# Check how well convolution smoothing works
num_smallflag = sum(smallgap)
# num_smallflag = 5
# smallflag_id = np.where(smallgap == 1)[0]
smallflag_id = np.where((maxgap_fullspan > 6) & (maxgap_fullspan <= 10))[0]
# for ij in np.arange(num_smallflag):
for ij in (np.floor(np.linspace(0,smallflag_id.size-1,15))):
    tmpid = smallflag_id[int(ij)]
    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF, zpass_fullspan[:,tmpid], '*y')
    ztmp = zpass_fullspan[:,tmpid]
    ax.plot(lidar_xFRF[np.isnan(ztmp)], np.zeros(shape=(lidar_xFRF[np.isnan(ztmp)]).shape), '*r')
    ax.plot(lidar_xFRF, zsmooth_fullspan[:,tmpid], 'ob')
    ax.grid(which='major', axis='both')

# what's the average slope along the profiles as a function of beach width?
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(avgslope_fullspan.T,avgslope_fullspan.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
# ax.set_xlim(95,180)
ax.set_title('full profile, avg slope')
ax.plot(xc_fullspan[0,:],tplot,'*k')
xc_sea = xc_fullspan[0,:]
zfilt = avgslope_fullspan.T[:]
for tt in np.arange(time_fullspan.size):
    zfilt[tt,lidar_xFRF <= xc_sea[tt]] = np.nan
zfiltscatter = np.reshape(zfilt,zfilt.size)
tt = timescatter[~np.isnan(zfiltscatter)]
xx = xscatter[~np.isnan(zfiltscatter)]
zz = zfiltscatter[~np.isnan(zfiltscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_xlim(95,180)
# PLOT - avgslope for profile between XCshore and XCsea
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(avgslope_withinXCs.T, avgslope_withinXCs.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_title('Avg. Slope for profile between XCshore and XCsea')
# PLOT - profile elevations between XCshore and XCsea
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zplot = zsmooth_fullspan[:]
zplot[np.isnan(avgslope_withinXCs)] = np.nan
zscatter = np.reshape(zplot.T, zplot.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='viridis')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_title('Elev. between XCshore and XCsea')

# PLOT - avgslope for profile beyond XCsea
zscatter = np.reshape(avgslope_beyondXCsea.T,avgslope_beyondXCsea.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_title('Avg. Slope for profile beyond XCsea')

# Plot the results of the filtering/convolution
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zpass_fullspan)
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_fullspan[:,flag])
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zsmooth_fullspan)


# find the number of profiles that are full between Xc(0) and Xc(end), scale them
nx = 2000
nt = time_fullspan.shape[0]
profile_width = np.empty(nt,)
scaled_profiles = np.empty((nt,nx))
scaled_avgslope = np.empty((nt,nx))
shift_avgslope = np.empty((nt,nx))
shift_avgslope_beyondXCsea = np.empty((nt,nx))
shift_zsmooth = np.empty((nt,nx))
Lkeep = 95
dx = 0.1
nkeep = int(np.ceil(Lkeep / dx))
unscaled_profile = np.empty((nt,nkeep))
profile_width[:] = np.nan
scaled_profiles[:] = np.nan
scaled_avgslope[:] = np.nan
shift_avgslope[:] = np.nan
shift_avgslope_beyondXCsea[:] = np.nan
shift_zsmooth[:] = np.nan
unscaled_profile[:] = np.nan
unscaled_xi = np.empty((nt,))
unscaled_xi[:] = np.nan
for tt in np.arange(nt):
    xc_shore = xc_fullspan[-1,tt]
    xc_sea = xc_fullspan[0,tt]
    if (~np.isnan(xc_shore)) & (~np.isnan(xc_sea)):
        tmp = (lidar_xFRF >= xc_shore) & (lidar_xFRF <= xc_sea)
        ix_inspan = np.where(tmp)[0]
        padding = 2
        itrim = np.arange(ix_inspan[0]-padding, ix_inspan[-1]+padding+1)
        xtmp = lidar_xFRF[itrim]
        ztmp = zsmooth_fullspan[itrim,tt]
        xinterp = np.linspace(xc_sea,xc_shore,nx)
        zinterp = np.interp(xinterp, xtmp, ztmp)
        scaled_profiles[tt, :] = zinterp
        profile_width[tt] = xc_sea - xc_shore
        unscaled_profile[tt, :] = zsmooth_fullspan[itrim[0]:itrim[0]+nkeep,tt]
        unscaled_xi[tt] = lidar_xFRF[itrim[0]]
        slptmp = avgslope_withinXCs[:, tt]
        xtmp = lidar_xFRF[~np.isnan(slptmp)]
        slpinterp = np.interp(xinterp, xtmp, slptmp[~np.isnan(slptmp)])
        scaled_avgslope[tt, :] = slpinterp
        shift_avgslope[tt,np.arange(slptmp[~np.isnan(slptmp)].size)] = slptmp[~np.isnan(slptmp)]
        ztmp = zsmooth_fullspan[:, tt]
        shift_zsmooth[tt,np.arange(ztmp[~np.isnan(slptmp)].size)] = ztmp[~np.isnan(slptmp)]
        slptmp = avgslope_beyondXCsea[:,tt]
        shift_avgslope_beyondXCsea[tt,np.arange(slptmp[~np.isnan(slptmp)].size)] = slptmp[~np.isnan(slptmp)]


# SAVE - smoothed, shifted, convoluted Z and Slope data
# with open('lidar_elev&slope_processed.pickle','wb') as file:
#     pickle.dump([profile_width,zsmooth_fullspan,shift_zsmooth,avgslope_fullspan,avgslope_withinXCs,avgslope_beyondXCsea,
#                  shift_avgslope,shift_avgslope_beyondXCsea,maxgap_fullspan,xc_fullspan,dXcdt_fullspan,time_fullspan,lidar_xFRF,
#                  scaled_profiles,scaled_avgslope,unscaled_profile],file)
# with open('elev_processed_base.pickle', 'wb') as file:
#     pickle.dump([time_fullspan,lidar_xFRF,profile_width,maxgap_fullspan,xc_fullspan,dXcdt_fullspan], file)
# with open('elev_processed_slopes.pickle', 'wb') as file:
#     pickle.dump([avgslope_fullspan, avgslope_withinXCs,avgslope_beyondXCsea], file)
# with open('elev_processed_slopes_shift.pickle', 'wb') as file:
# #     pickle.dump([shift_avgslope,shift_avgslope_beyondXCsea], file)
# with open('elev_processed_elev.pickle', 'wb') as file:
#     pickle.dump([zsmooth_fullspan,shift_zsmooth,unscaled_profile], file)
# with open('elev_processed_elev&slopes_scaled.pickle', 'wb') as file:
#     pickle.dump([scaled_profiles,scaled_avgslope], file)
# with open('elev_processed_unscaled_xi.pickle', 'wb') as file:
#     pickle.dump([unscaled_xi], file)

# LOAD  - smoothed, shifted, convoluted Z and Slope data
# with open('elev_processed_base.pickle', 'rb') as file:
#     time_fullspan,lidar_xFRF,profile_width,maxgap_fullspan,xc_fullspan,dXcdt_fullspan = pickle.load(file)
with open('elev_processed_base.pickle', 'rb') as file:
    time_fullspan,lidar_xFRF,profile_width,maxgap_fullspan,_,_ = pickle.load(file)
with open('elev_processed_slopes.pickle', 'rb') as file:
    avgslope_fullspan, avgslope_withinXCs,avgslope_beyondXCsea = pickle.load(file)
with open('elev_processed_slopes_shift.pickle', 'rb') as file:
    shift_avgslope,shift_avgslope_beyondXCsea = pickle.load(file)
with open('elev_processed_elev.pickle', 'rb') as file:
    zsmooth_fullspan,shift_zsmooth,unscaled_profile = pickle.load(file)
with open('elev_processed_elev&slopes_scaled.pickle', 'rb') as file:
    scaled_profiles,scaled_avgslope = pickle.load(file)

# Plot the range of beach widths
var = profile_width[~np.isnan(profile_width)]
fig, ax = plt.subplots()
ax.hist(var, density=True, cumulative=True)
ax.set_title('profile width, N = ' + str(sum(~np.isnan(var))))
ax.plot([np.nanmean(var), np.nanmean(var)], [0, 1], 'k')
ax.grid(which='both',axis='both')
# PLOT - avgslope for profile between XCshore and XCsea, shifted to same x-origin
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
xplot = np.arange(nx)*dx
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(shift_avgslope,shift_avgslope.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_title('Avg. Slope for profile between XCshore and XCsea - Shifted')
# PLOT - profile elevations between XCshore and XCsea - shifted
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(shift_zsmooth, shift_zsmooth.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m, NAVD88]')
# ax.set_title('Elev. between XCshore and XCsea - Shifted')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')

# PLOT - avgslope for profile between XCshore and XCsea, SCALED
xplot = np.linspace(0,1,nx)
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(scaled_avgslope,scaled_avgslope.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_title('Avg. Slope for profile between XCshore and XCsea - SCALED')
# PLOT - avgslope for profile beyond XCsea, shifted to same x-origin
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
xplot = np.arange(nx)*dx
XX, TT = np.meshgrid(lidar_xFRF, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(shift_avgslope_beyondXCsea.T,shift_avgslope_beyondXCsea.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_title('Avg. Slope for profile beyond XCsea - Shifted')


# check horz/vertical bounds on zsmooth_shift
nx = 2000
dx = 0.1
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
xplot = np.arange(nx)*dx
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(shift_zsmooth,shift_zsmooth.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('avg. slope [m/m]')
# ax.set_title('Avg. Slope for profile beyond XCsea - Shifted')

fig, ax = plt.subplots()
ax.scatter(tplot,profile_width,4)



# RECALL there are still GAPS in data, even if NO BAD DATA was removed
check_data = scaled_profiles
rowsnonans_scaled = np.where(np.sum(np.isnan(check_data),axis=1 ) == 0)[0]




# NEXT:  isolate times where full profile exists......
# profiles_to_process = zgood_scaled_filled
profiles_to_process = scaled_profiles
tkeep = np.sum(~np.isnan(profiles_to_process),axis=1 ) == nx
ikeep = np.where(tkeep)[0]
shorelines = profiles_to_process[tkeep,:]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev_fullspan)
# ax.set_ylim(-0.5,2.5)
ax.set_title('Profiles, all')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,zsmooth_fullspan[:,tkeep])
# ax.set_ylim(-0.5,2.5)
ax.set_title('Profiles, smoothed')
fig, ax = plt.subplots()
ax.plot(xinterp,shorelines.T)
plt.grid(which='both', axis='both')
# ax.set_ylim(-0.5,2.5)
ax.set_title('Profiles, smoothed & scaled')

# # do some tweaking...
# zgood_tweaked = shorelines
# # zgood_tweaked[:,0:40] = np.nan
# zgood_tweaked[:,0] = cont_elev[0]
# for ii in np.arange(ikeep.size):
#     ztmp = zgood_tweaked[ii,:]
#     nsmooth = 10
#     zmean = np.convolve(ztmp, np.ones(nsmooth)/nsmooth, mode='same')
#     zgood_tweaked[ii,:] = zmean
# fig, ax = plt.subplots()
# ax.plot(xplot,zgood_tweaked.T)




# FROM DYLAN
profiles_to_process = scaled_profiles
tkeep = np.where(np.sum(~np.isnan(profiles_to_process),axis=1 ) == profiles_to_process.shape[1])[0]
data = scaled_profiles[tkeep,:]
dataMean = np.mean(data,axis=0) # this will give you an average for each cross-shore transect
dataStd = np.std(data,axis=0)
dataNorm = (data[:,:] - dataMean) / dataStd
tmp1 = np.max(dataNorm,axis=1)
tmp2 = np.min(dataNorm,axis=1)
ikeep = np.where((tmp1<3.5)&(tmp2>-3.5))[0]

fig, ax = plt.subplots()
# ax.plot(dataNorm[ikeep,:].T)
xplot = np.linspace(0,1,nx)
ax.plot(xplot,scaled_profiles[:,:].T)
# ax.set_ylim(-5,5)
# ax.set_title('Normalized equi-length profiles')
ax.set_title('Scaled profiles')
ax.set_xlabel('x/b_w [-]')
ax.set_ylabel('z [m, NAVD88]')
ax.plot([0, 1],[mhw, mhw],'k--')
ax.plot([0, 1],[mwl, mwl],'k--')
ax.set_xlim(0,1)



fig, ax = plt.subplots()
ax.plot(dataNorm[ikeep,:].T)
# ax.plot(xinterp,dataNorm[:,:].T)
ax.set_ylim(-5,5)
ax.set_title('Normalized equi-length profiles')

# principal components analysis
ipca = PCA(n_components=min(dataNorm[ikeep,:].shape[0], dataNorm[ikeep,:].shape[1]))
PCs = ipca.fit_transform(dataNorm[ikeep,:])  # these are the temporal magnitudes of the spatial modes where PCs[:,0] are the varying amplitude of mode 1 with respect to time
EOFs = ipca.components_  # these are the spatial modes where EOFs[0,:] is mode 1, EOFs[1,:] is mode 2, and so on...
variance = ipca.explained_variance_ # this is the variance explained by each mode
nPercent = variance / np.sum(variance)  # this is the percent explained (the first mode will explain the greatest percentage of your data)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0   # this is the cumulative variance
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

fig, ax = plt.subplots(2,4)
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
tmp = tplot[notnanrow]
xplot = xinterp
ax[0,0].scatter(tmp[ikeep],PCs[:,0],4)
ax[0,0].set_title('Mode 1')
ax[1,0].plot(xplot,EOFs[0,:])
ax[0,1].scatter(tmp[ikeep],PCs[:,1],4)
ax[0,1].set_title('Mode 2')
ax[1,1].plot(xplot,EOFs[1,:])
ax[0,2].scatter(tmp[ikeep],PCs[:,2],4)
ax[0,2].set_title('Mode 3')
ax[1,2].plot(xplot,EOFs[2,:])
ax[0,3].scatter(tmp[ikeep],PCs[:,3],4)
ax[0,3].set_title('Mode 4')
ax[1,3].plot(xplot,EOFs[3,:])

fig, ax = plt.subplots()
mode1_alltimes = PCs[:,0]
mode2_alltimes = PCs[:,1]
mode3_alltimes = PCs[:,2]
ax.scatter(profile_width[notnanrow[ikeep]],mode1_alltimes,4,alpha=0.1)
plt.grid(which='major', axis='both')
fig, ax = plt.subplots(3,1)
ax[0].scatter(tmp[ikeep],mode1_alltimes,4)
ax[1].scatter(tmp[ikeep],mode2_alltimes,4)
ax[2].scatter(tmp[ikeep],mode3_alltimes,4)


# Calcs for Kate
print(tplot[-1]-tplot[0])
days_fullspan = 3178+16/24
days_avail = 18758/24



# Rescale profiles used for PCA:
scaled_PCAprofiles = data
fig, ax = plt.subplots()
xplot = np.linspace(0,1,nx)
a = profile_width[notnanrow[:]].T
b = np.tile(xplot,(a.size,1))
xplot_scaled = a.T*b.T
zplot = np.flip(scaled_PCAprofiles.T)
ax.plot(xplot_scaled[:,ikeep],zplot[:,ikeep])
rescaled_PCAprofiles = zplot[:,ikeep]
rescaled_xplot = xplot_scaled[:,ikeep]
rescaled_width = a[ikeep]
fig, ax = plt.subplots()
ax.hist(rescaled_width)

fig, ax = plt.subplots()
xplot = np.linspace(0,1,nx)
zplot = np.flip(scaled_PCAprofiles.T)
ax.plot(xplot,zplot)

fig, ax = plt.subplots()
xplot = PCs[:,0]
yplot = PCs[:,1]
# cplot = np.arange(xplot.size)
cplot = profile_width[tkeep[ikeep]]
# cmap = plt.cm.rainbow(np.linspace(0, 1, xplot.size))
# ax.set_prop_cycle('color', cmap)
ph = ax.scatter(xplot,yplot,4,cplot,cmap='plasma')
ax.set_xlabel('Mode 1')
ax.set_ylabel('Mode 2')
cbar = plt.colorbar(ph)
ph.set_clim(30,75)
ax.set_ylim(-100,100)
# ax.set_xlim(-100,100)
ax.set_xlim(-110,110)
plt.grid(which='both', axis='both')
cbar.set_label('beach width [m]')


fig, ax = plt.subplots(2,1)
xplot = np.linspace(0,1,nx)
tmp = (PCs[:,1] > 30) & (PCs[:,2] < -15)
idemo = np.where(tmp)[0]
idemo4 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')


fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] < -20) & (PCs[:,2] >= 10) & (PCs[:,2] < 15)
idemo = np.where(tmp)[0]
idemo2 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')
fig, ax = plt.subplots()
ax.scatter(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo],1,c='tab:orange',alpha=0.1)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.plot([0,75],[mhw,mhw],'k--')
ax.plot([0,75],[mwl,mwl],'k--')
ax.set_xlim(0,75)


fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] < -16) & (PCs[:,2] > 15)
idemo = np.where(tmp)[0]
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')
fig, ax = plt.subplots()
ax.scatter(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo],1,c='tab:blue',alpha=0.1)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.plot([0,75],[mhw,mhw],'k--')
ax.plot([0,75],[mwl,mwl],'k--')
ax.set_xlim(0,75)



fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] > 25) & (PCs[:,2] > 25)
idemo = np.where(tmp)[0]
idemo1 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
plt.grid(which='both', axis='both')



fig, ax = plt.subplots(2,1)
tmp = (PCs[:,1] < -25) & (PCs[:,2] < -25)
idemo = np.where(tmp)[0]
idemo3 = idemo
cmap = plt.cm.jet(np.linspace(0, 1, idemo.size))
ax[0].set_prop_cycle('color', cmap)
ax[1].set_prop_cycle('color', cmap)
ax[0].plot(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo])
# ax[0].plot(xplot,rescaled_PCAprofiles[:,idemo])
for ii in np.arange(idemo.size):
    ax[1].scatter(PCs[idemo[ii],1],PCs[idemo[ii],2],10)
fig, ax = plt.subplots()
ax.scatter(rescaled_xplot[:,idemo],rescaled_PCAprofiles[:,idemo],1,c='tab:green',alpha=0.1)
ax.set_xlabel('x* [m]')
ax.set_ylabel('z [m]')
ax.plot([0,75],[mhw,mhw],'k--')
ax.plot([0,75],[mwl,mwl],'k--')
ax.set_xlim(0,75)



fig, ax = plt.subplots(2,2)
cplot1 = np.tile(PCs[idemo1,0],(rescaled_xplot.shape[0],))
ph1 = ax[0,1].scatter(rescaled_xplot[:,idemo1],rescaled_PCAprofiles[:,idemo1],4,cplot1,cmap='coolwarm')
# ph1 = ax[0,1].scatter(np.tile(xplot,(idemo1.size,1)).T,rescaled_PCAprofiles[:,idemo1],4,cplot1,cmap='coolwarm')
ph1.set_clim(-40, 40)

cplot2 = np.tile(PCs[idemo2,0],(rescaled_xplot.shape[0],))
ph2 = ax[0,0].scatter(rescaled_xplot[:,idemo2],rescaled_PCAprofiles[:,idemo2],4,cplot2,cmap='coolwarm')
# ph2 = ax[0,0].scatter(np.tile(xplot,(idemo2.size,1)).T,rescaled_PCAprofiles[:,idemo2],4,cplot2,cmap='coolwarm')
ph2.set_clim(-40, 40)

cplot3 = np.tile(PCs[idemo3,0],(rescaled_xplot.shape[0],))
ph3 = ax[1,0].scatter(rescaled_xplot[:,idemo3],rescaled_PCAprofiles[:,idemo3],4,cplot3,cmap='coolwarm')
# ph3 = ax[1,0].scatter(np.tile(xplot,(idemo3.size,1)).T,rescaled_PCAprofiles[:,idemo3],4,cplot3,cmap='coolwarm')
ph3.set_clim(-40, 40)

cplot4 = np.tile(PCs[idemo4,0],(rescaled_xplot.shape[0],))
ph4 = ax[1,1].scatter(rescaled_xplot[:,idemo4],rescaled_PCAprofiles[:,idemo4],4,cplot4,cmap='coolwarm')
# ph4 = ax[1,1].scatter(np.tile(xplot,(idemo4.size,1)).T,rescaled_PCAprofiles[:,idemo4],4,cplot4,cmap='coolwarm')
# ch = plt.colorbar(ph4)
ph4.set_clim(-40, 40)

# Calculate correlation between CHANGE in PCs and environmental variables
with open('elev_processed_elev&slopes_scaled.pickle', 'rb') as file:
    scaled_profiles,_ = pickle.load(file)
with open('elev_processed_base.pickle', 'rb') as file:
    _,lidar_xFRF,profile_width,_,_,_ = pickle.load(file)
with open('elev_processed_elev.pickle', 'rb') as file:
    zsmooth_fullspan, _, _ = pickle.load(file)
with open('IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,_,_,_,data_lidarwg110,_,_,_,_ = pickle.load(file)
with open('beach_stats.pickle', 'rb') as file:
    var_beachwid,var_beachvol,var_beachslp = pickle.load(file)
var_wave8m_Hs = data_wave8m[:,0]
var_wave8m_Tp = data_wave8m[:,1]
var_wave8m_dir = data_wave8m[:,2]
var_wave17m_Hs = data_wave17m[:,0]
var_wave17m_Tp = data_wave17m[:,1]
var_wave17m_dir = data_wave17m[:,2]
var_lidar_elev2p = data_lidar_elev2p[:].flatten()
var_tidegauge = data_tidegauge[:].flatten()
var_wg110_Hs = data_lidarwg110[:,0]
var_wg110_HsIN = data_lidarwg110[:,1]
var_wg110_HsIG = data_lidarwg110[:,2]
var_wg110_Tp = data_lidarwg110[:,3]
var_wg110_TmIG = data_lidarwg110[:,4]
var_beachwid = profile_width
var_beachvol = np.empty(shape=var_beachwid.shape)
var_beachslp = np.empty(shape=var_beachwid.shape)
var_beachvol[:] = np.nan
var_beachslp[:] = np.nan
for ii in np.arange(time_fullspan.size):
    if sum(~np.isnan(zsmooth_fullspan[:,ii])) > 0:
        tmp = np.abs(zsmooth_fullspan[:,ii] - var_tidegauge[ii])
        if sum(~np.isnan(tmp)) > 0:
            ix_shoreline = np.where(tmp == np.nanmin(tmp))[0][0]
            iix = np.arange(-50,50)+ix_shoreline
            ztmp = zsmooth_fullspan[iix,ii]
            ztmp = ztmp[~np.isnan(ztmp)]
            slptmp = (ztmp[-1]-ztmp[0])/(dx*ztmp.size)
            var_beachslp[ii] = slptmp
            delx = var_beachwid[ii]/nx
            var_beachvol[ii] = np.sum(scaled_profiles[ii,:])*delx

# with open('beach_stats.pickle', 'wb') as file:
#     pickle.dump([var_beachwid,var_beachvol,var_beachslp], file)
varnames = ['beachwid','beachvol','beachslp','tidegauge','wave8m_Hs','wave8m_Tp','wave8m_dir','wave17m_Hs',
            'wave17m_Tp','wave17m_dir','wg110_HsIN','wg110_HsIG','wg110_Tp','wg110_TmIG','lidar_elev2p']
PC_fullspan = np.empty((4,time_fullspan.size))
PC_fullspan[:] = np.nan
PC_fullspan[0,[tkeep[ikeep]]] = PCs[:,0]
PC_fullspan[1,[tkeep[ikeep]]] = PCs[:,1]
PC_fullspan[2,[tkeep[ikeep]]] = PCs[:,2]
PC_fullspan[3,[tkeep[ikeep]]] = PCs[:,3]
fig, ax = plt.subplots()
ax.plot(time_fullspan,PC_fullspan[0,:],'o')
ax.plot(time_fullspan[tkeep[ikeep]],PCs[:,0],'o')


delPC_fullspan = np.empty((4,time_fullspan.size))
delPC_fullspan[:] = np.nan
delPC_fullspan[:,1:] = PC_fullspan[:,1:] - PC_fullspan[:,0:-1]
pcorrvals = np.empty((4,len(varnames)))
for ii in np.arange(len(varnames)):
    for jj in np.arange(4):
        exec('varx = var_' + varnames[ii])
        vary = delPC_fullspan[jj,:]
        ijok = ~np.isnan(vary) & ~np.isnan(varx)
        # pearsons = np.cov(varx[ijok],vary[ijok])/(np.std(varx[ijok])*np.std(vary[ijok]))
        # numerator = np.sum((varx[ijok] - np.mean(varx[ijok])) * (vary[ijok] - np.mean(vary[ijok])))
        # denominator = np.sqrt(np.sum(varx[ijok] - np.mean(varx[ijok])) ** 2) * np.sqrt(np.sum(vary[ijok] - np.mean(vary[ijok])) ** 2)
        pearsons = np.corrcoef(varx[ijok],vary[ijok])
        pcorrvals[jj,ii] = pearsons[0,1]
# with open('pearsons_values.pickle', 'wb') as file:
#     pickle.dump([pcorrvals], file)
fig, ax = plt.subplots()
iplot = (np.arange(len(varnames)) < 6) + (np.arange(len(varnames)) > 9)
sns.heatmap(pcorrvals[:,iplot].T, annot=True, linewidth=.5, fmt=".2f", cmap='RdBu', vmin=-0.5, vmax=0.5)
ax.set_xlabel('D/Dt of Mode [1/hr]')

fig, ax = plt.subplots()
ax.hist(var_beachslp,bins=50)