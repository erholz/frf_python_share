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


def lidar_fillgaps(elev_input,lidartime,lidar_xFRF,halfspan_time,halfspan_x):
    timestartcode = time.time()

    lidar_gappy = np.copy(elev_input)
    lidar_filled = np.copy(elev_input)

    for tt in np.arange(len(lidartime)-1):
        if tt < halfspan_time:
            grabtt = np.arange(0,tt+1)
            numgrab = len(grabtt)
            grabtt = np.append(grabtt, np.arange(tt+1,tt+halfspan_time*2-numgrab+2))
        elif tt >= len(lidartime)-halfspan_time:
            grabtt = np.arange(tt,len(lidartime))
            numgrab = len(grabtt)
            grabtt = np.append(np.arange(tt-(halfspan_time*2-numgrab)-1,tt),grabtt)
        else:
            grabtt = np.arange(tt-halfspan_time,tt+halfspan_time+1)
        for ii in np.arange(len(lidar_xFRF)-1):
            if ii < halfspan_x:
                grabii = np.arange(0, ii + 1)
                numgrab = len(grabii)
                grabii = np.append(grabii, np.arange(ii + 1, ii + halfspan_x * 2 - numgrab + 2))
            elif ii >= len(lidar_xFRF) - halfspan_x:
                grabii = np.arange(ii, len(lidar_xFRF))
                numgrab = len(grabii)
                grabii = np.append(np.arange(ii - (halfspan_x * 2 - numgrab) - 1, ii), grabii)
            else:
                grabii = np.arange(ii - halfspan_x, ii + halfspan_x + 1)
            gappy_slice = lidar_gappy[grabtt,grabii[:,np.newaxis]]
            filled_slice = gappy_slice
            gappy_numel = gappy_slice.size
            gappy_numnotnan = sum(sum(~np.isnan(gappy_slice)))
            if (gappy_numnotnan/gappy_numel > 0.3) & (gappy_numnotnan/gappy_numel < 1):
                xv = lidar_xFRF[grabii]
                tv = lidartime[grabtt]
                [TT, XX] = np.meshgrid(tv, xv)
                # f = sp.interpolate.interp2d(tv, xv, gappy_slice)
                ij = ~np.isnan(gappy_slice)
                f = sp.interpolate.LinearNDInterpolator((TT[ij],XX[ij]),gappy_slice[ij])
                tnew = TT[np.isnan(gappy_slice)]
                xnew = XX[np.isnan(gappy_slice)]
                fnew = f((tnew,xnew))
                filled_slice[~ij] = fnew
            lidar_filled[grabtt, grabii[:, np.newaxis]] = filled_slice

    codeduration = time.time() - timestartcode
    print('Done!  lidar_fillgaps Duration = ' + str(codeduration) + ' seconds')

    return lidar_filled

def prof_extendfromlidarhydro(lidarelev,lidartime,lidar_xFRF,wlmin_lidar,cont_ts):
    # map WL depths onto lowest contour position, if possible
    prof_extended = np.empty(shape=lidarelev.shape)
    wlmean_zmin = np.empty(shape=lidartime.shape)
    hmean_zmin = np.empty(shape=lidartime.shape)
    wlmean_zmin[:] = np.nan
    hmean_zmin[:] = np.nan
    prof_extended[:] = np.nan
    for tt in np.arange(len(lidartime)):
        prof_extended[~np.isnan(lidarelev)] = lidarelev[~np.isnan(lidarelev)]
        xq = np.nanmax(cont_ts[:, tt])  # max (furthest seaward) contour position
        ztmp = wlmin_lidar[tt, :]  # min water elev observed across profile at time tt
        xtmp = lidar_xFRF  # all x-coords of profile
        if ~np.isnan(xq) & (len(ztmp[~np.isnan(ztmp)]) > 0):
            if xq < np.nanmin(xtmp[~np.isnan(ztmp)]):
                hmean_zmin[tt] = 0
                N = 5
                tmpWL = wlmin_lidar[tt, :]
                tmpij = ~np.isnan(tmpWL)
                new_elev = np.convolve(tmpWL[tmpij], np.ones(N) / N, 'valid')
                prof_extended[tt, np.argwhere(tmpij)[int(np.floor(N / 2)):-(int(np.floor(N / 2)))].T] = new_elev[:]

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

# CREATE FUNCTION HERE THAT EXTENDS PROFILES TO EQUAL LENGTH
#
# def prof_extendfromslopes(lidarelev,lidarslope,scaled_profiles,scaled_avgslope,shift_avgslope_beyondXCsea,
#                           time_fullspan,data_wave8m,data_tidegauge,data_lidar_elev2p):

# picklefile_dir = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/processed_26Nov2024/'
# with open(picklefile_dir+'elev_processed_base.pickle', 'rb') as file:
#     time_fullspan,lidar_xFRF,profile_width,maxgap_fullspan,xc_fullspan,dXcdt_fullspan = pickle.load(file)
# with open(picklefile_dir+'elev_processed_slopes.pickle', 'rb') as file:
#     avgslope_fullspan, avgslope_withinXCs,avgslope_beyondXCsea = pickle.load(file)
# with open(picklefile_dir+'elev_processed_slopes_shift.pickle', 'rb') as file:
#     shift_avgslope,shift_avgslope_beyondXCsea = pickle.load(file)
# with open(picklefile_dir+'elev_processed_elev.pickle', 'rb') as file:
#     zsmooth_fullspan,shift_zsmooth,unscaled_profile = pickle.load(file)
# with open(picklefile_dir+'elev_processed_elev&slopes_scaled.pickle', 'rb') as file:
#     scaled_profiles,scaled_avgslope = pickle.load(file)
# with open(picklefile_dir+'elev_processed_unscaled_xi.pickle', 'rb') as file:
#     unscaled_xi = pickle.load(file)

picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,_,_,lidarelev_fullspan = pickle.load(file)
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]
with open(picklefile_dir + 'final_profile_13Nov2024_shift.pickle', 'rb') as file:
    profile_fullspan_shift = pickle.load(file)
with open(picklefile_dir + 'final_profile_13Nov2024.pickle', 'rb') as file:
    profile_fullspan = pickle.load(file)

## Use contours to determine the profile 'width'
mwl = -0.13
zero = 0
mhw = 3.6
dune_toe = 3.22
cont_elev = np.array([mwl, zero, dune_toe, mhw]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(profile_fullspan.T,time_fullspan,lidar_xFRF,cont_elev)
fig, ax = plt.subplots()
ax.plot(cont_ts.T)
# profile_width = []


# figure out size of datasets
lidarelev = np.empty(shape=profile_fullspan_shift.T.shape)
lidarelev[:] = profile_fullspan_shift.T[:]
nx = lidarelev.shape[1]
dx = 0.1
lidar_xFRF_shift = dx*np.arange(0,(lidar_xFRF.size),1)


# Plot the shifted profiles
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev.T)
ax.set_title('Shifted Profiles - all, best available')
ax.set_xlabel('x* [m]')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,profile_fullspan)
ax.set_title('Profiles - all, best available')


## FIRST, load the processed data from Dylan
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
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

# NOW combine
bathylidar_combo = np.empty(shape=bathy_nolidar.shape)
bathylidar_combo[:] = profile_fullspan[:]
bathylidar_combo[~np.isnan(bathy_nolidar_clean)] = bathy_nolidar_clean[~np.isnan(bathy_nolidar_clean)]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,bathylidar_combo)
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('z [m]')
ax.set_title('Bathy-Lidar combined')

fig, ax = plt.subplots()
yplot_raw = 100*np.nansum(~np.isnan(lidarelev_fullspan),axis=1)/time_fullspan.size
yplot_lidar = 100*np.nansum(~np.isnan(profile_fullspan),axis=1)/time_fullspan.size
yplot_bathy = 100*np.nansum(~np.isnan(bathy_nolidar),axis=1)/time_fullspan.size
yplot_combo = 100*np.nansum(~np.isnan(bathylidar_combo),axis=1)/time_fullspan.size
ax.plot(lidar_xFRF,yplot_raw,label='raw lidar')
ax.plot(lidar_xFRF,yplot_lidar,label='processed lidar')
ax.plot(lidar_xFRF,yplot_bathy,label='bathy where lidar unavail')
ax.plot(lidar_xFRF,yplot_combo,label='processed lidar + bathy')
ax.set_xlabel('xFRF [m]')
ax.set_ylabel('% available')
ax.legend()
#
# ## SAVE COMBO PROFILES(!!!!!!)
# with open(picklefile_dir+'bathylidar_combo.pickle','wb') as file:
#     pickle.dump([lidar_xFRF,bathylidar_combo], file)

with open(picklefile_dir+'bathylidar_combo.pickle','rb') as file:
    lidar_xFRF, bathylidar_combo = pickle.load(file)


## Find the max gap in each profile
maxgap_bathylidar = np.empty(shape=time_fullspan.shape)
maxgap_bathylidar[:] = np.nan
maxgap_below_MHW = np.empty(shape=time_fullspan.shape)
maxgap_below_MHW[:] = np.nan
maxgap_loc = np.empty(shape=time_fullspan.shape)
maxgap_loc[:] = np.nan
for tt in np.arange(time_fullspan.size):
    ztmp = bathylidar_combo[:,tt]
    if np.sum(~np.isnan(ztmp)) > 0:
        ix_notnan = np.where(~np.isnan(ztmp))[0]
        approxlength = ix_notnan[-1] - ix_notnan[0]
        zinput = ztmp[np.arange(ix_notnan[0], ix_notnan[-1])]
        gapstart, gapend, gapsize, maxgap = find_nangaps(zinput)
        maxgap_bathylidar[tt] = maxgap
        if len(gapsize) > 1:
            if zinput[int(gapstart[gapsize==maxgap]-1)] <= mhw:
                maxgap_below_MHW[tt] = 1
                maxgap_loc[tt] = zinput[int(gapstart[gapsize==maxgap]-1)]
        elif (len(gapsize) == 1) & ~np.isnan(gapstart):
            if zinput[int(gapstart[0])-1] <= mhw:
                maxgap_below_MHW[tt] = 1
                maxgap_loc[tt] = zinput[int(gapstart[0])-1]

fig, ax = plt.subplots()
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
maxgap_bathylidar[maxgap_bathylidar == 0] = np.nan
ax.plot(tplot[maxgap_below_MHW==1],maxgap_bathylidar[maxgap_below_MHW==1],'x')
ax.plot(tplot[maxgap_below_MHW!=1],maxgap_bathylidar[maxgap_below_MHW!=1],'+')
ii_maxgap_aboveMHW = np.where(maxgap_below_MHW!=1)[0]
ii_maxgap_belowMHW = np.where(maxgap_below_MHW==1)[0]
fig, ax = plt.subplots()
# ax.plot(lidar_xFRF,bathylidar_combo[:,ii_maxgap_belowMHW[20:30]],'o')

ii_maxgap_tofiill = np.where(maxgap_bathylidar <= 100)[0]
bathylidar_fillmaxgaps = np.empty(shape=bathylidar_combo.shape)
bathylidar_fillmaxgaps[:] = bathylidar_combo[:]
bathylidar_fill = np.empty(shape=bathylidar_combo.shape)
bathylidar_fill[:] = np.nan
for jj in ii_maxgap_tofiill:
    if ~np.isnan(maxgap_bathylidar[jj]) & (maxgap_below_MHW[jj] == 1):
        xprof_tt = lidar_xFRF[0:]
        zprof_tt = bathylidar_combo[:, jj]
        ix_notnan = np.where(~np.isnan(zprof_tt) & (zprof_tt <= mhw))[0]
        zinput = zprof_tt[np.arange(ix_notnan[0], ix_notnan[-1])]
        xinput = xprof_tt[np.arange(ix_notnan[0], ix_notnan[-1])]
        # gapstart, gapend, gapsize, maxgap = find_nangaps(zinput)
        ii_tofill = np.where(np.isnan(zinput))
        # ii_tofill = ii_tofill[zinput < mhw]
        xin = xinput[~np.isnan(zinput)]
        yin = zinput[~np.isnan(zinput)]
        cs = CubicSpline(xin, yin, bc_type='natural')
        Lspline = xin[-1] - xin[0]
        xspline = np.linspace(xin[0], xin[-1], int(np.ceil(Lspline / 0.01)))
        zspline_tmp = cs(xspline)
        zspline_tmp[zspline_tmp >= mhw] = np.nan
        zspline_tofill = np.interp(lidar_xFRF[ii_tofill], xspline, zspline_tmp)
        zspline_tofill[zspline_tofill >= 3.45] = np.nan
        bathylidar_fillmaxgaps[ii_tofill,jj] = zspline_tofill
        bathylidar_fill[ii_tofill,jj] = zspline_tofill



fig, ax = plt.subplots()
# ax.plot(lidar_xFRF,bathylidar_fill)
ax.plot(lidar_xFRF,bathylidar_fillmaxgaps)
fig, ax = plt.subplots()
tmp = bathylidar_combo[:,ii_maxgap_tofiill]
# tmp[tmp > mhw] = np.nan
ax.plot(lidar_xFRF,tmp,'.')

fig, ax = plt.subplots()
tmp = bathylidar_fillmaxgaps[:,ii_maxgap_tofiill]
# tmp[tmp > mhw] = np.nan
ax.plot(lidar_xFRF,tmp,'.')


# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# with open(picklefile_dir+'bathylidar_fill.pickle','wb') as file:
#     pickle.dump([lidar_xFRF,bathylidar_fillmaxgaps], file)



## THEN, try to fit equilibrium profile
def equilibriumprofile_func_2param(x, a, b):
    return a * x * np.exp(b)
def equilibriumprofile_func_1param(x, a):
    return a * x ** (2/3)

# Find the profiles where there is data below the water line...
depth_fullspan = lidarelev - data_tidegauge
num_below_watlev = np.nansum(depth_fullspan <= 0.5,axis=1)
num_withdata = np.nansum(~np.isnan(lidarelev),axis=1)
id_below_watlev = np.where(num_below_watlev > 0)[0]
id_withdata = np.where(num_withdata > 0)[0]

nonequil_flag = np.empty(shape=time_fullspan.shape)
nonequil_flag[:] = np.nan
Acoef = np.empty(shape=time_fullspan.shape)
bcoef = np.empty(shape=time_fullspan.shape)
fitrmse = np.empty(shape=time_fullspan.shape)
Acoef[:] = np.nan
bcoef[:] = np.nan
fitrmse[:] = np.nan
profile_extend = np.empty(shape=lidarelev.shape)
profile_extend[:] = lidarelev[:]
profile_extend_testdata = np.empty(shape=lidarelev.shape)
profile_extend_testdata[:] = np.nan
prof_x_wl = np.empty(shape=time_fullspan.shape)
prof_x_wl[:] = np.nan
# for tt in np.arange(10):
# for tt in ([ 53613, 59241, 65637, 65861, 67527, 67528, 67576, 67577, 67638]):
for tt in np.arange(time_fullspan.size):
    Lgrab = 5          # try fitting equilibrium profile to last [Lgrab] meters of available data
    watlev_tt = data_tidegauge[tt]
    wlbuffer = 0.25
    if (sum(~np.isnan(lidarelev[tt,:])) > 10) & (~np.isnan(watlev_tt)) :
        ii_submerged = np.where(lidarelev[tt, :] <= watlev_tt + wlbuffer)[0]
        iiclose = np.where(abs(lidarelev[tt,:]-(wlbuffer+watlev_tt)) == np.nanmin(abs(lidarelev[tt,:]-(wlbuffer+watlev_tt))))[0]
        if iiclose.size > 1:
            iiclose = iiclose[0]
        prof_x_wl[tt] = np.interp((wlbuffer+watlev_tt),lidarelev[tt,np.arange(iiclose-1,iiclose+1)],lidar_xFRF_shift[np.arange(iiclose-1,iiclose+1)])
        if len(ii_submerged) > 5:
            id_last = sum(~np.isnan(lidarelev[tt,:]))
            if ii_submerged.size > Lgrab/dx:
                numgrab = Lgrab/dx
            else:
                numgrab = ii_submerged.size
            # numgrab = ii_submerged.size
            # iitest = np.arange(ii_submerged[0],ii_submerged[0]+numgrab).astype(int)
            # iitest = np.arange(ii_submerged[-1]-numgrab,ii_submerged[-1]).astype(int)
            iitest = ii_submerged

            htmp = (wlbuffer + watlev_tt) - lidarelev[tt,iitest]
            xtmp = dx*np.arange(htmp.size)
            iitest = iitest[~np.isnan(htmp)]
            xtmp = xtmp[~np.isnan(htmp)]
            htmp = htmp[~np.isnan(htmp)]
            # zobs_final = (wlbuffer + watlev_tt) - htmp[-1]
            profile_extend_testdata[tt,iitest] = lidarelev[tt,iitest]
            # xtmp[ztmp < 0] = []     # remove negative values
            # ztmp[ztmp < 0] = []     # remove negative values
            htmp = htmp - htmp[0]   # make initial value 0
            # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, htmp, bounds=([0, -np.inf], [15, np.inf]))
            # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, ztmp)
            popt, pcov = curve_fit(equilibriumprofile_func_1param, xtmp, htmp, bounds=([0.05], [1]))
            Acoef[tt] = popt[0]
            # bcoef[tt] = popt[1]
            hfit = equilibriumprofile_func_1param(xtmp, *popt)
            fitrmse[tt] = np.sqrt(np.mean((hfit-htmp)**2))
            # x_extend = np.arange(lidar_xFRF_shift[iitest[-1]]+dx,lidar_xFRF_shift[-1],dx)
            x_extend = np.arange(lidar_xFRF_shift[iitest[0]] + dx, lidar_xFRF_shift[-1], dx)
            x_extend_norm = x_extend - x_extend[0]
            z_extend = (wlbuffer + watlev_tt) - equilibriumprofile_func_1param(x_extend_norm, *popt)
            profile_extend[tt, np.arange(iitest[0] + 1, nx - 1)] = z_extend
            # # # # # # #
            # PLOT
            fig, ax = plt.subplots()
            ax.plot(lidar_xFRF_shift[iitest],watlev_tt - htmp,'o')
            # ax.plot(lidar_xFRF_shift[iitest],watlev_tt - equilibriumprofile_func_2param(xtmp, *popt), label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
            ax.plot(lidar_xFRF_shift[iitest], watlev_tt - equilibriumprofile_func_1param(xtmp, *popt),
                    label='fit: a=%5.3f' % tuple(popt))
            # ax.plot(x_extend,watlev_tt - z_extend)
            ax.plot(lidar_xFRF_shift,profile_extend[tt,:],':k',label='extend profile')
            ax.plot(lidar_xFRF_shift,watlev_tt*np.ones(shape=lidar_xFRF_shift.shape),'b',label='waterline')
            ax.legend()

# Plot Acoef and bcoef with error (RMSE)
profile_width = np.ones(shape=time_fullspan.shape)
fig, ax = plt.subplots()
ax.plot(Acoef,fitrmse,'o')
ax.set_xlabel('Acoef [-]')
ax.set_ylabel('fit RMSE [m]')
# ph = ax.scatter(Acoef,fitrmse,s=5, c=profile_width, cmap='rainbow')
# cbar = fig.colorbar(ph, ax=ax)
# cbar.set_label('profile width [m]')
# ax.set_xlabel('A coeff')
# ax.set_ylabel('RMSE [m]')
# fig, ax = plt.subplots()
# # ax.plot(bcoef,fitrmse,'o')
# ph = ax.scatter(bcoef,fitrmse,s=5, c=profile_width, cmap='rainbow')
# cbar = fig.colorbar(ph, ax=ax)
# ax.set_xlabel('b coeff')
# cbar.set_label('profile width [m]')
# ax.set_ylabel('RMSE [m]')
# fig, ax = plt.subplots()
# yplot = fitrmse[fitrmse <= 0.05]
# plt.hist(yplot,bins=np.linspace(0,0.05,50))
# ax.set_xlabel('RMSE [m] < 0.1 m')
# fig, ax = plt.subplots()
# yplot = fitrmse[fitrmse > 0.05]
# plt.hist(fitrmse,bins=np.linspace(0.05,np.nanmax(fitrmse),50))
# ax.set_xlabel('RMSE [m] > 0.1 m')

fig, ax = plt.subplots()
xtmp = np.arange(10)
ax.plot(xtmp,1 - 0.05*xtmp*np.exp(2/3))
ax.plot(xtmp,1 - 0.1 * xtmp * np.exp(2 / 3))
ax.plot(xtmp,1 - 0.15 * xtmp * np.exp(2 / 3))
ax.plot(xtmp,1 - 0.2 * xtmp * np.exp(2 / 3))
ax.plot(xtmp,1 - 0.25 * xtmp * np.exp(2 / 3))
ax.title('increase A from 0.5 to 2.5')
fig, ax = plt.subplots()
xtmp = np.arange(10)
ax.plot(xtmp, 1 - 0.15 * xtmp * np.exp(.25))
ax.plot(xtmp, 1 - 0.15 * xtmp * np.exp(.35))
ax.plot(xtmp, 1 - 0.15 * xtmp * np.exp(.45))
ax.plot(xtmp, 1 - 0.15 * xtmp * np.exp(.55))
ax.plot(xtmp, 1 - 0.15 * xtmp * np.exp(.65))
ax.plot(xtmp, 1 - 0.15 * xtmp * np.exp(.75))
ax.title('increase b from 0.25 to .75')


# Plot all the profiles were we did some extending
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend.T)
ax.set_title('Profiles with EXTENSIONS')
yplot1 = 100*np.sum(~np.isnan(profile_extend),axis=0)/time_fullspan.size
yplot2 = 100*np.sum(~np.isnan(lidarelev),axis=0)/time_fullspan.size
yplot3 = 100*np.sum(~np.isnan(profile_fullspan),axis=1)/time_fullspan.size
yplot4 = 100*np.sum(~np.isnan(lidarelev_fullspan),axis=1)/time_fullspan.size

fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot4,label='raw data')
ax.plot(lidar_xFRF,yplot3,label='best-avail/cleaned')
ax.set_xlabel('x [m]')
ax.set_ylabel('Percent available profiles (%)')
ax.legend()
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,yplot2,label='best-avail/cleaned -shifted')
ax.plot(lidar_xFRF_shift,yplot1,label='extended profiles')
ax.legend()
ax.set_xlabel('x* [m]')
ax.set_ylabel('Percent available profiles (%)')



# Plot where Acoef < 0.01
iiplot = np.where(Acoef < 0.01)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where Acoef < 0.01')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where Acoef < 0.01 - EXTENDED')

# Plot where Acoef < 0.02
iiplot = np.where((Acoef >= 0.01) & (Acoef < 0.02))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
# ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where 0.01 < Acoef < 0.02')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
# ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where 0.01 < Acoef < 0.02 - EXTENDED')

# Plot where Acoef < 0.03
iiplot = np.where((Acoef >= 0.02) & (Acoef < 0.03))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where 0.02 < Acoef < 0.03')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where 0.02 < Acoef < 0.03 - EXTENDED')

# Plot where Acoef < 0.05
iiplot = np.where((Acoef >= 0.03) & (Acoef < 0.05))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
# ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.set_title('Profiles where 0.03 < Acoef < 0.05')
ax.set_ylim([-6.5,4])
ax.set_xlim([0, 150])
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
# ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.set_title('Profiles where 0.03 < Acoef < 0.05 - EXTENDED')
ax.set_ylim([-6.5,4])
ax.set_xlim([0, 150])

# Plot where Acoef > 0.1
iiplot = np.where((Acoef >= 0.05))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
# ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.set_title('Profiles where Acoef >= 0.05')
ax.set_ylim([-6.5,4])
ax.set_xlim([0, 150])
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
# ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.set_title('Profiles where Acoef >= 0.05 - EXTENDED')
ax.set_ylim([-6.5,4])
ax.set_xlim([0, 150])

# Can we shift the profiles back??
profile_extend_shiftback = np.empty(shape=profile_extend.shape)
profile_extend_shiftback[:] = np.nan
for tt in np.arange(len(time_fullspan)):
    xc_shore = cont_ts[2,tt]
    if (~np.isnan(xc_shore)):
        # first, map to *_shift vectors
        ix_inspan = np.where((lidar_xFRF >= xc_shore))[0]
        padding = 2
        itrim = np.arange(ix_inspan[0] - padding, lidar_xFRF.size)
        profile_extend_shiftback[tt,itrim] = profile_extend[tt,0:itrim.size]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,profile_extend_shiftback.T)
ax.set_title('profiles, extended - shifted back')

yplot5 = 100*np.sum(~np.isnan(profile_extend_shiftback),axis=0)/time_fullspan.size
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,yplot4,label='raw data')
ax.plot(lidar_xFRF,yplot3,label='best-avail/cleaned')
ax.plot(lidar_xFRF,yplot5,label='extended (shifted back)')
ax.set_xlabel('x [m]')
ax.set_ylabel('Percent available profiles (%)')
ax.legend()



# Plot where Acoef < 0.01
iiplot = np.where(Acoef < 0.06)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
# ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where Acoef < 0.06')
ax.set_ylim([-3,4])
ax.set_xlim([0, 150])
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
# ax.plot(lidar_xFRF_shift,profile_extend_testdata[iiplot,:].T,'k:')
ax.plot(prof_x_wl[iiplot],data_tidegauge[iiplot]+wlbuffer,'bo')
ax.set_title('Profiles where Acoef < 0.06 - EXTENDED')
ax.set_ylim([-3,4])
ax.set_xlim([0, 150])

for tt in iiplot[np.arange(0,iiplot.size,250)]:
    # fig, ax = plt.subplots()
    plt.plot(lidar_xFRF_shift,profile_extend[tt,:].T)
    plt.plot(lidar_xFRF_shift, lidarelev[tt, :].T,'--')
    plt.plot(prof_x_wl[tt], data_tidegauge[tt] + wlbuffer, '*b')
    figsavedir = 'C:/Users/rdchlerh/PycharmProjects/frf_python_share/figs/profile_extend/2024Dec03'
    plt.savefig('ii.png')
plt.savefig('foo.pdf')

plt.savefig('foo.png', bbox_inches='tight')



























tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
xplot = np.arange(nx) * dx
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(lidarslope, lidarslope.size)
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
zscatter = np.reshape(lidarelev, lidarelev.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m, NAVD88]')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')

# PLOT as profiles
fig, ax = plt.subplots()
ax.plot(xplot,lidarelev.T)
ax.set_ylabel('z [m, NAVD88]')
ax.set_xlabel('x [m, FRF]')

# PLOT slopes seaward of XCsea
XX, TT = np.meshgrid(xplot, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(lidarslope_seaward, lidarslope_seaward.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('avg. slope [m/m]')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')

# Plot unscaled profiles
xtmp = np.arange(unscaled_profile.shape[1]) * dx
XX, TT = np.meshgrid(xtmp, tplot)
timescatter = np.reshape(TT, TT.size)
xscatter = np.reshape(XX, XX.size)
zscatter = np.reshape(unscaled_profile, unscaled_profile.size)
tt = timescatter[~np.isnan(zscatter)]
xx = xscatter[~np.isnan(zscatter)]
zz = zscatter[~np.isnan(zscatter)]
fig, ax = plt.subplots()
ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('z [m]')
ax.set_xlabel('x [m, FRF]')
ax.set_ylabel('time')

# PLOT as profiles
fig, ax = plt.subplots()
ax.plot(xtmp, unscaled_profile.T)
ax.set_ylabel('z [m, NAVD88]')
ax.set_xlabel('x [m, FRF]')

## NOW try a statistical method...

# make single matrix of slopes to match size of zsmooth_equal_length
lidarslope_equallength = np.empty(lidarelev_equallength.shape)
lidarslope_equallength[:] = np.nan
numx = lidarslope_equallength.shape[1]
for jj in np.arange(time_fullspan.size):
    ytmp = lidarslope[jj,:]
    lidarslope_equallength[jj,np.arange(sum(~np.isnan(ytmp)))] = ytmp[np.arange(sum(~np.isnan(ytmp)))]
    numgrab = numx - sum(~np.isnan(ytmp))
    lidarslope_equallength[jj,np.arange(sum(~np.isnan(ytmp)),numx)] = lidarslope_seaward[jj,np.arange(numgrab)]

    # PLOT and COMPARE - lidarslope_equallength & lidarelev_equallength
    xplot = np.arange(numx) * dx
    XX, TT = np.meshgrid(xplot, tplot)
    timescatter = np.reshape(TT, TT.size)
    xscatter = np.reshape(XX, XX.size)
    zscatter = np.reshape(lidarslope_equallength, lidarslope_equallength.size)
    tt = timescatter[~np.isnan(zscatter)]
    xx = xscatter[~np.isnan(zscatter)]
    zz = zscatter[~np.isnan(zscatter)]
    fig, ax = plt.subplots()
    ph = ax.scatter(xx, tt, s=5, c=zz, cmap='rainbow', vmin=-0.1, vmax=0.25)
    cbar = fig.colorbar(ph, ax=ax)
    cbar.set_label('avg. slope [m/m]')
    ax.set_xlabel('x [m, FRF]')
    ax.set_ylabel('time')
    zscatter = np.reshape(lidarelev_equallength, lidarelev_equallength.size)
    tt = timescatter[~np.isnan(zscatter)]
    xx = xscatter[~np.isnan(zscatter)]
    zz = zscatter[~np.isnan(zscatter)]
    fig, ax = plt.subplots()
    ph = ax.scatter(xx, tt, s=5, c=zz, cmap='viridis')
    cbar = fig.colorbar(ph, ax=ax)
    cbar.set_label('z [m]')
    ax.set_xlabel('x [m, FRF]')
    ax.set_ylabel('time')

# Ok, now we can try to make a dataframe for z(x_i) as a function of knowns......
z_known = unscaled_profile
numx = z_known.shape[1]
xtrue = np.empty(shape=z_known.shape)
xscaled = xtrue/profile_width
# data_wave8m,data_tidegauge,data_lidar_elev2p


