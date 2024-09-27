import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import time
import pickle


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

# CREATE FUNCTION HERE THAT EXTENDS PROFILES TO EQUAL LENGTH
#
# def prof_extendfromslopes(lidarelev,lidarslope,scaled_profiles,scaled_avgslope,shift_avgslope_beyondXCsea,
#                           time_fullspan,data_wave8m,data_tidegauge,data_lidar_elev2p):


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
with open('elev_processed_unscaled_xi.pickle', 'rb') as file:
    unscaled_xi = pickle.load(file)





lidarelev = shift_zsmooth
lidarslope = shift_avgslope
lidarslope_seaward = shift_avgslope_beyondXCsea
lidarelev_equallength = unscaled_profile


with open('IO_alignedintime.pickle', 'rb') as file:
    _,data_wave8m,_,data_tidegauge,data_lidar_elev2p,_,_,_,_,_,_,_,_ = pickle.load(file)


# figure out size of datasets
nx = lidarelev.shape[1]
dx = 0.1

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


