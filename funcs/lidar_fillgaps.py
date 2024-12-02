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
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,profile_fullspan)
ax.set_title('Profiles - all, best available')



## FIRST find last [edge_length] of profile, try to fit equilibrium profile
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
# for tt in np.arange(10):
# for tt in ([ 53613, 59241, 65637, 65861, 67527, 67528, 67576, 67577, 67638]):
for tt in np.arange(time_fullspan.size):
    Lgrab = 3          # try fitting equilibrium profile to last [Lgrab] meters of available data
    watlev_tt = data_tidegauge[tt]
    wlbuffer = 0.5
    if (sum(~np.isnan(lidarelev[tt,:])) > 10) & (~np.isnan(watlev_tt)) :
        ii_submerged = np.where(lidarelev[tt, :] <= watlev_tt + wlbuffer)[0]
        if len(ii_submerged) > 5:
            id_last = sum(~np.isnan(lidarelev[tt,:]))
            if ii_submerged.size > Lgrab/dx:
                numgrab = Lgrab/dx
            else:
                numgrab = ii_submerged.size
            iitest = np.arange(ii_submerged[0],ii_submerged[0]+numgrab).astype(int)
            htmp = (wlbuffer + watlev_tt) - lidarelev[tt,iitest]
            xtmp = dx*np.arange(htmp.size)
            iitest = iitest[~np.isnan(htmp)]
            xtmp = xtmp[~np.isnan(htmp)]
            htmp = htmp[~np.isnan(htmp)]
            zobs_final = (wlbuffer + watlev_tt) - htmp[-1]
            # xtmp[ztmp < 0] = []     # remove negative values
            # ztmp[ztmp < 0] = []     # remove negative values
            htmp = htmp - htmp[0]   # make initial value 0
            # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, htmp, bounds=([0, -np.inf], [15, np.inf]))
            # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, ztmp)
            popt, pcov = curve_fit(equilibriumprofile_func_1param, xtmp, htmp, bounds=([0], [5]))
            Acoef[tt] = popt[0]
            # bcoef[tt] = popt[1]
            hfit = equilibriumprofile_func_1param(xtmp, *popt)
            fitrmse[tt] = np.sqrt(np.mean((hfit-htmp)**2))
            # x_extend = np.arange(lidar_xFRF_shift[iitest[-1]]+dx,lidar_xFRF_shift[-1],dx)
            x_extend = np.arange(lidar_xFRF_shift[iitest[0]] + dx, lidar_xFRF_shift[-1], dx)
            x_extend_norm = x_extend - x_extend[0]
            z_extend = (wlbuffer + watlev_tt) - equilibriumprofile_func_1param(x_extend_norm, *popt)
            # profile_extend[tt,np.arange(iitest[-1]+1,nx-1)] = zobs_final - z_extend
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
# ax.plot(Acoef,fitrmse,'o')
ph = ax.scatter(Acoef,fitrmse,s=5, c=profile_width, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
cbar.set_label('profile width [m]')
ax.set_xlabel('A coeff')
ax.set_ylabel('RMSE [m]')
fig, ax = plt.subplots()
# ax.plot(bcoef,fitrmse,'o')
ph = ax.scatter(bcoef,fitrmse,s=5, c=profile_width, cmap='rainbow')
cbar = fig.colorbar(ph, ax=ax)
ax.set_xlabel('b coeff')
cbar.set_label('profile width [m]')
ax.set_ylabel('RMSE [m]')
fig, ax = plt.subplots()
yplot = fitrmse[fitrmse <= 0.05]
plt.hist(yplot,bins=np.linspace(0,0.05,50))
ax.set_xlabel('RMSE [m] < 0.1 m')
fig, ax = plt.subplots()
yplot = fitrmse[fitrmse > 0.05]
plt.hist(fitrmse,bins=np.linspace(0.05,np.nanmax(fitrmse),50))
ax.set_xlabel('RMSE [m] > 0.1 m')

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
iiplot = id_below_watlev
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles where extension applied')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles with EXTENSIONS')


# Plot where Acoef < 0.01
iiplot = np.where(Acoef < 0.01)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles where Acoef < 0.01')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles where Acoef < 0.01 - EXTENDED')

# Plot where Acoef < 0.02
iiplot = np.where((Acoef >= 0.01) & (Acoef < 0.02))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles where 0.01 < Acoef < 0.02')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles where 0.01 < Acoef < 0.02 - EXTENDED')

# Plot where Acoef < 0.03
iiplot = np.where((Acoef >= 0.02) & (Acoef < 0.03))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles where 0.02 < Acoef < 0.03')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles where 0.02 < Acoef < 0.03 - EXTENDED')

# Plot where Acoef < 0.05
iiplot = np.where((Acoef >= 0.03) & (Acoef < 0.05))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles where 0.03 < Acoef < 0.05')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles where 0.03 < Acoef < 0.05 - EXTENDED')



# Plot where Acoef > 0.1
iiplot = np.where((Acoef >= 0.05))[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles where Acoef >= 0.05')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles where Acoef >= 0.05 - EXTENDED')



# Plot the profiles where the fit-error is low
iiplot = np.where(fitrmse < 0.015)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,lidarelev[iiplot,:].T)
ax.set_title('Profiles with low fit error (RMSE < 15 cm)')
fig, ax = plt.subplots()
ax.plot(Acoef[iiplot],bcoef[iiplot],'o')
ax.set_xlabel('A coef')
ax.set_ylabel('b coef')
ax.set_title('Profiles with low fit error (RMSE < 15 cm)')
fig, ax = plt.subplots()
ax.plot(lidar_xFRF_shift,profile_extend[iiplot,:].T)
ax.set_title('Profiles with low fit error (RMSE < 15 cm) - EXTENDED')


# Plot the profiles where the fit-error is high
iiplot = np.where(fitrmse > 0.1)[0]
fig, ax = plt.subplots()
ax.plot(lidar_xFRF,lidarelev[iiplot,:].T)
ax.set_title('Profiles with high fit error (RMSE > 100 cm)')
fig, ax = plt.subplots()
ax.plot(Acoef[iiplot],bcoef[iiplot],'o')
ax.set_xlabel('A coef')
ax.set_ylabel('b coef')
ax.set_title('Profiles with high fit error (RMSE > 100 cm)')





































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


