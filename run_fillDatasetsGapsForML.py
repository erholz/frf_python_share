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
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline, splev, CubicSpline
from funcs.find_nangaps import *



picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
with open(picklefile_dir+'datasets_ML_14Dec2024.pickle', 'rb') as file:
    datasets_ML = pickle.load(file)
    num_datasets = len(datasets_ML)
with open(picklefile_dir+'set_id_tokeep_14Dec2024.pickle', 'rb') as file:
    set_id_tokeep, plot_start_iikeep = pickle.load(file)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]
with open(picklefile_dir+'IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,_,_,_,_,_,_,_,_,_,_,_,_ = pickle.load(file)

# Examine some sample profiles
for jj in np.floor(np.linspace(0,len(datasets_ML)-1,20)):
    varname = outputname = 'dataset_' + str(int(jj))
    exec('timeslice = datasets_ML["' + varname + '"]["set_timeslice"]')
    exec('topobathy = datasets_ML["' + varname + '"]["set_topobathy"]')
    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF,topobathy,'.')
    tplot = pd.to_datetime(timeslice, unit='s', origin='unix')
    ax.set_title(str(tplot[0]))

# First try interpolating in the cross-shore
Nlook = 4*24
topobaty_prexshoreinterp = np.empty((lidar_xFRF.size,Nlook,num_datasets))
topobaty_prexshoreinterp[:] = np.nan
topobaty_postxshoreinterp = np.empty((lidar_xFRF.size,Nlook,num_datasets))
topobaty_postxshoreinterp[:] = np.nan
for jj in np.arange(num_datasets):
# for jj in np.arange(2214):
    varname = outputname = 'dataset_' + str(int(jj))
    exec('timeslice = datasets_ML["' + varname + '"]["set_timeslice"]')
    exec('topobathy = datasets_ML["' + varname + '"]["set_topobathy"]')
    topobaty_prexshoreinterp[:,:,jj] = topobathy[:]
    topobaty_postxshoreinterp[:,:,jj] = topobathy[:]

    # find cross-shore contour position
    mwl = -0.13
    zero = 0
    mhw = 3.6
    dune_toe = 3.22
    cont_elev = np.array([mhw]) #np.arange(0,2.5,0.5)   # <<< MUST BE POSITIVELY INCREASING
    cont_ts, cmean, cstd = create_contours(topobathy.T,timeslice,lidar_xFRF,cont_elev)
    # meanprofile = np.nanmean(topobathy,axis=1)
    ix_cont = np.nanmax(np.where(lidar_xFRF <= np.nanmax(cont_ts))[0])-5

    # go through x-shore locations ix_cont -> end
    for ii in np.arange(ix_cont,nx):
        xshore_slice = topobathy[ii,:]
        percent_avail = sum(~np.isnan(xshore_slice))/Nlook
        if percent_avail >= 0.66:
            tin = np.arange(0,Nlook)
            zin = xshore_slice
            tin = tin[~np.isnan(zin)]
            zin = zin[~np.isnan(zin)]
            zout = np.interp(np.arange(0,Nlook),tin,zin)
            topobaty_postxshoreinterp[ii,:,jj] = zout
# fig, ax = plt.subplots()
# ax.plot(lidar_xFRF,topobaty_prexshoreinterp[:,:,jj],'o')
# # fig, ax = plt.subplots()
# ax.plot(lidar_xFRF, topobaty_postxshoreinterp[:, :, jj],'.')

# # SAVE THIS
# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# with open(picklefile_dir+'topobathy_xshoreinterp.pickle','wb') as file:
#     pickle.dump([topobaty_prexshoreinterp,topobaty_postxshoreinterp], file)











# Ok, try to fill in from edge of profiles to at least z = -1m
def equilibriumprofile_func_2param(x, a, b):
    return a * x * np.exp(b)
def equilibriumprofile_func_1param(x, a):
    return a * x ** (2/3)

# Initialize the aggregate topobathy for pre- and post- extensions
dx = 0.1
nx = lidar_xFRF.size
Nlook = 4*24
topobaty_preextend = np.empty((lidar_xFRF.size,Nlook,num_datasets))
topobaty_preextend[:] = np.nan
topobaty_postextend = np.empty((lidar_xFRF.size,Nlook,num_datasets))
topobaty_postextend[:] = np.nan
avg_fiterror = np.empty(num_datasets,)
avg_fiterror[:] = np.nan
avg_Acoef = np.empty(num_datasets,)
avg_Acoef[:] = np.nan
numprof_notextended = np.empty(num_datasets,)
numprof_notextended[:] = np.nan
avg_zobsfinal = np.empty(num_datasets,)
avg_zobsfinal[:] = np.nan
for jj in np.arange(num_datasets):
# for jj in np.arange(2214):
    varname = outputname = 'dataset_' + str(int(jj))
    exec('timeslice = datasets_ML["' + varname + '"]["set_timeslice"]')
    # exec('topobathy = datasets_ML["' + varname + '"]["set_topobathy"]')
    topobathy = topobaty_postxshoreinterp[:,:,jj]
    exec('waterlevel = datasets_ML["' + varname + '"]["set_waterlevel"]')
    topobaty_preextend[:,:,jj] = topobathy[:]
    topobaty_postextend[:,:,jj] = topobathy[:]

    # initialize fit coefficints
    Acoef = np.empty(shape=timeslice.shape)
    Acoef[:] = np.nan
    # bcoef = np.empty(shape=timeslice.shape)
    # bcoef[:] = np.nan
    fitrmse = np.empty(shape=timeslice.shape)
    fitrmse[:] = np.nan
    profile_extend = np.empty(shape=topobathy.shape)
    profile_extend[:] = topobathy[:]
    profile_extend_testdata = np.empty(shape=topobathy.shape)
    profile_extend_testdata[:] = np.nan
    prof_x_wl = np.empty(shape=timeslice.shape)
    prof_x_wl[:] = np.nan
    zobs_final = np.empty(shape=timeslice.shape)
    zobs_final[:] = np.nan
    for tt in np.arange(timeslice.size):
        Lgrab = 5          # try fitting equilibrium profile to last [Lgrab] meters of available data
        watlev_tt = waterlevel[tt]
        wlbuffer = 0.25
        # first find if there are ANY gaps in the profile
        zinput = topobathy[:, tt]
        ix_notnan = np.where(~np.isnan(zinput))[0]
        if len(ix_notnan) > 0:
            zinput = zinput[np.arange(ix_notnan[0],ix_notnan[-1])]
            gapstart, gapend, gapsize, maxgap = find_nangaps(zinput)
            # if maxgap > 50:
            #     print('max gap size = ' + str(maxgap)+' for tt = '+str(tt))
        if (sum(~np.isnan(topobathy[:,tt])) > 10) & (~np.isnan(watlev_tt)):
            ii_submerged = np.where(topobathy[:, tt] <= watlev_tt + wlbuffer)[0]
            iiclose = np.where(abs(topobathy[:, tt]-(wlbuffer+watlev_tt)) == np.nanmin(abs(topobathy[:,tt]-(wlbuffer+watlev_tt))))[0]
            doublebuffer_flag = 0
            if iiclose.size > 1:
                iiclose = iiclose[0]
            prof_x_wl[tt] = np.interp((wlbuffer + watlev_tt)[0], topobathy[np.arange(iiclose - 1, iiclose + 1), tt],lidar_xFRF[np.arange(iiclose - 1, iiclose + 1)])
            if len(ii_submerged) <= 5:
                ii_submerged = np.where(topobathy[:, tt] <= watlev_tt + 2*wlbuffer)[0]
                iiclose = np.where(abs(topobathy[:, tt] - (2*wlbuffer + watlev_tt)) == np.nanmin(abs(topobathy[:, tt] - (2*wlbuffer + watlev_tt))))[0]
                if iiclose.size > 1:
                    iiclose = iiclose[0]
                prof_x_wl[tt] = np.interp((wlbuffer + 2*watlev_tt)[0], topobathy[np.arange(iiclose - 1, iiclose + 1), tt],lidar_xFRF[np.arange(iiclose - 1, iiclose + 1)])
                doublebuffer_flag = 1
            if len(ii_submerged) > 5:
                # print('double buffer = ' + str(doublebuffer_flag) + 'for tt = '+str(tt))
                iitest = ii_submerged
                if doublebuffer_flag == 1:
                    htmp = (2*wlbuffer + watlev_tt) - topobathy[iitest, tt]
                else:
                    htmp = (wlbuffer + watlev_tt) - topobathy[iitest,tt]
                xtmp = dx*np.arange(htmp.size)
                iitest = iitest[~np.isnan(htmp)]
                xtmp = xtmp[~np.isnan(htmp)]
                htmp = htmp[~np.isnan(htmp)]
                if doublebuffer_flag == 1:
                    zobs_final[tt] = (2*wlbuffer + watlev_tt) - htmp[-1]
                else:
                    zobs_final[tt] = (wlbuffer + watlev_tt) - htmp[-1]
                profile_extend_testdata[iitest,tt] = topobathy[iitest,tt]
                h_init = htmp[0]
                htest = htmp - h_init # make initial value 0
                # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, htmp, bounds=([0, -np.inf], [15, np.inf]))
                # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, ztmp)
                popt, pcov = curve_fit(equilibriumprofile_func_1param, xtmp, htest, bounds=([0.15], [0.3]))
                Acoef[tt] = popt[0]
                # bcoef[tt] = popt[1]
                hfit = equilibriumprofile_func_1param(xtmp, *popt)
                fitrmse[tt] = np.sqrt(np.mean((hfit-htest)**2))
                # x_extend = np.arange(lidar_xFRF_shift[iitest[-1]]+dx,lidar_xFRF_shift[-1],dx)
                x_extend = np.arange(lidar_xFRF[iitest[0]] + dx, lidar_xFRF[-1], dx)
                x_extend_norm = x_extend - x_extend[0]
                h_extend_norm = equilibriumprofile_func_1param(x_extend_norm, *popt)
                h_extend = h_extend_norm + h_init
                if doublebuffer_flag == 1:
                    z_extend = (2*wlbuffer + watlev_tt) - h_extend
                else:
                    z_extend = (wlbuffer + watlev_tt) - h_extend
                profile_extend[np.arange(iitest[0] + 1, nx - 1),tt] = z_extend
                # # # # # # #
                # # PLOT
                # fig, ax = plt.subplots()
                # ax.plot(lidar_xFRF[iitest],watlev_tt - htmp,'o')
                # # ax.plot(lidar_xFRF_shift[iitest],watlev_tt - equilibriumprofile_func_2param(xtmp, *popt), label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
                # ax.plot(lidar_xFRF[iitest], watlev_tt - equilibriumprofile_func_1param(xtmp, *popt),
                #         label='fit: a=%5.3f' % tuple(popt))
                # # ax.plot(x_extend,watlev_tt - z_extend)
                # ax.plot(lidar_xFRF,profile_extend[:,tt],':k',label='extend profile')
                # ax.plot(lidar_xFRF,watlev_tt*np.ones(shape=lidar_xFRF.shape),'b',label='waterline')
                # ax.legend()
        topobaty_postextend[:,:,jj] = profile_extend
    if sum(~np.isnan(Acoef)) > 0:
        avg_Acoef[jj] = np.nanmean(Acoef)
        avg_fiterror[jj] = np.nanmean(fitrmse)
        numprof_notextended[jj] = sum(np.isnan(Acoef))
        avg_zobsfinal[jj] = np.nanmean(zobs_final)
    elif sum(~np.isnan(Acoef)) == 0:
        numprof_notextended[jj] = sum(np.isnan(Acoef))

# picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
# with open(picklefile_dir+'topobathy_extend.pickle','wb') as file:
#     pickle.dump([topobaty_preextend,topobaty_postextend], file)


num_profiles = int(topobathy_preextend.size[1]*topobathy_preextend.size[2])
topobathy_check = np.empty((lidar_xFRF.size,))


fig, ax = plt.subplots()
for jj in np.arange(num_datasets):
    ax.plot(lidar_xFRF,topobaty_postextend[:,:,jj])






fig, ax = plt.subplots()
ax.plot(avg_fiterror,avg_Acoef,'.')
fig, ax = plt.subplots()
tplot = pd.to_datetime(time_fullspan[plot_start_iikeep[set_id_tokeep].astype(int)], unit='s', origin='unix')
ax.plot(tplot,numprof_notextended,'.')
fig, ax = plt.subplots()
ax.plot(tplot,avg_zobsfinal,'.')

for tti in (np.floor(np.linspace(40,num_datasets-1,20))):
    tt = int(tti)
    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF,topobaty_postextend[:,:,tt])

    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF,topobaty_postextend[:,:,jj])
    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF, topobaty_preextend[:, :, jj])

# cubic spline fix - tt =  1,26,76
exec('topobathy = datasets_ML["' + varname + '"]["set_topobathy"]')
mhw = 3.6
tt = 1
xprof_tt = lidar_xFRF[0:]
zprof_tt = topobathy[:, tt]
ix_notnan = np.where(~np.isnan(zprof_tt))[0]
zinput = zprof_tt[np.arange(ix_notnan[0], ix_notnan[-1])]
xinput = xprof_tt[np.arange(ix_notnan[0], ix_notnan[-1])]
gapstart, gapend, gapsize, maxgap = find_nangaps(zinput)

ii_tofill = np.where((np.isnan(zinput)))[0]
xin = xinput[~np.isnan(zinput)]
yin = zinput[~np.isnan(zinput)]
cs = CubicSpline(xin, yin, bc_type='natural')
Lspline = xtmp[-1] - xtmp[0]
xspline = np.linspace(xtmp[0],xtmp[-1],int(np.ceil(Lspline/0.01)))
zspline_tmp = cs(xspline)



zspline_tfill = np.interp(lidar_xFRF[ii_tofill],xspline,zspline_tmp)

fig, ax = plt.subplots()
ax.plot(xprof_tt,zprof_tt,'o')
ax.plot(xspline,zspline_tmp,'.')
ax.plot(lidar_xFRF[ii_tofill],zspline_tfill,'x')





















## Work on wave and waterlevel interp
for jji in np.arange(np.floor(np.linspace(40,num_datasets-1,20))):
    jj = int(jji)
    varname = outputname = 'dataset_' + str(int(jj))
    exec('timeslice = datasets_ML["' + varname + '"]["set_timeslice"]')
    exec('waterlevel = datasets_ML["' + varname + '"]["set_waterlevel"]')
    exec('Hs8m = datasets_ML["' + varname + '"]["set_Hs8m"]')
    exec('Tp8m = datasets_ML["' + varname + '"]["set_Tp8m"]')
    exec('dir8m = datasets_ML["' + varname + '"]["set_dir8m"]')

    numnan_watlev = sum(np.isnan(waterlevel))
    if numnan_watlev > 0:
        tq = timeslice
        ttmp = timeslice
        ytmp = waterlevel
        ttmp = ttmp[~np.isnan(waterlevel)]
        ytmp = ytmp[~np.isnan(waterlevel)]
        watlev_interp = np.interp(tq,ttmp,ytmp)

    numnan_Hs8m = sum(np.isnan(Hs8m))
    if numnan_Hs8m > 0:
        tq = timeslice
        ttmp = timeslice
        ytmp = Hs8m
        ttmp = ttmp[~np.isnan(Hs8m)]
        ytmp = ytmp[~np.isnan(Hs8m)]
        Hs8m_interp = np.interp(tq,ttmp,ytmp)

    numnan_Tp8m = sum(np.isnan(Tp8m))
    if numnan_Tp8m > 0:
        tq = timeslice
        ttmp = timeslice
        ytmp = Tp8m
        ttmp = ttmp[~np.isnan(Tp8m)]
        ytmp = ytmp[~np.isnan(Tp8m)]
        Tp8m_interp = np.interp(tq,ttmp,ytmp)

    numnan_dir8m = sum(np.isnan(dir8m))
    if numnan_dir8m > 0:
        tq = timeslice
        ttmp = timeslice
        ytmp = dir8m
        ttmp = ttmp[~np.isnan(dir8m)]
        ytmp = ytmp[~np.isnan(dir8m)]
        dir8m_interp = np.interp(tq,ttmp,ytmp)

    if (numnan_watlev + numnan_Hs8m + numnan_Tp8m + numnan_dir8m) > 0:
        fig, ax = plt.subplots()
        ax.plot(timeslice,watlerlevel)
        ax.plot(timeslice, watlev_interp,'o')
        ax.plot(timeslice,Hs8m)
        ax.plot(timeslice, Hs8m_interp,'o')
        ax.plot(timeslice,Tp8m)
        ax.plot(timeslice, Tp8m_interp,'o')
        ax.plot(timeslice,dir8m)
        ax.plot(timeslice, dir8m_interp,'o')
