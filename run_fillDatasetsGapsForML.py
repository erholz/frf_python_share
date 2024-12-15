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



picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_10Dec2024/'
with open(picklefile_dir+'datasets_ML_14Dec2024.pickle', 'rb') as file:
    datasets_ML = pickle.load(file)
    num_datasets = len(datasets_ML)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data/processed_26Nov2024/'
with open(picklefile_dir+'lidar_xFRF.pickle', 'rb') as file:
    lidar_xFRF = np.array(pickle.load(file))
    lidar_xFRF = lidar_xFRF[0][:]

# Examine some sample profiles
for jj in np.floor(np.linspace(0,len(datasets_ML)-1,20)):
    varname = outputname = 'dataset_' + str(int(jj))
    exec('timeslice = datasets_ML["' + varname + '"]["set_timeslice"]')
    exec('topobathy = datasets_ML["' + varname + '"]["set_topobathy"]')
    fig, ax = plt.subplots()
    ax.plot(lidar_xFRF,topobathy,'.')
    tplot = pd.to_datetime(timeslice, unit='s', origin='unix')
    ax.set_title(str(tplot[0]))

# Ok, try to fill in from edge of profiles to at least z = -1m
def equilibriumprofile_func_2param(x, a, b):
    return a * x * np.exp(b)
def equilibriumprofile_func_1param(x, a):
    return a * x ** (2/3)

# for jj in np.floor(np.linspace(0,len(datasets_ML)-1,20)):
dx = 0.1
nx = lidar_xFRF.size
avg_fiterror = np.empty(num_datasets,)
avg_fiterror[:] = np.nan
avg_Acoef = np.empty(num_datasets,)
avg_Acoef[:] = np.nan
numprof_notextended = np.empty(num_datasets,)
numprof_notextended[:] = np.nan
avg_zobsfinal = np.empty(num_datasets,)
avg_zobsfinal[:] = np.nan
# for jji in np.arange(np.floor(num_datasets/10)):
for jj in np.arange(num_datasets):
#     jj = int(jji)
    varname = outputname = 'dataset_' + str(int(jj))
    exec('timeslice = datasets_ML["' + varname + '"]["set_timeslice"]')
    exec('topobathy = datasets_ML["' + varname + '"]["set_topobathy"]')
    exec('waterlevel = datasets_ML["' + varname + '"]["set_waterlevel"]')

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
        # tt = int(ttii)
        Lgrab = 5          # try fitting equilibrium profile to last [Lgrab] meters of available data
        watlev_tt = waterlevel[tt]
        wlbuffer = 0.25
        if (sum(~np.isnan(topobathy[:,tt])) > 10) & (~np.isnan(watlev_tt)):
            ii_submerged = np.where(topobathy[:, tt] <= watlev_tt + wlbuffer)[0]
            iiclose = np.where(abs(topobathy[:, tt]-(wlbuffer+watlev_tt)) == np.nanmin(abs(topobathy[:,tt]-(wlbuffer+watlev_tt))))[0]
            if iiclose.size > 1:
                iiclose = iiclose[0]
            prof_x_wl[tt] = np.interp((wlbuffer+watlev_tt)[0],topobathy[np.arange(iiclose-1,iiclose+1),tt],lidar_xFRF[np.arange(iiclose-1,iiclose+1)])
            if len(ii_submerged) > 5:
                id_last = sum(~np.isnan(topobathy[:,tt]))
                if ii_submerged.size > Lgrab/dx:
                    numgrab = Lgrab/dx
                else:
                    numgrab = ii_submerged.size
                iitest = ii_submerged
                htmp = (wlbuffer + watlev_tt) - topobathy[iitest,tt]
                xtmp = dx*np.arange(htmp.size)
                iitest = iitest[~np.isnan(htmp)]
                xtmp = xtmp[~np.isnan(htmp)]
                htmp = htmp[~np.isnan(htmp)]
                zobs_final[tt] = (wlbuffer + watlev_tt) - htmp[-1]
                profile_extend_testdata[iitest,tt] = topobathy[iitest,tt]
                htmp = htmp - htmp[0]   # make initial value 0
                # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, htmp, bounds=([0, -np.inf], [15, np.inf]))
                # popt, pcov = curve_fit(equilibriumprofile_func_2param, xtmp, ztmp)
                popt, pcov = curve_fit(equilibriumprofile_func_1param, xtmp, htmp, bounds=([0.05], [1]))
                Acoef[tt] = popt[0]
                # bcoef[tt] = popt[1]
                hfit = equilibriumprofile_func_1param(xtmp, *popt)
                fitrmse[tt] = np.sqrt(np.mean((hfit-htmp)**2))
                # x_extend = np.arange(lidar_xFRF_shift[iitest[-1]]+dx,lidar_xFRF_shift[-1],dx)
                x_extend = np.arange(lidar_xFRF[iitest[0]] + dx, lidar_xFRF[-1], dx)
                x_extend_norm = x_extend - x_extend[0]
                z_extend = (wlbuffer + watlev_tt) - equilibriumprofile_func_1param(x_extend_norm, *popt)
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
    if sum(~np.isnan(Acoef)) > 0:
        avg_Acoef[jj] = np.nanmean(Acoef)
        avg_fiterror[jj] = np.nanmean(fitrmse)
        numprof_notextended[jj] = sum(np.isnan(Acoef))
        avg_zobsfinal[jj] = np.nanmean(zobs_final)
    elif sum(~np.isnan(Acoef)) == 0:
        numprof_notextended[jj] = sum(np.isnan(Acoef))

fig, ax = plt.subplots()
ax.plot(avg_fiterror,avg_Acoef,'.')
fig, ax = plt.subplots()
ax.plot(numprof_notextended,'o')
ax.plot(avg_zobsfinal,'.')



