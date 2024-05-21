import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from run_code import cont_elev,num_profs_plot,time_end,time_beg,TOI_duration,tzinfo
from funcs.lidar_check import daily_znum,daily_zstdev
from funcs.create_contours import cont_ts,cmean,cstd
from run_lidarcollect import lidartime,lidarelev,lidar_xFRF
from run_hydrocollect import wltime_lidar,wlmax_lidar,wlmin_lidar
from funcs.calculate_beachvol import beachVol, beachVol_xc, dBeachVol_dt, DT


# plot_ContourTimeSeries - Plot contour time series
def plot_ContourTimeSeries():
    fig, (ax1,ax2) = plt.subplots(2)
    cmap = plt.cm.rainbow(np.linspace(0,1,cont_elev.size))
    ax1.set_prop_cycle('color', cmap)
    tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
    for cc in np.arange(cont_elev.size):
        ax1.scatter(tplot, cont_ts[cc, :], s=1, label='z = ' + str(cont_elev[cc]) + ' m')
    ax1.grid(which='major',axis='both')
    # ax1.legend()
    ax1.set_ylabel('xFRF [m]')
    ax1.set_xlim(min(tplot),max(tplot))
    plt.suptitle(time_beg+' to '+time_end)
    plt.gcf().autofmt_xdate()
    tplot = pd.to_datetime(wltime_lidar, unit='s', origin='unix')
    ax2.scatter(tplot,np.nanmin(wlmin_lidar,axis=1),s=1,color=[0.5,0.5,0.5],label='min WL')
    ax2.scatter(tplot,np.nanmax(wlmax_lidar,axis=1),s=1,color='k',label='max WL')
    ax2.set_xlabel('time')
    ax2.set_ylabel('z [m]')
    ax2.grid(which='major',axis='both')
    ax2.set_xlim(min(tplot),max(tplot))
    ax2.legend()


# plot_ProfilesSubset - Make plots of profiles through time
def plot_ProfilesSubset():
    numprofs = np.array(lidarelev.shape)[0]
    print('There are ' + str(numprofs) + ' profiles in this subset')
    print('How many do you want to plot? --> Set [num_profs_plot]')
    # numplot = numprofs
    numplot = num_profs_plot
    print('Ok, plotting ' + str(numplot) + ' profiles...')
    iiplot = np.round(np.linspace(0, numprofs - 1, numplot)).astype(int)
    cmap = plt.cm.rainbow(np.linspace(0, 1, numplot))
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', cmap)
    if TOI_duration.days > 4:
        if TOI_duration.days > 365:
            leg_format = '%m/%d/%y'
        else:
            leg_format = '%m/%d/%y'
    elif TOI_duration.days <= 1:
        leg_format = '%H:%M'
    else:
        leg_format = '%m/%d %H:%M'
    for ii in iiplot:
        time_obj = dt.datetime.fromtimestamp(lidartime[ii], tzinfo)
        plt.plot(lidar_xFRF, lidarelev[ii, :], label=time_obj.strftime(leg_format))
    plt.legend()
    plt.title(time_beg + ' to ' + time_end)
    plt.grid(which='major', axis='both')
    plt.xlabel('xFRF [m]')
    plt.ylabel('z [m]')
    plt.gcf().autofmt_xdate()

# plot_ProfilesTimestack - Plot all profiles as timestacks
def plot_ProfilesTimestack():
    fig, ax = plt.subplots()
    # ph = ax.pcolormesh(tplot, lidar_xFRF, np.rot90(lidarelev,k=-1))
    tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
    XX,TT = np.meshgrid(lidar_xFRF,tplot)
    ph = ax.scatter(np.reshape(TT,TT.size),np.reshape(XX,XX.size),s=1,c=np.reshape(lidarelev,lidarelev.size))
    cbar1 = fig.colorbar(ph,ax=ax)
    cbar1.set_label('z [m]')
    xnotnan = lidar_xFRF[np.sum(np.isnan(lidarelev),axis=0) != len(lidartime)]
    ax.set_ylim(np.nanmin(xnotnan),np.nanmax(xnotnan))
    plt.gcf().autofmt_xdate()
    ax.set_xlim(min(tplot),max(tplot))

# plot_QualityDataWithContourPositions - Availability of "quality" data (no-nans) as a func. of xFRF
def plot_QualityDataWithContourPositions():
    fig, ax = plt.subplots()
    xplot = lidar_xFRF
    yplot = np.sum(~np.isnan(lidarelev), axis=0) / np.array(lidarelev.shape)[0]
    ax.plot(xplot, yplot, 'k', linewidth=2)
    plt.xlabel('xFRF [m]')
    plt.ylabel('fraction of data "passing" qaqc over time [0-1]')
    plt.show()
    plt.grid(which='major', axis='both')
    plt.title(time_beg + ' to ' + time_end)
    # add location of avg and stdev of contour xlocs
    cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size))
    ax.set_prop_cycle('color', cmap)
    for cc in np.arange(cont_elev.size):
        plt.plot([0, 0] + cmean[cc], [0, 1])
    for cc in np.arange(cont_elev.size):
        left, bottom, width, height = (cmean[cc] - cstd[cc], 0, cstd[cc] * 2, 1)
        patch = plt.Rectangle((left, bottom), width, height, alpha=0.1, color=cmap[cc, :])
        ax.add_patch(patch)

# plot_QualityDataTimeSeries - Availability of "quality" data (no-nans) as a func. of time
def plot_QualityDataTimeSeries():
    fig, ax = plt.subplots()
    tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
    yplot = np.sum(~np.isnan(lidarelev), axis=1) / np.array(lidarelev.shape)[1]
    ax.scatter(tplot, yplot, s=1)
    plt.xlabel('time')
    plt.ylabel('fraction of data "passing" qaqc over profile [0-1]')
    plt.title(time_beg + ' to ' + time_end)
    plt.gcf().autofmt_xdate()

# plot_DailyVariationTimestack - ok now plot elevation variation as a function of time and space...
def plot_DailyVariationTimestack():
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    yplot = list(reversed(lidar_xFRF))
    tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
    vplot = np.rot90(daily_zstdev,1)
    # vplot[vplot > 0.05] = np.nan
    ph1 = ax1.pcolormesh(tplot, yplot, vplot)
    cbar1 = fig.colorbar(ph1,ax=ax1)
    cbar1.set_label('Z_std [m]')
    ax1.set_ylabel('xFRF [m]')
    ax1.set_title('24-hr moving average')
    xnotnan = lidar_xFRF[np.sum(np.isnan(lidarelev),axis=0) != len(lidartime)]
    ax1.set_ylim(np.nanmin(xnotnan),np.nanmax(xnotnan))
    vplot = np.rot90(daily_znum/25)
    # vplot[vplot < 0.3] = np.nan
    ph2 = ax2.pcolormesh(tplot, yplot, vplot)
    cbar2 = fig.colorbar(ph2,ax=ax2)
    cbar2.set_label('frac avail')
    ax2.set_ylabel('xFRF [m]')
    ax2.set_ylim(np.nanmin(xnotnan),np.nanmax(xnotnan))
    # ax2.set_title('24-hr moving average')
    tmp1 = np.rot90(daily_zstdev,1)
    tmp2 = np.rot90(daily_znum/25)
    vplot = np.zeros(shape=tmp2.shape)
    vplot[((tmp1 <= 0.05) & (tmp2 > 0.3))] = 1
    vplot[vplot == 0] = np.nan
    ph3 = ax3.pcolormesh(tplot, yplot, vplot)
    cbar3 = fig.colorbar(ph3,ax=ax3)
    cbar3.set_label('StDv < 5cm & Hrs/Day > 30% ')
    ax3.set_ylabel('xFRF [m]')
    ax3.set_ylim(np.nanmin(xnotnan),np.nanmax(xnotnan))
    plt.gcf().autofmt_xdate()


def plot_BeachVolume():
    # Make some plots
    fig, (ax1, ax2) = plt.subplots(2)
    cmap = plt.cm.rainbow(np.linspace(0, 1, cont_elev.size + 1))
    ax1.set_prop_cycle('color', cmap[1:-1, :])
    tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
    for cc in np.arange(cont_elev.size - 1):
        ax1.scatter(tplot, beachVol[cc, :], s=1, label='z = ' + str(cont_elev[cc]) + ' m')
    # ax1.scatter(tplot,total_beachVol/total_obsBeachWid,s=1,color='k',label='Total Vol/Total obs. width')
    ax1.grid(which='major', axis='both')
    ax1.legend()
    ax1.set_ylabel('Profile Vol [m^2]')
    plt.suptitle(time_beg + ' to ' + time_end)
    tmptime = lidartime[0:len(lidartime) - 1] + DT / 2
    tplot = pd.to_datetime(tmptime, unit='s', origin='unix')
    ax2.set_prop_cycle('color', cmap[1:-1, :])
    for cc in np.arange(cont_elev.size - 1):
        ax2.scatter(tplot, dBeachVol_dt[cc, :], s=1, label='z = ' + str(cont_elev[cc]) + ' m')
    # ax2.scatter(tplot,total_dBeachVol_dt/total_obsBeachWid[1:len(lidartime)],s=1,color='k',label='Total dV/dt / Total obs. width')
    ax2.set_xlabel('time')
    ax2.set_ylabel('dV/dt [m^2/hr]')
    ax2.grid(which='major', axis='both')
    ax2.legend()
    plt.gcf().autofmt_xdate()


