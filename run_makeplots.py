import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from run_code import cont_elev,num_profs_plot,lidartime,lidarelev,cont_ts,time_end,time_beg,wltime_lidar,wlmax_lidar,wlmin_lidar,TOI_duration,lidar_xFRF,tzinfo



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
    print('hihi')
    print('There are ' + str(numprofs) + ' profiles in this subset')
    print('How many do you want to plot? --> Set [numplot]')
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