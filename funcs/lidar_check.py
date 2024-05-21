import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from funcs.create_contours import lidar_xFRF,lidarelev,time_beg,time_end,cont_elev,cmean,cstd,lidartime


# Availability of "quality" data (no-nans) as a func. of xFRF
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
cmap = plt.cm.rainbow(np.linspace(0,1,cont_elev.size))
ax.set_prop_cycle('color', cmap)
for cc in np.arange(cont_elev.size):
    plt.plot([0, 0] + cmean[cc], [0, 1])
for cc in np.arange(cont_elev.size):
    left, bottom, width, height = (cmean[cc] - cstd[cc], 0, cstd[cc] * 2, 1)
    patch = plt.Rectangle((left, bottom), width, height, alpha=0.1, color=cmap[cc, :])
    ax.add_patch(patch)

# Availability of "quality" data (no-nans) as a func. of time
fig, ax = plt.subplots()
tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
yplot = np.sum(~np.isnan(lidarelev), axis=1) / np.array(lidarelev.shape)[1]
ax.scatter(tplot, yplot, s=1)
plt.xlabel('time')
plt.ylabel('fraction of data "passing" qaqc over profile [0-1]')
plt.title(time_beg + ' to ' + time_end)
plt.gcf().autofmt_xdate()

# Ok, for each day in available data, find standard dev and num_profs as a function of x
daily_zstdev = np.empty(shape=lidarelev.shape)      # matrix of 24-hr moving statistic (z_stdev)
daily_zstdev[:] = np.nan
daily_znum = np.empty(shape=lidarelev.shape)        # matrix of 24-hr moving statistic (num profs)
daily_znum[:] = np.nan
for tt in range(len(lidartime)-1):
    proftt_notnan = ~np.isnan(lidarelev[tt,:])
    if tt < 13:
        dailygrab = np.arange(0,tt+1)
        numgrab = len(dailygrab)
        dailygrab = np.append(dailygrab, np.arange(tt+1,tt+24-numgrab+2))
    elif tt > len(lidartime)-13:
        dailygrab = np.arange(tt,len(lidartime))
        numgrab = len(dailygrab)
        dailygrab = np.append(np.arange(tt-(24-numgrab)-1,tt),dailygrab)
    else:
        dailygrab = np.arange(tt-12,tt+12+1)
    tmp_stdev = np.nanstd(lidarelev[dailygrab,:],axis=0)
    dum_profs = ~np.isnan(lidarelev[dailygrab,:])
    tmp_num = sum(dum_profs)
    daily_zstdev[tt,proftt_notnan] = tmp_stdev[proftt_notnan]
    daily_znum[tt, proftt_notnan] = tmp_num[proftt_notnan]
# ok now plot quality as a function of time and space...
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
ax1.set_ylim(80,185)
vplot = np.rot90(daily_znum/25)
# vplot[vplot < 0.3] = np.nan
ph2 = ax2.pcolormesh(tplot, yplot, vplot)
cbar2 = fig.colorbar(ph2,ax=ax2)
cbar2.set_label('frac avail')
ax2.set_ylabel('xFRF [m]')
ax2.set_ylim(80,185)
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
ax3.set_ylim(80,185)

# ax1.set_xlim(dt.datetime(2023,12,1),dt.datetime(2023,12,10))
# ax2.set_xlim(dt.datetime(2023,12,1),dt.datetime(2023,12,10))
# ax3.set_xlim(dt.datetime(2023,12,1),dt.datetime(2023,12,10))
# ax1.set_ylim(80,130)
# ax2.set_ylim(80,130)
# ax3.set_ylim(80,130)



