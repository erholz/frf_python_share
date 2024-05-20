import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from find_files import *
from getFRF_lidar import *
from getFRF_waves import *
from getFRF_waterlevels import *
from getFRF_17mWaverider import *
from wavefuncs import *
import numpy as np
import scipy as sp
import datetime as dt

# local_base = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/'
local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'

# Define the time you want to access
time_beg = '2023-09-11T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2023-09-24T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
time_format = '%Y-%m-%dT%H:%M:%S'
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4

# convert to datenum
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)

def find_files_in_range():
    list_of_files = np.array(find_files_local(floc, ext))
    fdate = []
    for fname_ii in list_of_files:
        fdate = np.append(fdate, int(fname_ii[-9:-3]))
    sorted_list = list_of_files[np.argsort(fdate)]
    sorted_fdate = fdate[np.argsort(fdate)]
    # Find names within the time of interest
    fname_epoch = []
    for datejj in sorted_fdate:
        datejj_str = str(datejj)
        tmp_epoch = dt.datetime.strptime(datejj_str[0:6], '%Y%m').timestamp()
        fname_epoch.append(tmp_epoch)
    fname_epoch = np.array(fname_epoch)
    ij_in_range = (fname_epoch >= epoch_beg) & (fname_epoch <= epoch_end)
    if sum(ij_in_range) == 0:
        # check if desired [year/month] has match in sorted_fdate
        begnum = int(dt.datetime.fromtimestamp(epoch_beg,tzinfo).strftime('%Y%m'))
        endnum = int(dt.datetime.fromtimestamp(epoch_end,tzinfo).strftime('%Y%m'))
        ij_in_range = (begnum == sorted_fdate ) | (endnum == sorted_fdate)
        if sum(ij_in_range) == 0:
            print('No data to see here, folks')
            print('Try another data range')
            exit()
    if min(np.argwhere(ij_in_range)) > 0:
        ij_in_range[min(np.argwhere(ij_in_range))-1] = True
    if max(np.argwhere(ij_in_range)) < ij_in_range.size - 1:
        ij_in_range[max(np.argwhere(ij_in_range)) + 1] = True
    fname_in_range = sorted_list[ij_in_range]
    return fname_in_range

# Get the data names of the LIDAR files...
floc = 'dune_lidar/lidar_transect/'
ext = 'nc'
fname_in_range = find_files_in_range()

# Process the LIDAR files within range of interest
lidartime = []
lidarelev = []
lidarelevstd = []
lidarqaqc = []
lidarmissing = []
for fname_ii in fname_in_range:
    print('reading... ' + fname_ii)
    full_path = floc + fname_ii
    qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF = (
        getlocal_lidar(full_path))
    lidartime = np.append(lidartime, lidar_time)
    lidarqaqc = np.append(lidarqaqc, qaqc_fac)
    if len(lidarelev) < 1:    # if lidarelev is empty, then initialize
        lidarelev = lidar_elev
        lidarelevstd = lidar_elevstd
        lidarmissing = lidar_pmissing
    else:
        lidarelev = np.append(lidarelev, lidar_elev, axis=0)
        lidarelevstd = np.append(lidarelevstd, lidar_elevstd, axis=0)
        lidarmissing = np.append(lidarmissing, lidar_pmissing, axis=0)
if len(lidarelev) == 0:
    print('STOP WHAT YOU''RE DOING')
    print('THERE IS NO DATA IN DESIRED TIMESPAN')
    exit()

# Trim full data set to just the obs of interest
ij_in_range = (lidartime >= epoch_beg) & (lidartime <= epoch_end)
lidartime = lidartime[ij_in_range]
lidarelev = lidarelev[ij_in_range]
lidarelevstd = lidarelevstd[ij_in_range]
lidarqaqc = lidarqaqc[ij_in_range]
lidarmissing = lidarmissing[ij_in_range]

# Remove weird data
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
lidarelev[tmpii] = np.nan

# Calculate contour time-series of interest
# CONTOURS MUST POSITIVELY INCREASE IN ELEVATION
# cont_elev = np.concatenate(([-0.25], np.arange(0,4.5,0.5)))  # <<< USER DEFINED
cont_elev = np.arange(-0.25,4.25,0.25)    # <<< USER DEFINED
cont_ts = np.empty((cont_elev.size,lidartime.size))
for tt in np.arange(lidartime.size):
    xtmp = lidar_xFRF[~np.isnan(lidarelev[tt,:])]
    ztmp = lidarelev[tt,~np.isnan(lidarelev[tt,:])]
    xcc = np.empty(shape=cont_elev.shape)
    xcc[:] = np.nan
    for cc in np.arange(cont_elev.size):
        if len(ztmp) > 0:
            if (min(ztmp) <= cont_elev[cc]) & (max(ztmp) >= cont_elev[cc]):
                iiclose = int(np.argwhere(abs(ztmp - cont_elev[cc]) == min(abs(ztmp - cont_elev[cc]))))
                xcc[cc] = np.interp(0,ztmp[iiclose-1:iiclose+2],xtmp[iiclose-1:iiclose+2])
    cont_ts[:, tt] = xcc

# Calculate mean and stddev of contour x-loc
cmean = np.nanmean(cont_ts,axis=1)
cstd = np.nanstd(cont_ts,axis=1)

# Get water level files in range
floc = 'waterlevel/'
ext = 'nc'
fname_in_range = find_files_in_range()

# Process water level files
wltime_noaa = []
wl_noaa = []
for fname_ii in fname_in_range:
    print('reading... ' + fname_ii)
    full_path = floc + fname_ii
    waterlevel_noaa, time_noaa = getlocal_waterlevels(full_path)
    wltime_noaa = np.append(wltime_noaa, time_noaa)
    wl_noaa = np.append(wl_noaa, waterlevel_noaa)
# Trim full data set to just the obs of interest
ij_in_range = (wltime_noaa >= epoch_beg) & (wltime_noaa <= epoch_end)
wltime_noaa = wltime_noaa[ij_in_range]
wl_noaa = wl_noaa[ij_in_range]


# Get water level from LIDAR
floc = 'waves_lidar/lidar_hydro/'
ext = 'nc'
fname_in_range = find_files_in_range()
# Process water level files
wltime_lidar = []
wlmin_lidar = []
wlmax_lidar = []
for fname_ii in fname_in_range:
    fullpath = local_base + floc + fname_ii
    ds = Dataset(fullpath, "r")
    minWL = ds.variables["minWaterLevel"][:]
    maxWL = ds.variables["maxWaterLevel"][:]
    time = ds.variables["time"][:]
    wltime_lidar = np.append(wltime_lidar, time)
    if len(wlmax_lidar) < 1:    # if matrix is empty, then initialize
        wlmax_lidar = maxWL
        wlmin_lidar = minWL
    else:
        wlmax_lidar = np.append(wlmax_lidar, maxWL, axis=0)
        wlmin_lidar = np.append(wlmin_lidar, minWL, axis=0)
# Trim full data set to just the obs of interest
ij_in_range = (wltime_lidar >= epoch_beg) & (wltime_lidar <= epoch_end)
wltime_lidar = wltime_lidar[ij_in_range]
wlmax_lidar = wlmax_lidar[ij_in_range]
wlmin_lidar = wlmin_lidar[ij_in_range]
wlmin_lidar[wlmin_lidar < -99] = np.nan
wlmax_lidar[wlmax_lidar < -99] = np.nan

# Plot contour time series
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


# Make plots of profiles through time
numprofs = np.array(lidarelev.shape)[0]
print('hihi')
print('There are ' + str(numprofs) + ' profiles in this subset')
print('How many do you want to plot? --> Set [numplot]')
# numplot = numprofs
numplot = 20
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

# Plot all profiles as timestacks
fig, ax = plt.subplots()
# ph = ax.pcolormesh(tplot, lidar_xFRF, np.rot90(lidarelev,k=-1))
XX,TT = np.meshgrid(lidar_xFRF,tplot)
ph = ax.scatter(np.reshape(TT,TT.size),np.reshape(XX,XX.size),s=1,c=np.reshape(lidarelev,lidarelev.size))
cbar1 = fig.colorbar(ph,ax=ax)
cbar1.set_label('z [m]')
xnotnan = lidar_xFRF[np.sum(np.isnan(lidarelev),axis=0) != len(lidartime)]
ax.set_ylim(np.nanmin(xnotnan),np.nanmax(xnotnan))
plt.gcf().autofmt_xdate()
ax.set_xlim(min(tplot),max(tplot))

# Run quality check script
# exec(open('lidar_check.py').read())

# Try filling gaps??
# exec(open('lidar_fillgaps.py').read())




