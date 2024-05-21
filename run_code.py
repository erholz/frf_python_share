import matplotlib
matplotlib.use("TKAgg")
from funcs.find_files import *
import numpy as np
import datetime as dt



# DEFINE WHERE FRF DATA FILES ARE LOCATED
local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'

# DEFINE TIME PERIOD OF INTEREST
time_beg = '2023-09-11T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2023-09-24T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
time_format = '%Y-%m-%dT%H:%M:%S'
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4

# DEFINE CONTOUR ELEVATIONS OF INTEREST
cont_elev = np.arange(-0.25,4.25,0.5)    # <<< MUST BE POSITIVELY INCREASING

# DEFINE NUMBER OF PROFILES TO PLOT
num_profs_plot = 15

# DEFINE SUBDIR WITH LIDAR FILES
lidarfloc = local_base + 'dune_lidar/lidar_transect/'
lidarext = 'nc'  # << change not recommended; defines file type to look for

# DEFINE SUBDIR WITH NOAA WATERLEVEL FILES
noaawlfloc = local_base + 'waterlevel/'
noaawlext = 'nc'  # << change not recommended; defines file type to look for

# DEFINE SUBDIR WITH LIDAR HYDRO FILES
lidarhydrofloc = local_base + 'waves_lidar/lidar_hydro/'
lidarhydroext = 'nc'  # << change not recommended; defines file type to look for




# -------------------- BEGIN RUN_CODE.PY --------------------


# convert period of interest to datenum
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)

# prep file collection
def find_files_in_range(floc,ext,epoch_beg,epoch_end):
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


# run file run_lidarcollect.py
exec(open('run_lidarcollect.py').read())

# Remove weird data
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
lidarelev[tmpii] = np.nan

# run file create_contours.py
exec(open('funcs/create_contours.py').read())

# run file run_hydrocollect.py
exec(open('run_hydrocollect.py').read())

# Run quality check script
# exec(open('lidar_check.py').read())

# Try filling gaps??
# exec(open('lidar_fillgaps.py').read())

# run file run_makeplots.py
# exec(open('run_makeplots.py').read())
from run_makeplots import *

plot_ContourTimeSeries()
plot_ProfilesSubset()
plot_ProfilesTimestack()