import matplotlib
matplotlib.use("TKAgg")
import numpy as np
import datetime as dt


# DEFINE WHERE FRF DATA FILES ARE LOCATED
local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'

# DEFINE TIME PERIOD OF INTEREST
time_beg = '2023-09-11T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2023-09-24T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4

# DEFINE CONTOUR ELEVATIONS OF INTEREST
cont_elev = np.arange(-0.25,4.25,0.75)    # <<< MUST BE POSITIVELY INCREASING

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
time_format = '%Y-%m-%dT%H:%M:%S'
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)

# run file run_lidarcollect.py
exec(open('run_lidarcollect.py').read())

# Remove weird data (first order filtering)
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
lidarelev[tmpii] = np.nan

# run file create_contours.py
exec(open('funcs/create_contours.py').read())

# run file run_hydrocollect.py
exec(open('run_hydrocollect.py').read())

# Run quality check script
exec(open('funcs/lidar_check.py').read())

# Run beach volume calculation script -- MUST RUN create_contours.py FIRST!
exec(open('funcs/calculate_beachvol.py').read())

# make some plots
from run_makeplots import *
plot_QualityDataTimeSeries()
plot_QualityDataWithContourPositions()
plot_DailyVariationTimestack()
plot_ContourTimeSeries()
plot_ProfilesSubset()
plot_ProfilesTimestack()
plot_BeachVolume()


# Try filling gaps??
exec(open('funcs/lidar_fillgaps.py').read())


# Re-run file create_contours.py
exec(open('funcs/create_contours.py').read())

# Re-run beach volume calculation script
exec(open('funcs/calculate_beachvol.py').read())

# Re-run some plots
plot_ContourTimeSeries()
plot_BeachVolume()
