import numpy as np
import datetime as dt
from run_lidarcollect import *
from run_hydrocollect import *
from funcs.create_contours import *
from funcs.lidar_check import *
from funcs.calculate_beachvol import *
from funcs.lidar_fillgaps import *
from run_makeplots import *

# DEFINE WHERE FRF DATA FILES ARE LOCATED
local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'
local_base = '/volumes/macDrive/FRF_data/'

# DEFINE TIME PERIOD OF INTEREST
time_beg = '2023-09-11T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2023-09-24T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
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
time_format = '%Y-%m-%dT%H:%M:%S'
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)

# run file run_lidarcollect.py
lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext, epoch_end, epoch_beg, tzinfo)

# Remove weird data (first order filtering)
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
lidarelev[tmpii] = np.nan

# run file create_contours.py
elev_input = lidarelev
cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)

# run file run_hydrocollect.py
wlmax_lidar,wlmin_lidar,wltime_lidar = run_hydrocollect_func(noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext, epoch_end, epoch_beg, tzinfo)

# Run quality check script
# exec(open('funcs/lidar_check.py').read())
daily_zstdev,daily_znum = lidar_check(elev_input,lidartime)

# Run beach volume calculation script -- MUST RUN create_contours.py FIRST!
beachVol, beachVol_xc, dBeachVol_dt = calculate_beachvol(elev_input,lidartime,lidar_xFRF,cont_elev,cont_ts)

# make some plots
plot_ContourTimeSeries(cont_elev,cont_ts,lidartime,wlmax_lidar,wlmin_lidar,wltime_lidar,time_beg,time_end)
plot_ProfilesSubset(elev_input,lidartime,lidar_xFRF,num_profs_plot,time_beg,time_end,tzinfo,TOI_duration)
plot_ProfilesTimestack(elev_input,lidartime,lidar_xFRF)
plot_QualityDataWithContourPositions(elev_input, lidar_xFRF, cont_elev, cmean, cstd, time_beg, time_end)
plot_QualityDataTimeSeries(elev_input,lidartime,time_beg,time_end)
plot_DailyVariationTimestack(elev_input,lidartime,lidar_xFRF,daily_zstdev,daily_znum)
plot_BeachVolume(lidartime,cont_elev,beachVol,dBeachVol_dt,time_beg,time_end)


# Try filling gaps??
halfspan_time = 4
halfspan_x = 5
lidar_filled = lidar_fillgaps(lidarelev,lidartime,lidar_xFRF,halfspan_time,halfspan_x)

# Plot the new lidar, filled in gaps
plot_PrefillPostfillTimestack(lidarelev,lidar_filled,lidartime,lidar_xFRF)

# Re-run file create_contours.py
elev_input = lidar_filled
cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)

# Re-run beach volume calculation script
beachVol, beachVol_xc, dBeachVol_dt = calculate_beachvol(elev_input,lidartime,lidar_xFRF,cont_elev,cont_ts)

# Re-run some plots
plot_ContourTimeSeries(cont_elev,cont_ts,lidartime,wlmax_lidar,wlmin_lidar,wltime_lidar,time_beg,time_end)
plot_BeachVolume(lidartime,cont_elev,beachVol,dBeachVol_dt,time_beg,time_end)
