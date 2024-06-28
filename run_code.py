import numpy as np
import datetime as dt
from run_lidarcollect import *
from run_hydrocollect import *
from funcs.create_contours import *
from funcs.lidar_check import *
from funcs.calculate_beachvol import *
from funcs.lidar_fillgaps import *
from run_makeplots import *
import pickle

# DEFINE WHERE FRF DATA FILES ARE LOCATED
local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'

# DEFINE TIME PERIOD OF INTEREST
time_beg = '2015-10-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2025-01-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4

# DEFINE CONTOUR ELEVATIONS OF INTEREST
cont_elev = np.arange(0,2.5,0.5)    # <<< MUST BE POSITIVELY INCREASING

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


# -------------------- SAVE VARIABLES AS PICKLE --------------------

# convert period of interest to datenum
time_format = '%Y-%m-%dT%H:%M:%S'
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)
# Save timing variables
with open('timeinfo.pickle','wb') as file:
    pickle.dump([tzinfo,time_format,time_beg,time_end,epoch_beg,epoch_end,TOI_duration], file)

# Save file locations
with open('fileinfo.pickle','wb') as file:
    pickle.dump([local_base,lidarfloc,lidarext,noaawlfloc,noaawlext,lidarhydrofloc,lidarhydroext],file)


# -------------------- BEGIN RUN_CODE.PY --------------------

# run file run_lidarcollect.py
lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext)

# Remove weird data (first order filtering)
stdthresh = 0.05        # [m], e.g., 0.05 equals 5cm standard deviation in hrly reading
pmissthresh = 0.75      # [0-1]. e.g., 0.75 equals 75% time series missing
tmpii = (lidarelevstd >= stdthresh) + (lidarmissing > pmissthresh)
lidarelev[tmpii] = np.nan

# run file create_contours.py
elev_input = lidarelev
cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)

# run file run_hydrocollect.py
wlmax_lidar,wlmin_lidar,wltime_lidar,wlmean_lidar = run_hydrocollect_func(noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext)

# Run quality check script
# exec(open('funcs/lidar_check.py').read())
daily_zstdev,daily_znum = lidar_check(elev_input,lidartime)

# Run beach volume calculation script -- MUST RUN create_contours.py FIRST!
beachVol, beachVol_xc, dBeachVol_dt,total_beachVol, total_dBeachVol_dt, total_obsBeachWid = calculate_beachvol(elev_input,lidartime,lidar_xFRF,cont_elev,cont_ts)

# make some plots
fig, ax1, ax2 = plot_ContourTimeSeries(cont_elev,cont_ts,lidartime,wlmean_lidar,wltime_lidar,lidar_xFRF)
ax2.grid(which='minor',axis='x')
ax1.grid(which='minor',axis='x')
# fig, ax = plot_ProfilesSubset(elev_input,lidartime,lidar_xFRF,num_profs_plot)
# fig, ax = plot_ProfilesTimestack(elev_input,lidartime,lidar_xFRF)
# fig, ax = plot_QualityDataWithContourPositions(elev_input, lidar_xFRF, cont_elev, cmean, cstd)
# fig, ax = plot_QualityDataTimeSeries(elev_input,lidartime)
# fig, ax1,ax2,ax3 = plot_DailyVariationTimestack(elev_input,lidartime,lidar_xFRF,daily_zstdev,daily_znum)
# fig, ax1, ax2 = plot_BeachVolume(lidartime,cont_elev,beachVol,dBeachVol_dt)


# Try filling gaps??
halfspan_time = 6
halfspan_x = 10
lidar_filled = lidar_fillgaps(lidarelev,lidartime,lidar_xFRF,halfspan_time,halfspan_x)
#
# # Plot the new lidar, filled in gaps
fig1, ax1, fig2, ax2 = plot_PrefillPostfillTimestack(lidarelev,lidar_filled,lidartime,lidar_xFRF)
#
# # Re-run file create_contours.py
# # elev_input = lidar_filled
# cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)
#
# # Re-run beach volume calculation script
# beachVol, beachVol_xc, dBeachVol_dt = calculate_beachvol(elev_input,lidartime,lidar_xFRF,cont_elev,cont_ts)
#
# # Re-run some plots
# fig, ax1, ax2 = plot_ContourTimeSeries(cont_elev,cont_ts,lidartime,wlmax_lidar,wlmin_lidar,wltime_lidar)
# fig, ax1, ax2 = plot_BeachVolume(lidartime,cont_elev,beachVol,dBeachVol_dt)
