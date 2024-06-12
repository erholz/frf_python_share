import matplotlib
matplotlib.use("TKAgg")
from funcs.getFRF_funcs.getFRF_waterlevels import getlocal_waterlevels
from funcs.find_files import find_files_in_range
import numpy as np
from netCDF4 import Dataset
from funcs.get_timeinfo import get_TimeInfo

def run_hydrocollect_func(noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext):

    # Get timing info from run_code.py
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    # # start with NOAA water level files
    # floc = noaawlfloc
    # ext = noaawlext
    # fname_in_range = find_files_in_range(floc,ext,epoch_beg,epoch_end, tzinfo)
    # wltime_noaa = []
    # wl_noaa = []
    # for fname_ii in fname_in_range:
    #     print('reading... ' + fname_ii)
    #     full_path = floc + fname_ii
    #     waterlevel_noaa, time_noaa = getlocal_waterlevels(full_path)
    #     wltime_noaa = np.append(wltime_noaa, time_noaa)
    #     wl_noaa = np.append(wl_noaa, waterlevel_noaa)
    # # Trim full data set to just the obs of interest
    # ij_in_range = (wltime_noaa >= epoch_beg) & (wltime_noaa <= epoch_end)
    # wltime_noaa = wltime_noaa[ij_in_range]
    # wl_noaa = wl_noaa[ij_in_range]


    # ok, now get water levels from lidarHydro files
    floc = lidarhydrofloc
    ext = lidarhydroext
    fname_in_range = find_files_in_range(floc,ext,epoch_beg,epoch_end,tzinfo)
    wltime_lidar = []
    wlmin_lidar = []
    wlmax_lidar = []
    wlmean_lidar = []
    for fname_ii in fname_in_range:
        fullpath = floc + fname_ii
        ds = Dataset(fullpath, "r")
        minWL = ds.variables["minWaterLevel"][:]
        maxWL = ds.variables["maxWaterLevel"][:]
        meanWL = ds.variables["waterLevel"][:]
        time = ds.variables["time"][:]
        wltime_lidar = np.append(wltime_lidar, time)
        if len(wlmax_lidar) < 1:    # if matrix is empty, then initialize
            wlmax_lidar = maxWL
            wlmin_lidar = minWL
            wlmean_lidar = meanWL
        else:
            wlmax_lidar = np.append(wlmax_lidar, maxWL, axis=0)
            wlmin_lidar = np.append(wlmin_lidar, minWL, axis=0)
            wlmean_lidar = np.append(wlmean_lidar, meanWL, axis=0)
    # Trim full data set to just the obs of interest
    ij_in_range = (wltime_lidar >= epoch_beg) & (wltime_lidar <= epoch_end)
    wltime_lidar = wltime_lidar[ij_in_range]
    wlmax_lidar = wlmax_lidar[ij_in_range]
    wlmin_lidar = wlmin_lidar[ij_in_range]
    wlmean_lidar = wlmean_lidar[ij_in_range]
    wlmin_lidar[wlmin_lidar < -99] = np.nan
    wlmax_lidar[wlmax_lidar < -99] = np.nan
    wlmean_lidar[wlmean_lidar < -99] = np.nan

    return wlmax_lidar,wlmin_lidar,wltime_lidar,wlmean_lidar
