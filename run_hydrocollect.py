import matplotlib
matplotlib.use("TKAgg")
from funcs.getFRF_funcs.getFRF_waterlevels import getlocal_waterlevels
from funcs.find_files import find_files_in_range
import numpy as np
from netCDF4 import Dataset
from funcs.get_timeinfo import get_TimeInfo
from funcs.getFRF_funcs.getFRF_waves import *
from funcs.getFRF_funcs.getFRF_lidar import *

def run_hydrocollect_func(noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext):

    # Get timing info from run_code.py
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    # start with NOAA water level files
    floc = noaawlfloc
    ext = noaawlext
    fname_in_range = find_files_in_range(floc,ext,epoch_beg,epoch_end, tzinfo)
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

    return wlmax_lidar,wlmin_lidar,wltime_lidar,wlmean_lidar, wl_noaa, wltime_noaa


def run_wavecollect17m_func(wave17mfloc, wave17mext):

    # Get timing info from run_code.py
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    floc = wave17mfloc
    ext = wave17mext
    fname_in_range = find_files_in_range(floc, ext, epoch_beg, epoch_end, tzinfo)
    wave17m_time = []
    wave17m_Tp = []
    wave17m_Hs = []
    wave17m_dir = []
    for fname_ii in fname_in_range:
        full_path = floc + fname_ii
        wave_Tp, wave_Hs, wave_time, wave_dir = getlocal_waves17m(full_path)
        wave17m_time = np.append(wave17m_time, wave_time, axis=0)
        wave17m_Tp = np.append(wave17m_Tp, wave_Tp, axis=0)
        wave17m_Hs = np.append(wave17m_Hs, wave_Hs, axis=0)
        wave17m_dir = np.append(wave17m_dir, wave_dir, axis=0)
    # Trim full data set to just the obs of interest
    ij_in_range = (wave17m_time >= epoch_beg) & (wave17m_time <= epoch_end)
    wave17m_time = wave17m_time[ij_in_range]
    wave17m_Tp = wave17m_Tp[ij_in_range]
    wave17m_Hs = wave17m_Hs[ij_in_range]
    wave17m_dir = wave17m_dir[ij_in_range]
    wave17m_Tp[wave17m_Tp < -99] = np.nan
    wave17m_Hs[wave17m_Hs < -99] = np.nan
    wave17m_dir[wave17m_dir < -99] = np.nan

    return wave17m_time,wave17m_Tp,wave17m_Hs,wave17m_dir

def run_wavecollect8m_func(wave8mfloc, wave8mext):

    # Get timing info from run_code.py
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    floc = wave8mfloc
    ext = wave8mext
    fname_in_range = find_files_in_range(floc, ext, epoch_beg, epoch_end, tzinfo)
    wave8m_time = []
    wave8m_Tp = []
    wave8m_Hs = []
    wave8m_dir = []

    for fname_ii in fname_in_range:
        full_path = floc + fname_ii
        qaqc_fac, wave_peakdir, wave_Tp, wave_Hs, wave_time, src_WL, wave_WL = getlocal_waves8m(full_path)
        wave8m_time = np.append(wave8m_time, wave_time, axis=0)
        wave8m_Tp = np.append(wave8m_Tp, wave_Tp, axis=0)
        wave8m_Hs = np.append(wave8m_Hs, wave_Hs, axis=0)
        wave8m_dir = np.append(wave8m_dir, wave_peakdir, axis=0)
    # Trim full data set to just the obs of interest
    ij_in_range = (wave8m_time >= epoch_beg) & (wave8m_time <= epoch_end)
    wave8m_time = wave8m_time[ij_in_range]
    wave8m_Tp = wave8m_Tp[ij_in_range]
    wave8m_Hs = wave8m_Hs[ij_in_range]
    wave8m_dir = wave8m_dir[ij_in_range]
    wave8m_Tp[wave8m_Tp < -99] = np.nan
    wave8m_Hs[wave8m_Hs < -99] = np.nan
    wave8m_dir[wave8m_dir < -99] = np.nan

    return wave8m_time,wave8m_Tp,wave8m_Hs,wave8m_dir


def run_lidarwavegauge_func(lidarwavegaugefloc, lidarwavegaugeext):

    # Get timing info from run_code.py
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    floc = lidarwavegaugefloc
    ext = lidarwavegaugeext
    fname_in_range = find_files_in_range(floc, ext, epoch_beg, epoch_end, tzinfo)
    lidarwg_xFRF = []
    lidarwg_time = []
    lidarwg_Tp = []
    lidarwg_TmIG = []
    lidarwg_Hs = []
    lidarwg_HsIN = []
    lidarwg_HsIG = []
    for fname_ii in fname_in_range:
        full_path = floc + fname_ii
        wg_xFRF, wave_time, wave_Hs, wave_HsIN, wave_HsIG, wave_Tp, wave_TmIG = getlocal_lidarwavegauge(full_path)
        lidarwg_xFRF = wg_xFRF
        lidarwg_time = np.append(lidarwg_time, wave_time, axis=0)
        lidarwg_Tp = np.append(lidarwg_Tp, wave_Tp, axis=0)
        lidarwg_TmIG = np.append(lidarwg_TmIG, wave_TmIG, axis=0)
        lidarwg_Hs = np.append(lidarwg_Hs, wave_Hs, axis=0)
        lidarwg_HsIN = np.append(lidarwg_HsIN, wave_HsIN, axis=0)
        lidarwg_HsIG = np.append(lidarwg_HsIG, wave_HsIG, axis=0)
    # Trim full data set to just the obs of interest
    ij_in_range = (lidarwg_time >= epoch_beg) & (lidarwg_time <= epoch_end)
    lidarwg_time = lidarwg_time[ij_in_range]
    lidarwg_Tp = lidarwg_Tp[ij_in_range]
    lidarwg_TmIG = lidarwg_TmIG[ij_in_range]
    lidarwg_Hs = lidarwg_Hs[ij_in_range]
    lidarwg_HsIN = lidarwg_HsIN[ij_in_range]
    lidarwg_HsIG = lidarwg_HsIG[ij_in_range]
    lidarwg_Tp[lidarwg_Tp < -99] = np.nan
    lidarwg_TmIG[lidarwg_TmIG < -99] = np.nan
    lidarwg_Hs[lidarwg_Hs < -99] = np.nan
    lidarwg_HsIN[lidarwg_HsIN < -99] = np.nan
    lidarwg_HsIG[lidarwg_HsIG < -99] = np.nan

    return lidarwg_xFRF,lidarwg_time,lidarwg_Tp,lidarwg_TmIG,lidarwg_Hs,lidarwg_HsIN,lidarwg_HsIG


def run_lidarrunup_func(lidarrunupfloc, lidarrunupext):

    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    floc = lidarrunupfloc
    ext = lidarrunupext
    fname_in_range = find_files_in_range(floc, ext, epoch_beg, epoch_end, tzinfo)
    lidarrunup_time = []
    lidarrunup_elev2p = []
    for fname_ii in fname_in_range:
        full_path = floc + fname_ii
        lidar_time, lidar_elev2p = getlocal_lidarrunup(full_path)
        lidarrunup_time = np.append(lidarrunup_time, lidar_time, axis=0)
        lidarrunup_elev2p = np.append(lidarrunup_elev2p, lidar_elev2p, axis=0)
    # Trim full data set to just the obs of interest
    ij_in_range = (lidarrunup_time >= epoch_beg) & (lidarrunup_time <= epoch_end)
    lidarrunup_time = lidarrunup_time[ij_in_range]
    lidarrunup_elev2p = lidarrunup_elev2p[ij_in_range]
    lidarrunup_time[lidarrunup_time < -99] = np.nan
    lidarrunup_elev2p[lidarrunup_elev2p < -99] = np.nan

    return lidarrunup_time,lidarrunup_elev2p