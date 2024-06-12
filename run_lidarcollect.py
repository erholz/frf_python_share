from funcs.getFRF_funcs.getFRF_lidar import getlocal_lidar
from funcs.find_files import find_files_in_range, find_files_local
import numpy as np
from funcs.get_timeinfo import get_TimeInfo


def run_lidarcollect(lidarfloc, lidarext):
    floc = lidarfloc
    ext = lidarext

    # Get timing info from run_code.py
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = get_TimeInfo()

    # Get the data names of the LIDAR files...
    fname_in_range = find_files_in_range(floc, ext, epoch_beg, epoch_end, tzinfo)

    if np.size(fname_in_range) == 0:
        lidarelev = []
        lidartime = []
        lidar_xFRF = []
        lidarelevstd = []
        lidarmissing = []
        return lidarelev, lidartime, lidar_xFRF, lidarelevstd, lidarmissing

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
        if len(lidarelev) < 1:  # if lidarelev is empty, then initialize
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

    return lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing