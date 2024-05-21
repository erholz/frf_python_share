import matplotlib
matplotlib.use("TKAgg")
from funcs.getFRF_funcs.getFRF_lidar import *
from run_code import find_files_in_range, lidarfloc, lidarext, epoch_end, epoch_beg


floc = lidarfloc
ext = lidarext


# Get the data names of the LIDAR files...
fname_in_range = find_files_in_range(floc,ext,epoch_beg,epoch_end)

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
