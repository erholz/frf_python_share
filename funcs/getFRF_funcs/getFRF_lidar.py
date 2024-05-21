import datetime as dt
import numpy as np
from netCDF4 import Dataset


def getthredds_lidar(full_path):

    """
    :param full_path: consists of floc (path after .../thredds/dodsC/frf/) + filename
    :return: qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF
    """

    ## Get the date information from the input file name
    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"

    ## Lidar dataset
    ds = Dataset(frf_base + full_path, "r")
    qaqc_fac = ds.variables["beachProfileQCFlag"][:]
    lidar_pmissing = ds.variables["percentTimeSeriesMissing"][:, :]
    lidar_elev = ds.variables["elevation"][:, :]
    lidar_elevstd = ds.variables["elevationSigma"][:]
    lidar_time = ds.variables["time"][:]
    lidar_xFRF = ds.variables["xFRF"][:]
    lidar_yFRF = ds.variables["yFRF"][:]

    ## Remove masked values, fill as NaNs
    lidar_pmissing = lidar_pmissing.filled(fill_value=np.NaN)
    lidar_elev = lidar_elev.filled(fill_value=np.NaN)
    lidar_elevstd = lidar_elevstd.filled(fill_value=np.NaN)
    lidar_time = lidar_time.filled(fill_value=np.NaN)
    lidar_xFRF = np.array(lidar_xFRF)
    lidar_yFRF = np.array(lidar_yFRF)

    return qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF



def getlocal_lidar(full_path):

    """
    :param: full_path - consists of floc (full path to file) + filename
    :return: qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF
    """

    ## Lidar dataset
    ds = Dataset(full_path, "r")
    qaqc_fac = ds.variables["beachProfileQCFlag"][:]
    lidar_pmissing = ds.variables["percentTimeSeriesMissing"][:, :]
    lidar_elev = ds.variables["elevation"][:, :]
    lidar_elevstd = ds.variables["elevationSigma"][:]
    lidar_time = ds.variables["time"][:]
    lidar_xFRF = ds.variables["xFRF"][:]
    lidar_yFRF = ds.variables["yFRF"][:]

    ## Remove masked values, fill as NaNs
    lidar_pmissing = lidar_pmissing.filled(fill_value=np.NaN)
    lidar_elev = lidar_elev.filled(fill_value=np.NaN)
    lidar_elevstd = lidar_elevstd.filled(fill_value=np.NaN)
    lidar_time = lidar_time.filled(fill_value=np.NaN)
    lidar_xFRF = np.array(lidar_xFRF)
    lidar_yFRF = np.array(lidar_yFRF)

    return qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF
