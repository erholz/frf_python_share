import numpy as np
from netCDF4 import Dataset

def getthredds_waterlevels(full_path):

    """
    :param full_path: consists of floc (path after .../thredds/dodsC/frf/) + filename
    :return: waterlevel_noaa, time_noaa
    """


    ## Get the date information from the input file name
    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"

    ## Water Level Dataset
    ds = Dataset(frf_base + full_path, "r")
    waterlevel_noaa = ds.variables["waterLevel"][:]
    time_noaa = np.asarray(ds.variables["time"][:])

    is_waterlevel_masked = np.ma.is_masked(waterlevel_noaa)
    if is_waterlevel_masked:
        waterlevel_noaa = waterlevel_noaa.filled(fill_value=np.NaN)


    return waterlevel_noaa, time_noaa


def getlocal_waterlevels(full_path):

    """
    :param :param: full_path - consists of floc (full path to file) + filename
    :return: waterlevel_noaa, time_noaa
    """

    ## Water Level Dataset
    ds = Dataset(full_path, "r")
    waterlevel_noaa = ds.variables["waterLevel"][:]
    time_noaa = np.asarray(ds.variables["time"][:])

    is_waterlevel_masked = np.ma.is_masked(waterlevel_noaa)
    if is_waterlevel_masked:
        waterlevel_noaa = waterlevel_noaa.filled(fill_value=np.NaN)

    return waterlevel_noaa, time_noaa