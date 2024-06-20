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

def getlocal_lidarwavegauge(full_path):
    """

    :param full_path: consists of floc (path after .../FY24_SMARTSEED/FRF_data/) + filename
    :return: wg_xFRF, wave_time, wave_Hs, wave_HsIN, wave_HsIG, wave_Tp, wave_TmIG
    """

    import datetime as dt
    import numpy as np
    from netCDF4 import Dataset

    ## Wave dataset
    ds = Dataset(full_path, "r")
    wg_xFRF = ds.variables["xFRF"][:]
    wave_time = ds.variables["time"][:]
    wave_HsIG = ds.variables["waveHsIG"][:]
    wave_HsIN = ds.variables["waveHs"][:]
    wave_Hs = ds.variables["waveHsTotal"][:]
    wave_Tp = ds.variables["waveTp"][:]
    wave_TmIG = ds.variables["waveTmIG"][:]

    is_Hs_masked = np.ma.isMA(wave_Hs)
    if is_Hs_masked:
        wave_Tp = wave_Tp.filled(fill_value=np.NaN)
        wave_Hs = wave_Hs.filled(fill_value=np.NaN)
        wave_time = wave_time.filled(fill_value=np.NaN)
        wave_TmIG = wave_TmIG.filled(fill_value=np.NaN)
        wave_HsIN = wave_HsIN.filled(fill_value=np.NaN)
        wave_HsIG = wave_HsIG.filled(fill_value=np.NaN)
        wg_xFRF = wg_xFRF.filled(fill_value=np.NaN)

    return wg_xFRF, wave_time, wave_Hs, wave_HsIN, wave_HsIG, wave_Tp, wave_TmIG

def getlocal_lidarrunup(full_path):
    """

    :param full_path: consists of floc (path after .../FY24_SMARTSEED/FRF_data/) + filename
    :return: lidar_time, lidar_elev2p
    """

    import datetime as dt
    import numpy as np
    from netCDF4 import Dataset

    # Runup dataset
    ds = Dataset(full_path, "r")
    lidar_time = ds.variables["time"][:]
    lidar_elev2p = ds.variables["totalWaterLevel"][:]

    is_elev_masked = np.ma.isMA(lidar_elev2p)
    if is_elev_masked:
        lidar_time = lidar_time.filled(fill_value=np.NaN)
        lidar_elev2p = lidar_elev2p.filled(fill_value=np.NaN)


    return lidar_time, lidar_elev2p
