def getthredds_waves17m(full_path):

    """

    :param full_path: consists of floc (path after .../thredds/dodsC/frf/) + filename
    :return: wave_Tp, wave_Hs, wave_time, wave_depth, wave_dir
    """

    import datetime as dt
    import numpy as np
    from netCDF4 import Dataset

    ## Get the date information from the input file name
    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"
    mon_str = full_path[len(full_path) - 5:len(full_path) - 3]
    yr_str = full_path[len(full_path) - 9:len(full_path) - 5]

    ## Wave dataset
    # print(mon_str)
    # print(frf_base)
    ds = Dataset(frf_base + full_path, "r")
    wave_Tp = ds.variables["waveTp"][:]
    wave_Hs = ds.variables["waveHs"][:]
    wave_time = ds.variables["time"][:]
    wave_depth = ds.variables["gaugeDepth"][:]
    wave_dir = ds.variables["wavePeakDirectionPeakFrequency"][:]

    is_Hs_masked = np.ma.isMA(wave_Hs)
    if is_Hs_masked:
        wave_Tp = wave_Tp.filled(fill_value=np.NaN)
        wave_Hs = wave_Hs.filled(fill_value=np.NaN)
        wave_time = wave_time.filled(fill_value=np.NaN)
        wave_depth = wave_depth.filled(fill_value=np.NaN)
        wave_dir = wave_dir.filled(fill_value=np.NaN)


    return wave_Tp, wave_Hs, wave_time, wave_depth, wave_dir







def getlocal_waves17m(full_path):
    """

    :param full_path: consists of floc (path after .../FY24_SMARTSEED/FRF_data/) + filename
    :return: wave_Tp, wave_Hs, wave_time, wave_depth, wave_dir
    """

    import datetime as dt
    import numpy as np
    from netCDF4 import Dataset

    ## Get the date information from the input file name
    # local_base = 'D:/Projects/FY24/FY24_SMARTSEED/FRF_data/'
    # local_base = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/'
    local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'
    mon_str = full_path[len(full_path) - 5:len(full_path) - 3]
    yr_str = full_path[len(full_path) - 9:len(full_path) - 5]

    ## Wave dataset
    ds = Dataset(local_base + full_path, "r")
    wave_Tp = ds.variables["waveTp"][:]
    wave_Hs = ds.variables["waveHs"][:]
    wave_time = ds.variables["time"][:]
    wave_depth = ds.variables["gaugeDepth"][:]
    wave_dir = ds.variables["wavePeakDirectionPeakFrequency"][:]


    is_Hs_masked = np.ma.isMA(wave_Hs)
    if is_Hs_masked:
        wave_Tp = wave_Tp.filled(fill_value=np.NaN)
        wave_Hs = wave_Hs.filled(fill_value=np.NaN)
        wave_time = wave_time.filled(fill_value=np.NaN)
        wave_depth = wave_depth.filled(fill_value=np.NaN)
        wave_dir = wave_dir.filled(fill_value=np.NaN)

    return wave_Tp, wave_Hs, wave_time, wave_depth, wave_dir
