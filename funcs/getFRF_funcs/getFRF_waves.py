def getthredds_waves(full_path):

    """

    :param full_path: consists of floc (path after .../thredds/dodsC/frf/) + filename
    :return: qaqc_fac, wave_peakdir, wave_Tp, wave_Hs, wave_time, src_WL, wave_WL
    """

    import datetime as dt
    import numpy as np
    from netCDF4 import Dataset

    ## Get the date information from the input file name
    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"
    mon_str = full_path[len(full_path)-5:len(full_path)-3]
    yr_str = full_path[len(full_path)-9:len(full_path)-5]

    ## Wave dataset
    # print(mon_str)
    # print(frf_base)
    ds = Dataset(frf_base + full_path, "r")
    qaqc_fac = ds.variables["qcFlagE"][:]
    wave_peakdir = ds.variables["wavePrincipleDirection"][:]
    wave_Tp = ds.variables["waveTp"][:]
    wave_Hs = ds.variables["waveHs"][:]
    wave_time = ds.variables["time"][:]


    ## Water Level Dataset
    src_WL = [None]
    wave_WL = [None]*len(wave_time)

    try:
        ## Try EOP
        ds2 = Dataset(frf_base
                      + "oceanography/waterlevel/eopNoaaTide/"
                      + yr_str
                      + "/FRF-ocean_waterlevel_eopNoaaTide_"
                      + yr_str
                      + mon_str
                      + ".nc",
                      "r",
                      )
        waterlevel = ds2.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds2.variables["time"][:])
        #print("Water level sourced from EOP")
        src_WL = 1
    except:
        ## If no EOP, grab from 8m array
        waterlevel = ds.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds.variables["time"][:])
        # print("Water level sourced from 8m array")
        src_WL = 0


    for tt in range(len(wave_time)):
        ind_WL = np.abs(thredds_time_WL - wave_time[tt]).argmin()

        is_waterlevel_masked = np.ma.is_masked(waterlevel)
        if is_waterlevel_masked:
            waterlevel = waterlevel.filled(fill_value=np.NaN)
            wave_peakdir = wave_peakdir.filled(fill_value=np.NaN)
            wave_Tp = wave_Tp.filled(fill_value=np.NaN)
            wave_Hs = wave_Hs.filled(fill_value=np.NaN)
            wave_time = wave_time.filled(fill_value=np.NaN)
            # qaqc_fac = qaqc_fac.filled(fill_value=np.NaN)
        if np.isnan(waterlevel[ind_WL]):
            wave_WL[tt] = np.NaN
        else:
            wave_WL[tt] = round(waterlevel[ind_WL], 2)

    return qaqc_fac, wave_peakdir, wave_Tp, wave_Hs, wave_time, src_WL, wave_WL




def getlocal_waves(full_path):

    """

    :param full_path: consists of floc (path after .../FY24_SMARTSEED/FRF_data/) + filename
    :return: qaqc_fac, wave_peakdir, wave_Tp, wave_Hs, wave_time, src_WL, wave_WL
    """

    import datetime as dt
    import numpy as np
    from netCDF4 import Dataset

    ## Get the date information from the input file name
    # local_base = 'D:/Projects/FY24/FY24_SMARTSEED/FRF_data/'
    # local_base = 'F:/Projects/FY24/FY24_SMARTSEED/FRF_data/'
    local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'
    mon_str = full_path[len(full_path)-5:len(full_path)-3]
    yr_str = full_path[len(full_path)-9:len(full_path)-5]

    ## Wave dataset
    ds = Dataset(local_base + full_path, "r")
    qaqc_fac = ds.variables["qcFlagE"][:]
    wave_peakdir = ds.variables["wavePrincipleDirection"][:]
    wave_Tp = ds.variables["waveTp"][:]
    wave_Hs = ds.variables["waveHs"][:]
    wave_time = ds.variables["time"][:]

    ## Water Level Dataset
    src_WL = [None]
    wave_WL = [None]*len(wave_time)

    try:
        ## Try EOP
        ds2 = Dataset(local_base
                      + "/waterlevel/"
                      + "/FRF-ocean_waterlevel_eopNoaaTide_"
                      + yr_str
                      + mon_str
                      + ".nc",
                      "r",
                      )
        waterlevel = ds2.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds2.variables["time"][:])
        #print("Water level sourced from EOP")
        src_WL = 1
    except:
        ## If no EOP, grab from 8m array
        waterlevel = ds.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds.variables["time"][:])
        # print("Water level sourced from 8m array")
        src_WL = 0


    for tt in range(len(wave_time)):
        ind_WL = np.abs(thredds_time_WL - wave_time[tt]).argmin()

        is_Hs_masked = np.ma.isMA(wave_Hs)
        if is_Hs_masked:
            waterlevel = waterlevel.filled(fill_value=np.NaN)
            wave_peakdir = wave_peakdir.filled(fill_value=np.NaN)
            wave_Tp = wave_Tp.filled(fill_value=np.NaN)
            wave_Hs = wave_Hs.filled(fill_value=np.NaN)
            wave_time = wave_time.filled(fill_value=np.NaN)
            # qaqc_fac = qaqc_fac.filled(fill_value=np.NaN)
        if np.isnan(waterlevel[ind_WL]):
            wave_WL[tt] = np.NaN
        else:
            wave_WL[tt] = round(waterlevel[ind_WL], 2)

    return qaqc_fac, wave_peakdir, wave_Tp, wave_Hs, wave_time, src_WL, wave_WL