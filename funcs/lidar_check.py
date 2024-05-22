import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def lidar_check(elev_input,lidartime):
    # Ok, for each day in available data, find standard dev and num_profs as a function of x
    daily_zstdev = np.empty(shape=elev_input.shape)      # matrix of 24-hr moving statistic (z_stdev)
    daily_zstdev[:] = np.nan
    daily_znum = np.empty(shape=elev_input.shape)        # matrix of 24-hr moving statistic (num profs)
    daily_znum[:] = np.nan
    for tt in range(len(lidartime)-1):
        proftt_notnan = ~np.isnan(elev_input[tt,:])
        if tt < 13:
            dailygrab = np.arange(0,tt+1)
            numgrab = len(dailygrab)
            dailygrab = np.append(dailygrab, np.arange(tt+1,tt+24-numgrab+2))
        elif tt > len(lidartime)-13:
            dailygrab = np.arange(tt,len(lidartime))
            numgrab = len(dailygrab)
            dailygrab = np.append(np.arange(tt-(24-numgrab)-1,tt),dailygrab)
        else:
            dailygrab = np.arange(tt-12,tt+12+1)
        tmp_stdev = np.nanstd(elev_input[dailygrab,:],axis=0)
        dum_profs = ~np.isnan(elev_input[dailygrab,:])
        tmp_num = sum(dum_profs)
        daily_zstdev[tt,proftt_notnan] = tmp_stdev[proftt_notnan]
        daily_znum[tt, proftt_notnan] = tmp_num[proftt_notnan]

    return daily_zstdev,daily_znum