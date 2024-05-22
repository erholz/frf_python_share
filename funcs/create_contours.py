import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import scipy as sp
import datetime as dt

def create_contours(elev_input,lidartime,lidar_xFRF,cont_elev):
    cont_ts = np.empty((cont_elev.size,lidartime.size))
    for tt in np.arange(lidartime.size):
        xtmp = lidar_xFRF[~np.isnan(elev_input[tt,:])]
        ztmp = elev_input[tt,~np.isnan(elev_input[tt,:])]
        xcc = np.empty(shape=cont_elev.shape)
        xcc[:] = np.nan
        for cc in np.arange(cont_elev.size):
            if len(ztmp) > 0:
                if (min(ztmp) <= cont_elev[cc]) & (max(ztmp) >= cont_elev[cc]):
                    iiclose = int(np.argwhere(abs(ztmp - cont_elev[cc]) == min(abs(ztmp - cont_elev[cc]))))
                    xcc[cc] = np.interp(0,ztmp[iiclose-1:iiclose+2],xtmp[iiclose-1:iiclose+2])
        cont_ts[:, tt] = xcc

    # Calculate mean and stddev of contour x-loc
    cmean = np.nanmean(cont_ts,axis=1)
    cstd = np.nanstd(cont_ts,axis=1)

    return cont_ts, cmean, cstd

