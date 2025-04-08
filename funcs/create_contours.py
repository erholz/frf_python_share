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
        xtmp = lidar_xFRF
        ztmp = elev_input[tt,:]
        xcc = np.empty(shape=cont_elev.shape)
        xcc[:] = np.nan
        for cc in np.arange(cont_elev.size):
            if len(ztmp) > 0:
                if (np.nanmin(ztmp) <= cont_elev[cc]) & (np.nanmax(ztmp) >= cont_elev[cc]):
                    vartmp = abs(ztmp - cont_elev[cc]) == np.nanmin(abs(ztmp - cont_elev[cc]))
                    difftmp = (ztmp[0:-2]-cont_elev[cc])*(ztmp[1:-1]-cont_elev[cc])
                    iiclosest = np.where(difftmp < 0)[0]
                    if len(iiclosest) > 0:
                        if iiclosest.size > 1:
                            iiclosest = np.min(iiclosest)
                        if isinstance(iiclosest, np.ndarray):
                            iiclosest = iiclosest[0]
                        # iiclose = np.min(np.argwhere(vartmp)[0])
                        # iicloserng = np.arange(iiclose-3,iiclose+3)
                        # difftmp = (ztmp[iiclose-3:iiclose+3]-cont_elev[cc])*(ztmp[iiclose-2:iiclose+4]-cont_elev[cc])
                        # if (sum(np.isnan(difftmp)) > 0) or (sum(difftmp < 0) == 0):
                        #     iiclosest = iiclose
                        # else:
                        #     iiclosest = iicloserng[np.min(np.where(difftmp < 0))]
                        xcc[cc] = np.interp(0, np.flip(ztmp[iiclosest:iiclosest + 2] - cont_elev[cc]), np.flip(xtmp[iiclosest:iiclosest + 2]))
        cont_ts[:, tt] = xcc
    # Calculate mean and stddev of contour x-loc
    cmean = np.nanmean(cont_ts,axis=1)
    cstd = np.nanstd(cont_ts,axis=1)

    return cont_ts, cmean, cstd

