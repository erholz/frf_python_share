import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import time

def lidar_fillgaps(elev_input,lidartime,lidar_xFRF,halfspan_time,halfspan_x):
    timestartcode = time.time()

    lidar_gappy = np.copy(elev_input)
    lidar_filled = np.copy(elev_input)

    for tt in np.arange(len(lidartime)-1):
        if tt < halfspan_time:
            grabtt = np.arange(0,tt+1)
            numgrab = len(grabtt)
            grabtt = np.append(grabtt, np.arange(tt+1,tt+halfspan_time*2-numgrab+2))
        elif tt >= len(lidartime)-halfspan_time:
            grabtt = np.arange(tt,len(lidartime))
            numgrab = len(grabtt)
            grabtt = np.append(np.arange(tt-(halfspan_time*2-numgrab)-1,tt),grabtt)
        else:
            grabtt = np.arange(tt-halfspan_time,tt+halfspan_time+1)
        for ii in np.arange(len(lidar_xFRF)-1):
            if ii < halfspan_x:
                grabii = np.arange(0, ii + 1)
                numgrab = len(grabii)
                grabii = np.append(grabii, np.arange(ii + 1, ii + halfspan_x * 2 - numgrab + 2))
            elif ii >= len(lidar_xFRF) - halfspan_x:
                grabii = np.arange(ii, len(lidar_xFRF))
                numgrab = len(grabii)
                grabii = np.append(np.arange(ii - (halfspan_x * 2 - numgrab) - 1, ii), grabii)
            else:
                grabii = np.arange(ii - halfspan_x, ii + halfspan_x + 1)
            gappy_slice = lidar_gappy[grabtt,grabii[:,np.newaxis]]
            filled_slice = gappy_slice
            gappy_numel = gappy_slice.size
            gappy_numnotnan = sum(sum(~np.isnan(gappy_slice)))
            if (gappy_numnotnan/gappy_numel > 0.3) & (gappy_numnotnan/gappy_numel < 1):
                xv = lidar_xFRF[grabii]
                tv = lidartime[grabtt]
                [TT, XX] = np.meshgrid(tv, xv)
                # f = sp.interpolate.interp2d(tv, xv, gappy_slice)
                ij = ~np.isnan(gappy_slice)
                f = sp.interpolate.LinearNDInterpolator((TT[ij],XX[ij]),gappy_slice[ij])
                tnew = TT[np.isnan(gappy_slice)]
                xnew = XX[np.isnan(gappy_slice)]
                fnew = f((tnew,xnew))
                filled_slice[~ij] = fnew
            lidar_filled[grabtt, grabii[:, np.newaxis]] = filled_slice

    codeduration = time.time() - timestartcode
    print('Done!  lidar_fillgaps Duration = ' + str(codeduration) + ' seconds')

    return lidar_filled