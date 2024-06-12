import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def calculate_beachvol(elev_input,lidartime,lidar_xFRF,cont_elev,cont_ts):
    # Ok, now calculate volume
    zdatum = -0.5                # <<<< DEFINE BASE above which area is summed for a volume
    beachVol = np.empty((len(cont_elev) ,len(lidartime)))
    beachVol[:] = np.nan
    beachVol_xc = np.empty((len(cont_elev) ,len(lidartime)))
    beachVol_xc[:] = np.nan
    DX = lidar_xFRF[2] - lidar_xFRF[1]  # spatial step [m]
    for tt in np.arange(len(lidartime)):
        prof_tt = elev_input[tt, :]
        if len(prof_tt) > 0:
            for cc in np.arange(len(cont_elev)-1):
                # find portion of the beach between X_contour(cc) and X_contour(cc+1)
                xcc = cont_ts[cc,tt]
                xccp1 = cont_ts[cc+1,tt]
                iisect = (lidar_xFRF <= xcc) & (lidar_xFRF >= xccp1)
                # calculate area under profile for that section of beach
                xsect = lidar_xFRF[iisect]
                zsect = prof_tt[iisect] - zdatum
                if len(zsect) > 0:
                    beachVol[cc,tt] = sum(zsect)*DX
                    beachVol_xc[cc,tt] = np.nanmean(xsect)

    # Calculate change in Volume
    DT = (lidartime[2]-lidartime[1])/3600      # time step [hr]
    dBeachVol_dt = (beachVol[:,1:len(lidartime)] - beachVol[:,0:len(lidartime)-1]) / DT
    total_beachVol = np.nansum(beachVol, axis=0)
    total_dBeachVol_dt = (total_beachVol[1:len(lidartime)] - total_beachVol[0:len(lidartime)-1]) / DT
    total_obsBeachWid = np.nanmax(beachVol_xc,axis=0) - np.nanmin(beachVol_xc,axis=0)

    return beachVol, beachVol_xc, dBeachVol_dt, total_beachVol, total_dBeachVol_dt, total_obsBeachWid
