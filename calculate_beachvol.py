import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import scipy as sp
import datetime as dt
from create_contours import lidar_xFRF,lidarelev,cont_elev,cont_ts,lidartime,time_beg,time_end


# Ok, now calculate volume
zdatum = -0.5                # <<<< DEFINE BASE above which area is summed for a volume
beachVol = np.empty((len(cont_elev) ,len(lidartime)))
beachVol[:] = np.nan
beachVol_xc = np.empty((len(cont_elev) ,len(lidartime)))
beachVol_xc[:] = np.nan
DX = lidar_xFRF[2] - lidar_xFRF[1]  # spatial step [m]
for tt in np.arange(len(lidartime)):
    prof_tt = lidarelev[tt, :]
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

# Make some plots
fig, (ax1,ax2) = plt.subplots(2)
cmap = plt.cm.rainbow(np.linspace(0,1,cont_elev.size+1))
ax1.set_prop_cycle('color', cmap[1:-1,:])
tplot = pd.to_datetime(lidartime, unit='s', origin='unix')
for cc in np.arange(cont_elev.size-1):
    ax1.scatter(tplot, beachVol[cc, :], s=1, label='z = ' + str(cont_elev[cc]) + ' m')
# ax1.scatter(tplot,total_beachVol/total_obsBeachWid,s=1,color='k',label='Total Vol/Total obs. width')
ax1.grid(which='major',axis='both')
ax1.legend()
ax1.set_ylabel('Profile Vol [m^2]')
plt.suptitle(time_beg+' to '+time_end)
tmptime = lidartime[0:len(lidartime)-1] + DT/2
tplot = pd.to_datetime(tmptime, unit='s', origin='unix')
ax2.set_prop_cycle('color', cmap[1:-1,:])
for cc in np.arange(cont_elev.size-1):
    ax2.scatter(tplot, dBeachVol_dt[cc, :], s=1, label='z = ' + str(cont_elev[cc]) + ' m')
# ax2.scatter(tplot,total_dBeachVol_dt/total_obsBeachWid[1:len(lidartime)],s=1,color='k',label='Total dV/dt / Total obs. width')
ax2.set_xlabel('time')
ax2.set_ylabel('dV/dt [m^2/hr]')
ax2.grid(which='major',axis='both')
ax2.legend()
plt.gcf().autofmt_xdate()

# What is average x-shore slope across profile?

