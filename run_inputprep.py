from run_hydrocollect import *
from funcs.get_timeinfo import *
from run_lidarcollect import *
from funcs.create_contours import *
from matplotlib import pyplot as plt
import numpy as np
import os
from funcs.align_data_time import align_data_fullspan

# define constants/file directories
local_base, lidarfloc, lidarext, noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext = get_FileInfo()
frf_shoredir = 72

# Get water levels from lidar_hydro (min,max,mean) and NOAA tide gauge (on pier?)
noaawlfloc = local_base + '/waterlevel/eopNOAA/'
noaawlext = '.nc'
lidarhydrofloc = local_base + '/waves_lidar/lidar_hydro/'
lidarhydroext= '.nc'
wlmax_lidar,wlmin_lidar,wltime_lidar,wlmean_lidar, wl_noaa, wltime_noaa = run_hydrocollect_func(noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext)

# Get wave stats at 8m array
wave8mfloc = local_base + '/waves_8marray/'
wave8mext = '.nc'
wave8m_depth = 8.774985     # known constant, only nominal depth provided; total depth = nominal + SWE
wave8m_time,wave8m_Tp,wave8m_Hs,wave8m_dir = run_wavecollect8m_func(wave8mfloc, wave8mext)
wave8m_dir = wave8m_dir - frf_shoredir
wave8m_dir[wave8m_dir > 180] = wave8m_dir[wave8m_dir > 180] - 360

# Get wave stats at 17m buoy
wave17mfloc = local_base + '/waves_17mwaverider/'
wave17mext = '.nc'
wave17m_depth = 16.77       # known constant, only nominal depth provided; total depth = nominal + SWE
wave17m_time,wave17m_Tp,wave17m_Hs,wave17m_dir = run_wavecollect17m_func(wave17mfloc, wave17mext)
wave17m_dir = wave17m_dir - frf_shoredir
wave17m_dir[wave17m_dir > 180] = wave17m_dir[wave17m_dir > 180] - 360

# Get wave stats from virtual/lidar wave gauges
lidarwgext = '.nc'
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge080/'
lidarwg080_xFRF,lidarwg080_time,lidarwg080_Tp,lidarwg080_TmIG,lidarwg080_Hs,lidarwg080_HsIN,lidarwg080_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext)
lidarwgext = '.nc'
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge090/'
lidarwg090_xFRF,lidarwg090_time,lidarwg090_Tp,lidarwg090_TmIG,lidarwg090_Hs,lidarwg090_HsIN,lidarwg090_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext)
lidarwgext = '.nc'
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge100/'
lidarwg100_xFRF,lidarwg100_time,lidarwg100_Tp,lidarwg100_TmIG,lidarwg100_Hs,lidarwg100_HsIN,lidarwg100_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext)
lidarwgext = '.nc'
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge110/'
lidarwg110_xFRF,lidarwg110_time,lidarwg110_Tp,lidarwg110_TmIG,lidarwg110_Hs,lidarwg110_HsIN,lidarwg110_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext)
lidarwgext = '.nc'
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge140/'
lidarwg140_xFRF,lidarwg140_time,lidarwg140_Tp,lidarwg140_TmIG,lidarwg140_Hs,lidarwg140_HsIN,lidarwg140_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext)

# Get lidar runup
lidarrunupext = '.nc'
lidarrunupfloc = local_base + '/waves_lidar/lidar_waverunup/'
lidarrunup_time,lidarrunup_elev2p = run_lidarrunup_func(lidarrunupfloc, lidarrunupext)

# Get topobathy
lidarrunupfloc = local_base + '/dune_lidar/lidar_transect/'
lidarrunupext = '.nc'
lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing = run_lidarcollect(lidarfloc, lidarext)

# Find contour
elev_input = lidarelev
cont_elev = np.arange(0,2.5,0.5)    # <<< MUST BE POSITIVELY INCREASING
cont_ts, cmean, cstd = create_contours(elev_input,lidartime,lidar_xFRF,cont_elev)

# Define output as DeltaX_contour(i) = X_contour(i+1) - X_contour(i)
dXContdt = cont_ts[:,1:] - cont_ts[:,0:-1]

# Make histograms of input data
param_list = ['wave8m_dir','wave17m_dir','wave8m_Tp','wave17m_Tp','wave8m_Hs',
              'wave17m_Hs','lidarrunup_elev2p','wl_noaa']
count = 0
for param_ii in param_list:
    fig, ax = plt.subplots()
    var = eval(param_ii)
    if count < 2:
        binlocs = np.arange(np.nanmin(var), np.nanmax(var), 2)
    elif (count >= 2) & (count < 4):
        binlocs = np.arange(np.nanmin(var), np.nanmax(var), 0.1)
    else:
        binlocs = np.arange(np.nanmin(var),np.nanmax(var),0.1)
    plt.hist(var, bins=binlocs, density=True, stacked=True, rwidth=1)
    plt.title(param_ii + ', N = ' + str(sum(~np.isnan(var))))
    plt.xlabel('value')
    plt.ylabel('density')
    count = count + 1
    imagedir = './figs/data/allavailabledata/'
    imagename = imagedir + param_ii + '_pdf.svg'
    if os.path.exists(imagedir) == False:
        os.makedirs(imagedir)
    fig.savefig(imagename, format='svg', dpi=1200)
# plot and save distributions of lidar_wg data
wavegaugenames = ['lidarwg080','lidarwg090','lidarwg100','lidarwg110','lidarwg140']
lblstrs = ['80m','90m','100m','110m','140m']
varnames = ['Tp','TmIG','Hs','HsIG','HsIN']
binlocs1 = np.arange(3.5,15,0.05)
binlocs2 = np.arange(30,60,0.5)
binlocs3 = np.arange(0,2.5,0.05)
binlocs4 = np.arange(0,1.1,0.05)
binlocs5 = np.arange(0,2.1,0.05)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
wgcount = 0
for wg_ii in wavegaugenames:
    varcount = 1
    for var_jj in varnames:
        var = eval(wg_ii + '_' + var_jj)
        lblstrjj = lblstrs[wgcount] + ', N = ' + str(sum(~np.isnan(var)))
        eval('ax' + str(varcount) + '.hist(var, bins=binlocs' + str(varcount) + ', density=True, stacked=True, alpha=0.4, label=lblstrjj)')
        if wgcount == 4:
            eval('ax' + str(varcount) + '.legend()')
            eval('ax' + str(varcount) + '.set_title(var_jj)')
            xlblstr = 'value'
            ylblstr = 'density'
            eval('ax' + str(varcount) + '.set_xlabel(xlblstr)')
            eval('ax' + str(varcount) + '.set_ylabel(ylblstr)')
            imagedir = './figs/data/allavailabledata/'
            imagename = imagedir + 'lidarwg_' + var_jj + '_pdf.svg'
            imageformat = 'svg'
            eval('fig'+str(varcount)+'.savefig(imagename, format=imageformat, dpi=1200)')
        varcount = varcount+1
    wgcount = wgcount + 1
# plot and save distributions of contour position (Xc)
fig, ax = plt.subplots()
for ii in np.arange(cont_elev.size):
    var = cont_ts[ii,:]
    binlocs = np.arange(np.nanmin(var), 150, 0.5)
    lblstr = 'Zc = ' + str(cont_elev[ii]) + ', N = ' + str(sum(~np.isnan(var)))
    plt.hist(var, bins=binlocs, density=True, stacked=True, rwidth=1, alpha=0.4, label=lblstr)
plt.title('Xc [m, FRF]')
plt.xlabel('value')
plt.ylabel('density')
plt.legend()
imagedir = './figs/data/allavailabledata/'
imagename = imagedir + 'Xc' + '_pdf.svg'
fig.savefig(imagename, format='svg', dpi=1200)
# plot and save distributions of contour gradients (DXc/DT)
fig, ax = plt.subplots()
for ii in np.arange(cont_elev.size):
    var = 100*dXContdt[ii,:]
    binlocs = np.arange(-50, 50, 2.5)
    # binlocs = np.linspace(-50,50,200)
    lblstr = 'Zc = ' + str(cont_elev[ii]) + ', N = ' + str(sum(~np.isnan(var)))
    plt.hist(var, bins=binlocs, density=True, stacked=True, alpha=0.4, label=lblstr)
    # plt.hist(var, density=True, stacked=True, alpha=0.4, label=lblstr)
plt.title('DXc/Dt [cm/hr]')
plt.xlabel('value')
plt.ylabel('density')
plt.legend()
imagedir = './figs/data/allavailabledata/'
imagename = imagedir + 'DXcDt' + '_pdf.svg'
fig.savefig(imagename, format='svg', dpi=1200)

# What is temporal alignment of datasets
param_list = ['lidartime','lidarrunup_time','lidarwg080_time',
              'lidarwg090_time','lidarwg100_time','lidarwg110_time',
              'lidarwg140_time','wave17m_time','wave8m_time']
fig, ax = plt.subplots()
for ii in np.arange(len(param_list)):
    tplot = pd.to_datetime(eval(param_list[ii]), unit='s', origin='unix')
    yplot = ii * np.ones(shape=tplot.shape)
    lblstr = param_list[ii][0:-4].replace('_','')
    plt.plot(tplot,yplot,'o', label=lblstr)
plt.gcf().autofmt_xdate()
plt.xlabel('time')
plt.ylabel('variable')
plt.legend()
ax.grid(which='major', axis='both')
imagedir = './figs/data/allavailabledata/'
imagename = imagedir + 'data_availability_times.svg'
fig.savefig(imagename, format='svg', dpi=1200)

# Interp all data to full time series with identical sampling times
dt = 3600
time_fullspan = np.arange(np.nanmin(wave8m_time),np.nanmax(lidartime),dt)
wave17m_Tp_fullspan = align_data_fullspan(time_fullspan, wave17m_time, wave17m_Tp)
wave17m_Hs_fullspan = align_data_fullspan(time_fullspan, wave17m_time, wave17m_Hs)
wave17m_dir_fullspan = align_data_fullspan(time_fullspan, wave17m_time, wave17m_dir)
wave8m_Tp_fullspan = align_data_fullspan(time_fullspan, wave8m_time, wave8m_Tp)
wave8m_Hs_fullspan = align_data_fullspan(time_fullspan, wave8m_time, wave8m_Hs)
wave8m_dir_fullspan = align_data_fullspan(time_fullspan, wave8m_time, wave8m_dir)
wl_noaa_fullspan = align_data_fullspan(time_fullspan, wltime_noaa, wl_noaa)
lidarrunup_elev2p_fullspan = align_data_fullspan(time_fullspan, lidarrunup_time, lidarrunup_elev2p)
# Xc and DXc/Dt time series
cont_ts_fullspan = np.empty(shape=(cont_elev.size,time_fullspan.size))
dXContdt_fullspan = np.empty(shape=(cont_elev.size,time_fullspan.size))
cont_ts_fullspan[:] = np.nan
dXContdt_fullspan[:] = np.nan
for ii in np.arange(cont_elev.size):
    time_available = lidartime
    data_fullspan = align_data_fullspan(time_fullspan, time_available, cont_ts[ii,:])
    cont_ts_fullspan[ii,:] = data_fullspan
    data_fullspan = align_data_fullspan(time_fullspan, time_available[0:-1], dXContdt[ii, :])
    dXContdt_fullspan[ii, :] = data_fullspan
# lidar wave gauges...
wavegaugenames = ['lidarwg080','lidarwg090','lidarwg100','lidarwg110','lidarwg140']
varnames = ['Tp','TmIG','Hs','HsIG','HsIN']
for wg_ii in wavegaugenames:
    for var_jj in varnames:
        data_available = eval(wg_ii + '_' + var_jj)
        time_available = eval(wg_ii + '_time')
        data_fullspan = align_data_fullspan(time_fullspan, time_available, data_available)
        exec(wg_ii + '_' + var_jj + '_fullspan = data_fullspan')


# Find times when all data is available...
#
# FOR 8m and 17m WAVES - [Hs, Tp, dir]
data_wave8m = np.empty((time_fullspan.size,3))
data_wave8m[:] = np.nan
data_wave8m[:,0] = wave8m_Hs_fullspan.copy()
data_wave8m[:,1] = wave8m_Tp_fullspan.copy()
data_wave8m[:,2] = wave8m_dir_fullspan.copy()
data_wave17m = np.empty((time_fullspan.size,3))
data_wave17m[:] = np.nan
data_wave17m[:,0] = wave17m_Hs_fullspan.copy()
data_wave17m[:,1] = wave17m_Tp_fullspan.copy()
data_wave17m[:,2] = wave17m_dir_fullspan.copy()
data_lidar_elev2p = lidarrunup_elev2p_fullspan.copy()
data_tidegauge = wl_noaa_fullspan.copy()
# For lidar wavegauges - [Hs, HsIN, HsIG, Tp, TmIG]
for wg_ii in wavegaugenames:
    exec('data_' + wg_ii + '= np.empty((time_fullspan.size,5))')
    exec('data_' + wg_ii + '[:] = np.nan')
    exec('data_' + wg_ii + '[:,0] = ' + wg_ii + '_Hs_fullspan.copy()')
    exec('data_' + wg_ii + '[:,1] = ' + wg_ii + '_HsIN_fullspan.copy()')
    exec('data_' + wg_ii + '[:,2] = ' + wg_ii + '_HsIG_fullspan.copy()')
    exec('data_' + wg_ii + '[:,3] = ' + wg_ii + '_Tp_fullspan.copy()')
    exec('data_' + wg_ii + '[:,4] = ' + wg_ii + '_TmIG_fullspan.copy()')
    tmp = eval('np.sum(np.isnan(data_' + wg_ii + '),axis=1)')
    exec('datavail_' + wg_ii + ' = np.ones(shape=time_fullspan.shape)')
    exec('datavail_' + wg_ii + '[tmp >= 1] = 0')
# Create vectors when instrument data not available (0) vs available (1)
tmp = np.sum(np.isnan(data_wave17m),axis=1)
datavail_wave17m = np.ones(shape=time_fullspan.shape)
datavail_wave17m[tmp >= 1] = 0
tmp = np.sum(np.isnan(data_wave8m),axis=1)
datavail_wave8m = np.ones(shape=time_fullspan.shape)
datavail_wave8m[tmp >= 1] = 0
datavail_tidegauge = np.ones(shape=time_fullspan.shape)
datavail_tidegauge[np.isnan(data_tidegauge)] = 0
datavail_lidar_elev2p = np.ones(shape=time_fullspan.shape)
datavail_lidar_elev2p[np.isnan(data_lidar_elev2p)] = 0
fig, ax = plt.subplots()
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
plt.plot(tplot,datavail_wave8m,'o',label='8m array, N = '+str(sum(datavail_wave8m)))
plt.plot(tplot,datavail_wave17m*2,'o',label='17m waverider, N = '+str(sum(datavail_wave17m)))
plt.plot(tplot,datavail_tidegauge*3,'o',label='NOAA tidegauge, N = '+str(sum(datavail_tidegauge)))
plt.plot(tplot,datavail_lidar_elev2p*4,'o',label='lidar runup, N = '+str(sum(datavail_lidar_elev2p)))
plt.plot(tplot,datavail_lidarwg080*5,'o',label='lidarwg x = 80m, N = '+str(sum(datavail_lidarwg080)))
plt.plot(tplot,datavail_lidarwg090*6,'o',label='lidarwg x = 90m, N = '+str(sum(datavail_lidarwg090)))
plt.plot(tplot,datavail_lidarwg100*7,'o',label='lidarwg x = 100m, N = '+str(sum(datavail_lidarwg100)))
plt.plot(tplot,datavail_lidarwg110*8,'o',label='lidarwg x = 110m, N = '+str(sum(datavail_lidarwg110)))
plt.plot(tplot,datavail_lidarwg140*9,'o',label='lidarwg x = 140m, N = '+str(sum(datavail_lidarwg140)))
tmp = datavail_wave8m + datavail_wave17m + datavail_tidegauge + datavail_lidar_elev2p + datavail_lidarwg090 + datavail_lidarwg100 + datavail_lidarwg110 + datavail_lidarwg140
datavail_all = np.ones(shape=time_fullspan.shape)
datavail_all[tmp > 1] = 0
plt.plot(tplot,datavail_all*10,'o',label='all, N = '+str(sum(datavail_all)))
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
imagedir = './figs/data/allavailabledata/'
imagename = imagedir + 'data_availability_timeoverlap.svg'
fig.savefig(imagename, format='svg', dpi=1200)


# Normalize output, simple method (remove mean, divide by standard deviatiion)
# *** NOTE!! this should technically be done *after* partitioning data into train-test sets

# Normalize input, simple method (remove mean, divide by standard deviatiion)
# *** NOTE!! this should technically be done *after* partitioning data into train-test sets

# Create arrays indicating data availability (1 = presence, 0 = absence)