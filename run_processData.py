import numpy as np
import datetime as dt
from funcs.run_lidarcollect import *
from funcs.run_hydrocollect import *
import pickle
from funcs.getFRF_funcs.downloadFRF import *
from funcs.align_data_time import align_data_fullspan
from funcs.wavefuncs import *



############ DEFINE WHERE FRF DATA FILES ARE LOCATED ############

local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'
server_base = 'https://chldata.erdc.dren.mil/thredds/catalog/frf/'
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'
time_pickle_path = picklefile_dir + 'timeinfo.pickle'


############ PROCESS THE DATA INTO WORKING DIR ############

frf_shoredir = 72

## Get lidar hydro & NOAA WL
noaawlfloc = local_base + 'waterlevel/'
noaawlext = '.nc'
lidarhydrofloc = local_base + '/waves_lidar/lidar_hydro/'
lidarhydroext = '.nc'
wlmax_lidar,wlmin_lidar,wltime_lidar,wlmean_lidar, wl_noaa, wltime_noaa = run_hydrocollect_func(noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext,time_pickle_path)

# Get 8m waves
wave8mfloc = local_base + 'waves_8marray/'
wave8mext = '.nc'
wave8m_time,wave8m_Tp,wave8m_Hs,wave8m_dir=run_wavecollect8m_func(wave8mfloc, wave8mext,time_pickle_path)
wave8m_dir = wave8m_dir - frf_shoredir
wave8m_dir[wave8m_dir > 180] = wave8m_dir[wave8m_dir > 180] - 360

# Get 17m waves
wave17mfloc = local_base + 'waves_17mwaverider/'
wave17mext = '.nc'
wave17m_time,wave17m_Tp,wave17m_Hs,wave17m_dir=run_wavecollect17m_func(wave17mfloc, wave17mext,time_pickle_path)
wave17m_dir = wave17m_dir - frf_shoredir
wave17m_dir[wave17m_dir > 180] = wave17m_dir[wave17m_dir > 180] - 360

# Get 26m waves
wave26mfloc = local_base + 'waves_26mwaverider/'
wave26mext = '.nc'
wave26m_time, wave26m_Tp, wave26m_Hs, wave26m_dir=run_wavecollect26m_func(wave26mfloc, wave26mext,time_pickle_path)
wave26m_dir = wave26m_dir - frf_shoredir
wave26m_dir[wave26m_dir > 180] = wave26m_dir[wave26m_dir > 180] - 360

# Get WIS waves
wisfloc = local_base + 'waves_WIS63218/'
wisext = '.nc'
waveWIS_time,waveWIS_Tp,waveWIS_Hs,waveWIS_dir=run_getWIS_func(wisfloc, wisext,time_pickle_path)
waveWIS_dir = waveWIS_dir - frf_shoredir
waveWIS_dir[waveWIS_dir > 180] = waveWIS_dir[waveWIS_dir > 180] - 360

# Get lidarWGs
lidarwgfloc = local_base + 'waves_lidar/lidar_wg080/'
lidarwgext = '.nc'
lidarwg080_xFRF,lidarwg080_time,lidarwg080_Tp,lidarwg080_TmIG,lidarwg080_Hs,lidarwg080_HsIN,lidarwg080_HsIG=run_lidarwavegauge_func(lidarwgfloc, lidarwgext,time_pickle_path)
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge090/'
lidarwgext = '.nc'
lidarwg090_xFRF,lidarwg090_time,lidarwg090_Tp,lidarwg090_TmIG,lidarwg090_Hs,lidarwg090_HsIN,lidarwg090_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext,time_pickle_path)
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge100/'
lidarwgext = '.nc'
lidarwg100_xFRF,lidarwg100_time,lidarwg100_Tp,lidarwg100_TmIG,lidarwg100_Hs,lidarwg100_HsIN,lidarwg100_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext,time_pickle_path)
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge110/'
lidarwgext = '.nc'
lidarwg110_xFRF,lidarwg110_time,lidarwg110_Tp,lidarwg110_TmIG,lidarwg110_Hs,lidarwg110_HsIN,lidarwg110_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext,time_pickle_path)
lidarwgfloc = local_base + '/waves_lidar/lidar_wavegauge140/'
lidarwgext = '.nc'
lidarwg140_xFRF,lidarwg140_time,lidarwg140_Tp,lidarwg140_TmIG,lidarwg140_Hs,lidarwg140_HsIN,lidarwg140_HsIG = run_lidarwavegauge_func(lidarwgfloc, lidarwgext,time_pickle_path)

# Get lidar runup
lidarrunupfloc = local_base + '/waves_lidar/lidar_waverunup/'
lidarrunupext = '.nc'
lidarrunup_time,lidarrunup_elev2p = run_lidarrunup_func(lidarrunupfloc, lidarrunupext,time_pickle_path)

# Get lidar topobathy
lidarfloc = local_base + '/dune_lidar/lidar_transect/'
lidarext = '.nc'
lidarelev,lidartime,lidar_xFRF,lidarelevstd,lidarmissing=run_lidarcollect(lidarfloc, lidarext,time_pickle_path)



############ TEMPORALLY ALIGN DATA ############

with open(time_pickle_path, 'rb') as file:
    tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = pickle.load(file)

dt = 3600
time_fullspan = np.arange(epoch_beg,epoch_end,dt)
wave17m_Tp_fullspan = align_data_fullspan(time_fullspan, wave17m_time, wave17m_Tp)
wave17m_Hs_fullspan = align_data_fullspan(time_fullspan, wave17m_time, wave17m_Hs)
wave17m_dir_fullspan = align_data_fullspan(time_fullspan, wave17m_time, wave17m_dir)
wave8m_Tp_fullspan = align_data_fullspan(time_fullspan, wave8m_time, wave8m_Tp)
wave8m_Hs_fullspan = align_data_fullspan(time_fullspan, wave8m_time, wave8m_Hs)
wave8m_dir_fullspan = align_data_fullspan(time_fullspan, wave8m_time, wave8m_dir)
wl_noaa_fullspan = align_data_fullspan(time_fullspan, wltime_noaa, wl_noaa)
lidarrunup_elev2p_fullspan = align_data_fullspan(time_fullspan, lidarrunup_time, lidarrunup_elev2p)

# expand lidarelev surface to full span of time...
lidarelevstd_fullspan = np.empty((lidar_xFRF.size,time_fullspan.size))*np.nan
lidarmissing_fullspan = np.empty((lidar_xFRF.size,time_fullspan.size))*np.nan
lidarelev_fullspan = np.empty((lidar_xFRF.size,time_fullspan.size))*np.nan
lidarhydro_max_fullspan = np.empty((lidar_xFRF.size,time_fullspan.size))*np.nan
lidarhydro_min_fullspan = np.empty((lidar_xFRF.size,time_fullspan.size))*np.nan
lidarhydro_mean_fullspan = np.empty((lidar_xFRF.size,time_fullspan.size))*np.nan
for ii in np.arange(lidar_xFRF.size):
    lidarelev_fullspan[ii,:] = align_data_fullspan(time_fullspan, lidartime, lidarelev[:,ii])
    lidarmissing_fullspan[ii, :] = align_data_fullspan(time_fullspan, lidartime, lidarmissing[:, ii])
    lidarelevstd_fullspan[ii, :] = align_data_fullspan(time_fullspan, lidartime, lidarelevstd[:, ii])
    lidarhydro_max_fullspan[ii,:] = align_data_fullspan(time_fullspan,wltime_lidar,wlmax_lidar[:,ii])
    lidarhydro_min_fullspan[ii, :] = align_data_fullspan(time_fullspan, wltime_lidar, wlmin_lidar[:, ii])
    lidarhydro_mean_fullspan[ii, :] = align_data_fullspan(time_fullspan, wltime_lidar, wlmean_lidar[:, ii])

# map lidar wave gauges to full time span...
wavegaugenames = ['lidarwg080','lidarwg090','lidarwg100','lidarwg110','lidarwg140']
varnames = ['Tp','TmIG','Hs','HsIG','HsIN']
for wg_ii in wavegaugenames:
    for var_jj in varnames:
        data_available = eval(wg_ii + '_' + var_jj)
        time_available = eval(wg_ii + '_time')
        data_fullspan = align_data_fullspan(time_fullspan, time_available, data_available)
        exec(wg_ii + '_' + var_jj + '_fullspan = data_fullspan')



############ RECOMBINE VARS TO SIMPLIFY ############

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
# For lidar wavegauges - [Hs, HsIN, HsIG, Tp, TmIG]
for wg_ii in wavegaugenames:
    exec('data_' + wg_ii + '= np.empty((time_fullspan.size,5))')
    exec('data_' + wg_ii + '[:] = np.nan')
    exec('data_' + wg_ii + '[:,0] = ' + wg_ii + '_Hs_fullspan.copy()')
    exec('data_' + wg_ii + '[:,1] = ' + wg_ii + '_HsIN_fullspan.copy()')
    exec('data_' + wg_ii + '[:,2] = ' + wg_ii + '_HsIG_fullspan.copy()')
    exec('data_' + wg_ii + '[:,3] = ' + wg_ii + '_Tp_fullspan.copy()')
    exec('data_' + wg_ii + '[:,4] = ' + wg_ii + '_TmIG_fullspan.copy()')

# reshape for ease of combining into single data matrix/frame
data_lidar_elev2p = lidarrunup_elev2p_fullspan.copy()
data_tidegauge = wl_noaa_fullspan.copy()
data_lidar_elev2p = np.reshape(data_lidar_elev2p,(time_fullspan.size,1))
data_tidegauge = np.reshape(data_tidegauge,(time_fullspan.size,1))

## Re-save 8m and 17m wave data, fill 8m from 17m where avail/needed
wave17m_Hs_fullspan = data_wave17m[:,0]
wave17m_Tp_fullspan = data_wave17m[:,1]
wave17m_dir_fullspan = data_wave17m[:,2]
wave8m_Hs_fullspan = data_wave8m[:,0]
wave8m_Tp_fullspan = data_wave8m[:,1]
wave8m_dir_fullspan = data_wave8m[:,2]

# Define 17m waves as H1
H1 = wave17m_Hs_fullspan[:]
T = wave17m_Tp_fullspan[:]
theta1 = wave17m_dir_fullspan[:]
h1 = 17
h2 = 8
breakcrit = 0.7
g = 9.81
# do transformation
H2, theta2 = wavetransform_point([],[], H1, theta1, T, h2, h1, g, breakcrit)
wave8m_gapfilled_Hs = np.empty(shape=time_fullspan.shape)
wave8m_gapfilled_Tp = np.empty(shape=time_fullspan.shape)
wave8m_gapfilled_dir = np.empty(shape=time_fullspan.shape)
wave8m_gapfilled_Hs[:] = np.nan
wave8m_gapfilled_Tp[:] = np.nan
wave8m_gapfilled_dir[:] = np.nan
wave8m_gapfilled_Hs[~np.isnan(wave8m_Hs_fullspan)] = wave8m_Hs_fullspan[~np.isnan(wave8m_Hs_fullspan)]
wave8m_gapfilled_Tp[~np.isnan(wave8m_Tp_fullspan)] = wave8m_Tp_fullspan[~np.isnan(wave8m_Tp_fullspan)]
wave8m_gapfilled_dir[~np.isnan(wave8m_dir_fullspan)] = wave8m_dir_fullspan[~np.isnan(wave8m_dir_fullspan)]
wave8m_gapfilled_Hs[np.isnan(wave8m_Hs_fullspan)] = H2[np.isnan(wave8m_Hs_fullspan)]
wave8m_gapfilled_Tp[np.isnan(wave8m_Tp_fullspan)] = T[np.isnan(wave8m_Tp_fullspan)]
wave8m_gapfilled_dir[np.isnan(wave8m_dir_fullspan)] = theta2[np.isnan(wave8m_dir_fullspan)]

data_wave8m = np.empty((time_fullspan.size,3))
data_wave8m[:] = np.nan
data_wave8m[:,0] = wave8m_Hs_fullspan
data_wave8m[:,1] = wave8m_Tp_fullspan
data_wave8m[:,2] = wave8m_dir_fullspan
data_wave17m = np.empty((time_fullspan.size,3))
data_wave17m[:] = np.nan
data_wave17m[:,0] = wave17m_Hs_fullspan
data_wave17m[:,1] = wave17m_Tp_fullspan
data_wave17m[:,2] = wave17m_dir_fullspan
data_wave8m_filled = np.empty((time_fullspan.size,3))
data_wave8m_filled[:] = np.nan
data_wave8m_filled[:,0] = wave8m_gapfilled_Hs
data_wave8m_filled[:,1] = wave8m_gapfilled_Tp
data_wave8m_filled[:,2] = wave8m_gapfilled_dir

## Create legacy values
xc_fullspan = []
dXcdt_fullspan = []



############ SAVE ALIGNED MATRICES ############

with open(picklefile_dir+'IO_alignedintime.pickle','wb') as file:
    pickle.dump([time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,xc_fullspan,dXcdt_fullspan,lidarelev_fullspan],file)
with open(picklefile_dir+'lidar_xFRF.pickle','wb') as file:
    pickle.dump(lidar_xFRF,file)
with open(picklefile_dir+'IO_lidarquality.pickle','wb') as file:
    pickle.dump([lidarelevstd_fullspan,lidarmissing_fullspan], file)
with open(picklefile_dir+'waves_8m&17m_2015_2024.pickle','wb') as file:
    pickle.dump([data_wave8m,data_wave17m,data_wave8m_filled], file)
with open(picklefile_dir+'IO_lidarhydro_aligned.pickle','wb') as file:
    pickle.dump([lidarhydro_min_fullspan,lidarhydro_max_fullspan,lidarhydro_mean_fullspan], file)













