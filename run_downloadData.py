import os

import numpy as np
import datetime as dt
from funcs.run_lidarcollect import *
from funcs.run_hydrocollect import *
import pickle
from funcs.getFRF_funcs.downloadFRF import *


############ DEFINE WHERE FRF DATA FILES ARE LOCATED ############

local_base = 'C:/Users/rdchlerh/Desktop/FRF_data/'
server_base = 'https://chldata.erdc.dren.mil/thredds/catalog/frf/'


############ DEFINE TIME PERIOD OF INTEREST ############

time_beg = '2015-01-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss' (string), time of interest BEGIN
time_end = '2026-01-01T00:00:00'     # 'YYYY-MM-DDThh:mm:ss (string), time of interest END
tzinfo = dt.timezone(-dt.timedelta(hours=4))    # FRF = UTC-4
time_format = '%Y-%m-%dT%H:%M:%S'
epoch_beg = dt.datetime.strptime(time_beg,time_format).timestamp()
epoch_end = dt.datetime.strptime(time_end,time_format).timestamp()
TOI_duration = dt.datetime.fromtimestamp(epoch_end)-dt.datetime.fromtimestamp(epoch_beg)
picklefile_dir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/processed_to_share_02Apr2025/'
with open(picklefile_dir+'timeinfo.pickle','wb') as file:
    pickle.dump([tzinfo,time_format,time_beg,time_end,epoch_beg,epoch_end,TOI_duration], file)

years = [int(time_beg[0:4]),int(time_end[0:4])]


############ GET WAVE DATA ############

local_dir = local_base + 'waves_8marray/'
request_url = server_base + 'oceanography/waves/8m-array/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_17mwaverider/'
request_url = server_base + 'oceanography/waves/waverider_17m/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_26mwaverider/'
request_url = server_base + 'oceanography/waves/waverider_26m/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_WIS63218/'
request_url = 'https://chldata.erdc.dren.mil/thredds/catalog/wis/Atlantic/ST63218/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_lidar/lidar_wg080/'
request_url = server_base + 'oceanography/waves/lidarWaveGauge080/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_lidar/lidar_wg090/'
request_url = server_base + 'oceanography/waves/lidarWaveGauge090/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_lidar/lidar_wg100/'
request_url = server_base + 'oceanography/waves/lidarWaveGauge100/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_lidar/lidar_wg110/'
request_url = server_base + 'oceanography/waves/lidarWaveGauge110/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_lidar/lidar_wg140/'
request_url = server_base + 'oceanography/waves/lidarWaveGauge140/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waves_lidar/lidar_runup/'
request_url = server_base + 'oceanography/waves/lidarWaveRunup/'
downloadFRF_func(years,request_url,local_dir)


############ GET WATER LEVEL DATA ############

local_dir = local_base + 'waves_lidar/lidar_hydro/'
request_url = server_base + 'oceanography/waves/lidarHydrodynamics/'
downloadFRF_func(years,request_url,local_dir)

local_dir = local_base + 'waterlevel/'
request_url = server_base + 'oceanography/waterlevel/eopNoaaTide/'
downloadFRF_func(years,request_url,local_dir)


############ GET LIDAR PROFILES ############

local_dir = local_base + '/dune_lidar/lidar_transect/'
request_url = server_base + 'geomorphology/elevationTransects/duneLidarTransect/'
downloadFRF_func(years,request_url,local_dir)

