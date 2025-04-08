#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:06:23 2024

@author: Nick
"""

import sys
import os
from getdatatestbed import getDataFRF
from prepdata.prepDataLib import PrepDataTools as preptools
from prepdata import inputOutput
from prepdata import writeRunRead as wrr
from testbedutils import fileHandling
import datetime
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4
import os
import plotly
import pickle
import yaml
import netCDF4 as nc
import numpy, scipy.io
from interpgap import interpolate_with_max_gap
from datetime import datetime, timedelta

#USER INPUTS
#sys.path.insert(0, '/Users/Nick/Documents/Models/cmtb/src/cmtb')
#os.chdir('/Users/Nick/Documents/Models/cmtb/src/cmtb')
d1 = datetime.datetime(2015,1,1,0,0,0)
d2 = datetime.datetime(2025,1,1,0,0,0)
dt = 3600. #seconds has to be a float and not an integer
profile_num = 960
dx = 1
fric_fac = 0.015
projectDir = '/Users/Nick/Documents/USACE/CMTB/CSHORE/GammaSims'
cshore_exe = '/Users/Nick/Documents/GitHub/cshore-master/src-repo/cshore.out'


#WAVE DATA DOWNLOAD
reltime = np.arange(0,ftime+dt,dt)
go = getDataFRF.getObs(d1-datetime.timedelta(hours=3),d2+datetime.timedelta(hours=3))
rawwave_data8m = go.getWaveData('8m-array',spec=False)
rawwave_data17m = go.getWaveData('waverider-17m',spec=False)
rawwave_data26m = go.getWaveData('waverider-26m',spec=False)
rawwave_data11m = go.getWaveData('awac-11m',spec=False)
rawwave_data4m = go.getWaveData('awac-4.5m',spec=False)
rawwave_data6m = go.getWaveData('awac-6m',spec=False)
rawwave_data300m = go.getWaveData('sig940-300',spec=False)
rawwave_data80m = go.getWaveData('lidarWaveGauge080',spec=False)
rawwave_data90m = go.getWaveData('lidarWaveGauge090',spec=False)
rawwave_data100m = go.getWaveData('lidarWaveGauge100',spec=False)
rawwave_data110m = go.getWaveData('lidarWaveGauge110',spec=False)
rawwave_data140m = go.getWaveData('lidarWaveGauge140',spec=False)



#INTERPPOLATE TIME SERIES
t = np.arange(d1, d2, timedelta(hours=1)).astype(datetime)
tepoch = np.zeros(np.shape(t))
for i in range(np.size(t)):
    tepoch[i] = datetime.timestamp(t[i])
maxtimegap = 1*60*60*1000
Hs8mInterp = interpolate_with_max_gap(rawwave_data8m['epochtime'], np.array(rawwave_data8m['Hs']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Hs17mInterp = interpolate_with_max_gap(rawwave_data17m['epochtime'], np.array(rawwave_data17m['Hs']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Hs26mInterp = interpolate_with_max_gap(rawwave_data26m['epochtime'], np.array(rawwave_data26m['Hs']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Tm8mInterp = interpolate_with_max_gap(rawwave_data8m['epochtime'], np.array(rawwave_data8m['Tm']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Tm17mInterp = interpolate_with_max_gap(rawwave_data17m['epochtime'], np.array(rawwave_data17m['Tm']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Tm26mInterp = interpolate_with_max_gap(rawwave_data26m['epochtime'], np.array(rawwave_data26m['Tm']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Dm8mInterp = interpolate_with_max_gap(rawwave_data8m['epochtime'], np.array(rawwave_data8m['waveDm']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Dm17mInterp = interpolate_with_max_gap(rawwave_data17m['epochtime'], np.array(rawwave_data17m['waveDm']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Dm26mInterp = interpolate_with_max_gap(rawwave_data26m['epochtime'], np.array(rawwave_data26m['waveDm']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
fp8mInterp = interpolate_with_max_gap(rawwave_data8m['epochtime'], np.array(rawwave_data8m['peakf']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
fp17mInterp = interpolate_with_max_gap(rawwave_data17m['epochtime'], np.array(rawwave_data17m['peakf']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
fp26mInterp = interpolate_with_max_gap(rawwave_data26m['epochtime'], np.array(rawwave_data26m['peakf']), tepoch, max_gap=maxtimegap, orig_x_is_sorted=False, target_x_is_sorted=False)
Hs_all = np.zeros(np.shape(t))
Tp_all = np.zeros(np.shape(t))
Dml_all = np.zeros(np.shape(t))
for i in range(np.size(t)):
    if np.isnan(Hs8mInterp[i]) == False:
        Hs_all[i] = Hs8mInterp[i]
        Tp_all[i] = 1/fp8mInterp[i]
        Dml_all[i] = np.abs(71-Dm8mInterp[i])
    elif np.isnan(Hs17mInterp[i]) == False:
        [Hshoal,Tshoal,ashoal] = shoaling(Hs17mInterp[i],1/fp17mInterp[i],np.abs(71-Dm17mInterp[i]),np.array(rawwave_data17m['depth']),np.array(rawwave_data8m['depth']))
        Hs_all[i] = Hshoal[i]
        Tp_all[i] = 1/fp8mInterp[i]
        Dml_all[i] = 71+ashoal
    elif np.isnan(Hs26mInterp[i]) == False:
        [Hshoal,Tshoal,ashoal] = shoaling(Hs26mInterp[i],1/fp26mInterp[i],np.abs(71-Dm26mInterp[i]),np.array(rawwave_data26m['depth']),np.array(rawwave_data8m['depth']))
        Hs_all[i] = Hshoal[i]
        Tp_all[i] = 1/fp8mInterp[i]
        Dml_all[i] = 71+ashoal


#PULL BATHY DATA
dataloc = ("https://chldata.erdc.dren.mil/thredds/dodsC/frf/geomorphology/elevationTransects/survey/surveyTransects.ncml")
ncfile = nc.Dataset(dataloc)
bathy_date= ncfile["date"][:]
bathy_y = ncfile["profileNumber"][:]
ifind = np.where((bathy_date>=[tepoch[0]-45*24*60*60*1000]) & (bathy_date<=[tepoch[-1]+45*24*60*60*1000]) & (bathy_y == profile_num))
bathy_times = bathy_date[ifind]    
bathy_elevation= ncfile["elevation"][ifind]
bathy_x = ncfile["xFRF"][ifind]
bathy_dates_unique = np.unique(bathy_times)      
offshore_x = 900
xinterp = np.arange(75, offshore_x+1, 1)
zInterpSurvey = np.zeros([len(bathy_dates_unique), len(xinterp)])*np.nan
for i in range(len(bathy_dates_unique)):
    ifind_data = np.where((bathy_times == bathy_dates_unique[i]))

    if len(ifind_data) > 1:
        z_data_temp = bathy_elevation[ifind_data]
        x_data_temp = bathy_x[ifind_data]
                   
        zInterp = interpolate_with_max_gap(x_data_temp, z_data_temp, xinterp, max_gap=5, orig_x_is_sorted=False, target_x_is_sorted=False)
        zInterpSurvey[i,:] = zInterp
    else: 
        zInterpSurvey[i,:] = zInterp

zInterpAll = np.zeros([len(t), len(xinterp)])*np.nan
for ix in range(len(xinterp)):  
    tempz = zInterpSurvey[:,ix]
    inonan = np.where(np.isnan(tempz) == False) 
    zInterpAll[ = np.interp((t, bathy_dates_unique[inonan], tempz[inonan])
    

#deal with bathy for each time step
bathy_loc = 'survey'
bathy_data = go.getBathyTransectFromNC(profilenumbers=960)
b_dict = prepdata.prep_CSHOREbathy(bathy_data, bathy_loc, dx, waves, profile_num=profile_num, fric_fac=fric_fac)
rawWL = go.getWL()
wl_data = prepdata.prep_WL(rawWL,wlTimeList)

gammas = np.arange(0.4, 1, 0.05)

xFRF = bathy_data["xFRF"]
z = bathy_data["elevation"]
isort = np.argsort(xFRF)

offshore_x = 900
xinterp = np.arange(75, offshore_x+dx, dx)
zinterp = np.interp(xinterp, xFRF[isort], z[isort])

xlocal = offshore_x - xinterp
isort = np.argsort(xlocal)
xgrid = xlocal[isort]
zgrid = zinterp[isort]

#batch run cshore
count = -1
for it in range(len(t)):
    
    val = np.array([0, 3600])
    BC_dict = {"timebc_wave": val}
    BC_dict["x"] = xgrid
    BC_dict["zb"] = zgrid
    BC_dict["fw"] = np.ones(np.shape(zgrid))*0.015
    BC_dict["Hs"] = [Hs_all[it], Hs_all[it]]
    BC_dict["Tp"] = [Tp_all[it], Tp_all[it]]
    BC_dict["angle"] = [Dml_all[it], Dml_all[it]]
    BC_dict["swlbc"] = [wl_data["avgWL"][it], wl_data["avgWL"][it]]
    BC_dict["Wsetup"] = [0, 0]
    BC_dict["salin"] = 30  # salin in ppt
    BC_dict["temp"] = 15  # water temp in degrees C    
    
    tstart = t[it]
    tend = t[it] + timedelta(hours=1)
    
            
    for ig in range(len(gammas)):
    
        count = count +1
        
        run_name = 'run' + str(count)
  
        cshoreio = wrr.cshoreio(workingDirectory=projectDir, testName = run_name, versionPrefix='base',
                                startTime=tstart, endTime=tend, runFlag=True,
                                generateFlag=True, readFlag=False)
 
        if not os.path.isdir(cshoreio.workingDirectory):
            os.makedirs(cshoreio.workingDirectory)    

        cshoreio.setCSHORE_time(dt)
        cshoreio.readCSHORE_nml(nmlname='namelists.nml')
        cshoreio.makeCSHORE_meta(b_dict,waves,wl_data)
        cshoreio.BC_dict = BC_dict

        #cshoreio.makeCSHORE_bc(b_dict,waves,wlDict=wl_data)
        
        
        
        cshoreio.infile['gamma'] = gammas[ig]
        cshoreio.infile['iprofl'] = 1
        cshoreio.meta_dict['version'] = 'fixed'
        cshoreio.writeCSHORE_infile()
        cshoreio.runSimulation(cshore_exe)

        cshoreio.readCSHORE_ODOC()
        cshoreio.readCSHORE_infile()
        cshoreio.readCSHORE_OBPROF()
        cshoreio.readCSHORE_OSETUP()
        cshoreio.readCSHORE_OXVELO()
        cshoreio.readCSHORE_OYVELO()
        cshoreio.readCSHORE_OCROSS()
        cshoreio.readCSHORE_OLONGS()
        cshoreio.readCSHORE_OBSUSL()
        
        cshoreio.gatherCSHORE_params()
        cshoreio.gatherCSHORE_bc()
        cshoreio.gatherCSHORE_hydro()
        cshoreio.gatherCSHORE_sed()
        cshoreio.gatherCSHORE_veg()
        cshoreio.gatherCSHORE_morpho()
        cshoreio.gatherCSHORE_meta()
        
        results = cshoreio._resultsToDict()
        
        outname = projectDir + '/cshore_results' + str(count) + '.p'
    
        with open(outname, 'wb') as fid:
            pickle.dump(results, fid, protocol=pickle.HIGHEST_PROTOCOL)
               
        scipy.io.savemat(projectDir + '/cshore_zb_results' + str(count) + '.mat', mdict={'zb': results['zb'], 'Hs': results['Hs'], 'x': results['x']})
