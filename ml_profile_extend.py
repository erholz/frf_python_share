import pickle
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load the dataframe
import os


# LOAD - processed data
fdir = 'C:/Users/rdchlerh/Desktop/FRF_data_backup/processed/2024Aug08_MLpreprocess/'

with open(fdir+'elev_processed_base.pickle', 'rb') as file:
    time_fullspan,lidar_xFRF,profile_width,maxgap_fullspan,xc_fullspan,dXcdt_fullspan = pickle.load(file)
with open(fdir+'elev_processed_slopes.pickle', 'rb') as file:
    avgslope_fullspan, avgslope_withinXCs, avgslope_beyondXCsea = pickle.load(file)
with open(fdir + 'elev_processed_slopes_shift.pickle', 'rb') as file:
    shift_avgslope,shift_avgslope_beyondXCsea = pickle.load(file)
with open(fdir + 'elev_processed_elev.pickle', 'rb') as file:
    zsmooth_fullspan,shift_zsmooth,unscaled_profile = pickle.load(file)
with open(fdir + 'elev_processed_elev&slopes_scaled.pickle', 'rb') as file:
    scaled_profiles, scaled_avgslope = pickle.load(file)

# How much data exist between end of contours and end of profiles?
