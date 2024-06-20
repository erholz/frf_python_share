import numpy as np
from matplotlib import pyplot as plt

def align_data_fullspan(time_fullspan, time_available, data_available):

    # interp existing data to fullspan
    data_fullspan = np.interp(time_fullspan, time_available, data_available)

    # remove NaNs in the interp datasets
    delta_thresh = 3601                 # For FRF lidar data, standard sampling rate is hourly (3600)
    delta = time_available[1:] - time_available[0:-1]
    delta_large_ii = np.argwhere(delta > delta_thresh)
    delta_large = delta[delta > delta_thresh]
    for jj in np.arange(delta_large_ii.size):
        gap_beg = time_available[delta_large_ii[jj]]
        gap_end = time_available[delta_large_ii[jj]] + delta_large[jj]
        ingap_jj = (time_fullspan > gap_beg) & (time_fullspan < gap_end)
        data_fullspan[ingap_jj] = np.nan

    return data_fullspan




# # plot to verify correct nan-ing of subsampled data
# fig, ax = plt.subplots()
# plt.plot(wave17m_time,wave17m_Hs,'o',label='17m, original')
# plt.plot(time_fullspan,wave17m_Hs_fullspan,'+',label='17m, subsample')
# plt.plot(wave8m_time,wave8m_Hs,'o',label='8m')
# ax.grid(which='both', axis='both')