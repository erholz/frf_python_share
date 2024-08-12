import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mat73

# Set data directory
data_dir = r'/Users/dylananderson/Downloads/cuspCodesForDylan/'
# data_dir = r'C:/Users/RDCHLDLA/Documents/cuspCodesForDylan/'

# Load data
t1 = '20160310-0700'
t2 = '20160319-0300'
t3 = '20160707-0800'

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt

from scipy.signal.windows import hamming
import scipy.fft as fft
import numpy as np
from scipy.interpolate import interp1d

def extractContourFromDEM(ys, xs, DEM, contourL, filterSizeLarge, filterSizeSmall):
    """
    extractContourFromDEM
    Inputs:
        ys:             Alongshore vector
        xs:             Cross-shore vector
        DEM:            DEM data (2D numpy array)
        contourL:       Contour of interest (a specific elevation value)
        filterSizeLarge: Filter size (double moving average) to find trend of contour line
        filterSizeSmall: Filter size (double moving average) to remove irregularities in contour

    Outputs:
        contour_smSmall: Contour smoothed with a filterSizeSmall window length double moving average
        contour_smLarge: Contour smoothed with a filterSizeLarge window length double moving average
        elevation_smSmall: Elevation along the contour line smoothed with a filterSizeSmall window length double moving average
        elevation_smLarge: Elevation along the contour line smoothed with a filterSizeLarge window length double moving average
        noContour: 1 if more than 50 of 400 m along the contour line are nans
    """
    # Find the contour
    contours = plt.contour(xs, ys, DEM, levels=[contourL])
    plt.close()  # Close the plot to avoid displaying

    contourOut = [np.concatenate(x) for x in contours.allsegs][0]
    C_x, C_y = contourOut[:, 0], contourOut[:, 1]

    if len(C_x) != 0:
        # Remove points that aren't unique
        unique_y_indices = np.unique(C_y, return_index=True)[1]
        x_contour = np.array([C_x[i] for i in sorted(unique_y_indices)])
        y_contour = np.array([C_y[i] for i in sorted(unique_y_indices)])

        # Figure out how many alongshore locations have contours
        yC_rounded = np.round(y_contour)
        numYSLocs = np.array([np.sum(yC_rounded == y) for y in ys])

        if np.sum(numYSLocs != 0) > 350:
            # Interpolate onto a regular vector
            f_interp = interp1d(y_contour, x_contour, bounds_error=False, fill_value="extrapolate")
            contourFinal = f_interp(ys)

            if len(contourFinal) > 350:
                # Smooth contour line
                contour_smSmall = uniform_filter1d(contourFinal, size=filterSizeSmall, mode='nearest')
                contour_smLarge = uniform_filter1d(contourFinal, size=filterSizeLarge, mode='nearest')

                # Extract elevation at large contour
                xContourSmoothed_rounded = np.round(contour_smLarge).astype(int)
                elevation = np.array(
                    [DEM[yy, np.argmin(np.abs(xs - xContourSmoothed_rounded[yy]))] for yy in range(len(ys))])

                # Remove NaNs and interpolate
                valid_indices = ~np.isnan(elevation)
                ys_noNan = ys[valid_indices]
                elevation_noNan = elevation[valid_indices]
                f_interp_elev = interp1d(ys_noNan, elevation_noNan, bounds_error=False, fill_value="extrapolate")
                elevationFinal = f_interp_elev(ys)

                # Smooth elevation line
                elevation_smSmall = uniform_filter1d(elevationFinal, size=4, mode='nearest')
                elevation_smLarge = uniform_filter1d(elevationFinal, size=75, mode='nearest')

                noContour = 0
            else:
                contour_smSmall = np.nan
                contour_smLarge = np.nan
                elevation_smSmall = np.nan
                elevation_smLarge = np.nan
                noContour = 1
        else:
            contour_smSmall = np.nan
            contour_smLarge = np.nan
            elevation_smSmall = np.nan
            elevation_smLarge = np.nan
            noContour = 1
    else:
        contour_smSmall = np.nan
        contour_smLarge = np.nan
        elevation_smSmall = np.nan
        elevation_smLarge = np.nan
        noContour = 1

    return contour_smSmall, contour_smLarge, elevation_smSmall, elevation_smLarge, noContour





def interp1nan(x, y, xi):
    """
    Interpolates across NaNs.

    Parameters:
    - x: array-like, shape (n,): x data
    - y: array-like, shape (n,): y data
    - xi: array-like, shape (m,): xi values to interpolate to

    Returns:
    - yi: array-like, shape (m,): interpolated y values at each xi
    """
    if np.sum(~np.isnan(y)) > 1:
        interp_func = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], bounds_error=False, fill_value="extrapolate")
        yi = interp_func(xi)
    else:
        yi = np.full_like(xi, np.nan)

    return yi

# def interp1nan(x, y, x_new):
#     """Interpolate over NaNs"""
#     nans = np.isnan(y)
#     interp_func = interp1d(x[~nans], y[~nans], bounds_error=False, fill_value="extrapolate")
#     return interp_func(x_new)

def PSD_BandAve(Sj, fj, M):
    """
    Band averages Spectral values with non-overlapping band averages.

    Parameters:
    - Sj: array-like, shape (n,): Power Spectral Density
    - fj: array-like, shape (n,): Fourier frequencies that the PSD is calculated at
    - M: int: The number of degrees of freedom for band averaging (bin size will be M/2)

    Returns:
    - Sj_filt: array-like, shape (m,): Band-averaged Power Spectral Density
    - fj_filt: array-like, shape (m,): Band-averaged Fourier frequencies
    """

    N = len(Sj)  # Number of spectral estimates
    ave = M // 2  # Bin size (M/2)

    vec_end = np.arange(ave, N, ave)
    vec_beg = np.arange(1, N, ave)

    # Ensure the length of filtered outputs is determined by the smaller vector
    min_len = min(len(vec_end), len(vec_beg))
    Sj_filt = np.zeros(min_len)
    fj_filt = np.zeros(min_len)

    for ii in range(min_len):
        Sj_filt[ii] = np.mean(Sj[vec_beg[ii]:vec_end[ii]])
        fj_filt[ii] = np.mean(fj[vec_beg[ii]:vec_end[ii]])

    return Sj_filt, fj_filt

def fft_cusps(yn, yn_smoothed, delT, dof):
    """
    fft_cusps - Runs a Fourier transform over a contour line
    Inputs:
        yn:          Original contour line data
        yn_smoothed: Smoothed contour line data
        delT:        Grid resolution
        dof:         Degrees of freedom
    Outputs:
        fj_final: Final frequency vector
        Sj_final: Final power spectral density (PSD) (not band averaged)
        Sj_ave:   Band-averaged PSD
        freq_ave: Band-averaged frequency vector
    """

    if not np.isnan(np.nanmean(yn)):
        # Remove NaNs if there are any
        x = np.arange(len(yn))
        yn_smoothedNN = interp1nan(x, yn_smoothed, x)
        ynNN = interp1nan(x, yn, x)
        idxN1 = np.where(~np.isnan(ynNN))[0][0]
        idxNend = np.where(~np.isnan(ynNN))[0][-1]

        ynNN = ynNN[idxN1:idxNend + 1]
        yn_smoothedNN = yn_smoothedNN[idxN1:idxNend + 1]

        # Detrend
        yn_detrended = ynNN - yn_smoothedNN
        yn_demeaned = yn_detrended - np.mean(yn_detrended)

        # Window
        wn = hamming(len(yn_demeaned))
        yn_windowed = yn_demeaned * wn

        # Zero pad to 512
        yn_demeaned = np.pad(yn_demeaned, (0, 512 - len(yn_demeaned)), 'constant')
        yn_windowed = np.pad(yn_windowed, (0, 512 - len(yn_windowed)), 'constant')

        # FFT
        N = len(yn_windowed)
        delF = 1 / (N * delT)

        j = np.arange(N)
        fj = j / (N * delT)
        fn = 1 / (2 * delT)

        # Fourier transform
        Yj_original = (1 / N) * fft.fft(yn_demeaned)
        # Yj = (1 / N) * fft(yn_windowed, N)
        Yj = (1 / N) * fft.fft(yn_windowed)

        # Spectral density
        Sj = np.real(N * delT * Yj[:N // 2] * np.conj(Yj[:N // 2]))
        Sj_f = 2 * Sj
        Sj_f[0] /= 2
        fj_final = fj[:N // 2]

        # Boost the magnitudes of the PSD
        var_original = np.sum(np.abs(Yj_original) ** 2)

        var_windowed = np.sum(np.abs(Yj) ** 2)
        Sj_final = Sj_f * np.sqrt(var_original ** 2 / var_windowed ** 2)

        # Band average
        if dof == 2:
            Sj_ave = Sj_final
            freq_ave = fj_final
        else:
            Sj_ave, freq_ave = PSD_BandAve(Sj_final, fj_final, dof)

        # Remove 0 frequency
        Sj_final = Sj_final[1:]
        fj_final = fj_final[1:]

    else:
        fj_final = np.nan
        Sj_final = np.nan
        Sj_ave = np.nan
        freq_ave = np.nan

    return fj_final, Sj_final, Sj_ave, freq_ave


# def PSD_BandAve(Sj_final, fj_final, dof):
#     """ Band averages the power spectral density """
#     Sj_ave = []
#     freq_ave = []
#     band_width = len(Sj_final) // (dof // 2)
#     for i in range(0, len(Sj_final), band_width):
#         Sj_ave.append(np.mean(Sj_final[i:i + band_width]))
#         freq_ave.append(np.mean(fj_final[i:i + band_width]))
#
#     return np.array(Sj_ave), np.array(freq_ave)
def demeanedElevation(elevation_smSmall, elevation_smLarge, ys):
    """
    Demeans the small scale elevation by subtracting the large scale elevation
    and interpolates across missing values (NaNs).

    Parameters:
    - elevation_smSmall: array-like, shape (n,): Smoothed small scale elevation data
    - elevation_smLarge: array-like, shape (n,): Smoothed large scale elevation data
    - ys: array-like, shape (n,): Alongshore vector

    Returns:
    - elevationDemeaned_RG: array-like, shape (n,): Demeaned and interpolated elevation data
    """

    # Calculate the demeaned elevation
    elevationDemeaned = elevation_smSmall - elevation_smLarge

    if not np.isnan(np.nanmean(elevationDemeaned)):
        # Remove NaNs for interpolation
        valid_indices = ~np.isnan(elevationDemeaned)
        elevationNoNans = elevationDemeaned[valid_indices]
        yNoNans = ys[valid_indices]

        # Interpolate across NaNs
        interp_func = interp1d(yNoNans, elevationNoNans, bounds_error=False, fill_value="extrapolate")
        elevationDemeaned_RG = interp_func(ys)
    else:
        # Return an array of NaNs if elevationDemeaned is invalid
        elevationDemeaned_RG = np.full_like(ys, np.nan)

    return elevationDemeaned_RG

def findContoursRunFFT(ys, xs, DEM, contourLine, largeFilter, smallFilter, delT, dof, zOrX):
    """
    findContoursRunFFT - Takes the DEM, finds contours, and returns the spectral densities along those contours.

    Inputs:
        ys:             y vector
        xs:             x vector
        DEM:            DEM (Digital Elevation Model)
        contourLine:    Contour where we want to run FFT
        largeFilter:    Filter size (double moving average) to find the trend of contour line
        smallFilter:    Filter size (double moving average) to remove irregularities in contour line
        delT:           Grid resolution
        dof:            Degrees of freedom
        zOrX:           'X' for horizontal contour displacement, 'Z' for elevation change

    Outputs:
        contour_smSmall: Smoothed contour
        contour_smLarge: Contour trend
        elevation_smSmall: Elevation along contour trend
        elevation_smLarge: Elevation trend
        fj_final: Frequency vector
        Sj_final: PSD (not band averaged)
        fj_ave: Band-averaged frequency vector
        Sj_ave: Band-averaged PSD
        ED_RG: Interpolated, detrended contour line
    """

    # Extract contours from DEM
    contour_smSmall, contour_smLarge, elevation_smSmall, elevation_smLarge, _ = extractContourFromDEM(
        ys, xs, DEM, contourLine, largeFilter, smallFilter
    )

    if zOrX == 'Z':
        fj_final, Sj_final, Sj_ave, fj_ave = fft_cusps(elevation_smSmall, elevation_smLarge, delT, dof)
        ED_RG = demeanedElevation(elevation_smSmall, elevation_smLarge, ys)
    elif zOrX == 'X':
        fj_final, Sj_final, Sj_ave, fj_ave = fft_cusps(contour_smSmall, contour_smLarge, delT, dof)
        ED_RG = demeanedElevation(contour_smSmall, contour_smLarge, ys)
    else:
        raise ValueError("zOrX must be 'Z' or 'X'")

    return contour_smSmall, contour_smLarge, elevation_smSmall, elevation_smLarge, fj_final, Sj_final, Sj_ave, fj_ave, ED_RG












# Load threshold
threshold_data = sio.loadmat(data_dir + 'threshold/threshold.mat')
threshold = threshold_data['threshold'].flatten()  # Adjust depending on the structure of your .mat file

# Load example DEMs
dem1_data = mat73.loadmat(data_dir + 'exampleDEMs/' + t1 + '-01.FRFNProp.frame.data.mat')
DEM1 = dem1_data['frameGriddedData']['data']
xs = dem1_data['frameGriddedData']['xs']
ys = dem1_data['frameGriddedData']['as']

dem2_data = mat73.loadmat(data_dir + 'exampleDEMs/' + t2 + '-01.FRFNProp.frame.data.mat')
DEM2 = dem2_data['frameGriddedData']['data']

dem3_data = mat73.loadmat(data_dir + 'exampleDEMs/' + t3 + '-01.FRFNProp.frame.data.mat')
DEM3 = dem3_data['frameGriddedData']['data']

# Use only 400m of 500m DEM to focus on area with best data coverage
idx_y = np.arange(50, 451)  # MATLAB is 1-indexed, Python is 0-indexed
DEMC1 = DEM1[idx_y, :]
DEMC2 = DEM2[idx_y, :]
DEMC3 = DEM3[idx_y, :]
ysC = ys[idx_y]

# FFT parameters
filter_large = 50  # Moving average to find trend
filter_small = 4   # Smoothing contour
dof = 4
N = 512            # Zero-padded to N = 512
delT = 1           # DEM spatial resolution
contourOI = 1      # Contour of interest in meters

# Extract contour and run FFT
(contour_smSmall1, contour_smLarge1, elevation_smSmall1, elevation_smLarge1,
 fj_final1, Sj_final1, fj_ave1, Sj_ave1, ED_RG1) = findContoursRunFFT(ysC, xs, DEMC1, contourOI,
                                                                      filter_large, filter_small, delT, dof, 'X')



(contour_smSmall2, contour_smLarge2, elevation_smSmall2, elevation_smLarge2,
 fj_final2, Sj_final2, fj_ave2, Sj_ave2, ED_RG2) = findContoursRunFFT(ysC, xs, DEMC2, contourOI,
                                                                      filter_large, filter_small, delT, dof, 'X')

(contour_smSmall3, contour_smLarge3, elevation_smSmall3, elevation_smLarge3,
 fj_final3, Sj_final3, fj_ave3, Sj_ave3, ED_RG3) = findContoursRunFFT(ysC, xs, DEMC3, contourOI,
                                                                      filter_large, filter_small, delT, dof, 'X')

# Plot
plt.figure(figsize=(11, 8))

# DEM1 plot
plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=1)
plt.pcolor(xs, ysC, DEMC1)
plt.plot(contour_smSmall1, ysC, 'k-', linewidth=1.5)
plt.plot(contour_smLarge1, ysC, 'k--', linewidth=1)
plt.xlabel('Cross-shore x (m)')
plt.ylabel('Alongshore y (m)')
# plt.axis([50, 120, 750, 1150])
# plt.title(t1)

# DEM2 plot
plt.subplot2grid((3,3),(0,1),rowspan=2,colspan=1)
plt.pcolor(xs, ysC, DEMC2)
plt.plot(contour_smSmall2, ysC, 'k-', linewidth=1.5)
plt.plot(contour_smLarge2, ysC, 'k--', linewidth=1)
plt.xlabel('Cross-shore x (m)')
plt.ylabel('Alongshore y (m)')
# plt.axis([50, 120, 750, 1150])
# plt.title(t2)

# DEM3 plot
plt.subplot2grid((3,3),(0,2),rowspan=2,colspan=1)
plt.pcolor(xs, ysC, DEMC3)
plt.plot(contour_smSmall3, ysC, 'k-', linewidth=1.5)
plt.plot(contour_smLarge3, ysC, 'k--', linewidth=1)
plt.xlabel('Cross-shore x (m)')
plt.ylabel('Alongshore y (m)')
# plt.axis([50, 120, 750, 1150])
# plt.title(t3)

# Spectral density plot
plt.subplot2grid((3,3),(2,0),rowspan=1,colspan=3)
plt.plot(fj_final1, Sj_final1, linewidth=1.5)
plt.plot(fj_final2, Sj_final2, linewidth=1.5)
plt.plot(fj_final3, Sj_final3, linewidth=1.5)
plt.plot(fj_final1, threshold, 'k--')
plt.xlabel('Wavenumber (m^{-1})')
plt.ylabel('Spectral density (m^2 m)')
plt.xlim([0, 0.1])
plt.ylim([0, 1000])
# plt.legend([t1, t2, t3])

plt.tight_layout()
plt.show()