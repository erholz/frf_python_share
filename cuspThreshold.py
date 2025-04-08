import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import loadmat, savemat
from scipy.signal.windows import hamming
import scipy.fft as fft
from scipy.interpolate import interp1d
# Load data
data_dir = 'C:/Users/RDCHLDLA/Documents/cuspCodesForDylan/threshold/'
demeanedContour = loadmat(data_dir + 'demeanedContour.mat')['demeanedContour']
demeanedElevation = loadmat(data_dir + 'demeanedElevation.mat')['demeanedElevation']

# Parameters
amplitude_decrease = 0.7  # Decrease amplitude of cusps
alongshore_extent = 300  # Reduction in cusp length
threshold_name = 'C:/Users/RDCHLDLA/Documents/cuspCodesForDylan/threshold/thresholdPython.mat'


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



# Function to generate synthetic cusp fields

def make_synthetic_cusp_field(halfCusp):
    # Make a single full cusp by mirroring the half cusp
    cusp = np.concatenate((halfCusp, np.flip(halfCusp)))

    # Create a cusp field by repeating the single cusp until the length is at least 500
    cuspField = []
    while len(cuspField) < 500:
        cuspField = np.concatenate((cuspField, cusp))

    # Demean the cusp field
    cuspFieldDemeaned = cuspField - np.mean(cuspField)

    # Limit the output to 400 elements
    cuspFieldDemeaned = cuspFieldDemeaned[:400]

    return cuspFieldDemeaned

# Generate synthetic cusp fields
T = []
for i in range(30):
    start_indices = [266, 242, 192, 329, 154, 249, 226, 77, 210, 213, 324, 281, 198, 193, 90, 290, 287, 253, 327, 188,
                     295, 131, 227, 197, 310, 147, 356, 143, 285, 206]
    end_indices = [280, 254, 208, 339, 167, 265, 238, 87, 225, 219, 332, 293, 207, 208, 111, 298, 296, 259, 343, 199,
                   310, 152, 243, 206, 319, 162, 363, 162, 298, 218]
    T.append(make_synthetic_cusp_field(demeanedContour[start_indices[i]:end_indices[i], i % demeanedContour.shape[1]]))

T = np.array(T)

# Plot cusp fields
jet8 = plt.cm.jet(np.linspace(0, 1, T.shape[0]))
plt.figure()
for i in range(T.shape[0]):
    plt.plot(np.arange(1, 401), T[i, :], color=jet8[i], linewidth=1.5)
    plt.title(f'Cusp field {i + 1}')
    plt.pause(0.5)
    plt.clf()

# Generate spectra from idealized cusp fields
filter_large = 50
filter_small = 4
dof = 4
N = 512
delT = 1
j = np.arange(N)
fj = j / (N * delT)
fj_final = fj[:N // 2]
freq_final = fj_final[1:]

Sj = np.zeros((len(freq_final), T.shape[0]))
maxSj = np.zeros(T.shape[0])
idxMaxSj = np.zeros(T.shape[0], dtype=int)

for tp in range(T.shape[0]):
    contour_smSmall = T[tp, :]
    contour_smLarge = np.zeros(400)

    fj_final, Sj_final, Sj_ave, fj_ave = fft_cusps(contour_smSmall, contour_smLarge, delT, dof)

    Sj[:, tp] = Sj_final
    maxSj[tp] = np.max(Sj_final)
    idxMaxSj[tp] = np.argmax(Sj_final)

# Plot to check spectra
plt.figure()
for i in range(T.shape[0]):
    plt.plot(freq_final, Sj[:, i], color=jet8[i], linewidth=1.5)
    plt.grid(True)
    plt.title(f'Spectra {i + 1}')
    plt.pause(0.5)
    plt.clf()


# Fit maxima
def fit_func(kMax, a, b):
    return a * kMax ** b


popt, _ = curve_fit(fit_func, freq_final[idxMaxSj], maxSj,p0=[.1,0])#, bounds=([0, 0], [np.inf, np.max(freq_final)]))
maxSjFit = fit_func(freq_final, *popt)

plt.figure()
plt.plot(freq_final[idxMaxSj], maxSj, 'r.')
plt.plot(freq_final, maxSjFit, 'k-')
plt.xlim([0, 0.15])
plt.ylim([0, 600])

# Create modified cusp fields
TC = np.copy(T)
idx300All = np.zeros(TC.shape[0], dtype=int)

for tp in range(TC.shape[0]):
    idxDCs = np.zeros(TC.shape[1], dtype=int)
    for i in range(TC.shape[1] - 1):
        if (TC[tp, i + 1] < 0 and TC[tp, i] > 0) or (TC[tp, i + 1] > 0 and TC[tp, i] < 0):
            idxDCs[i] = 1

    idxDC = np.where(idxDCs == 1)[0]
    idx300 = np.argmin(np.abs(idxDC - alongshore_extent))
    TC[tp, idxDC[idx300] + 1:] = 0
    idx300All[tp] = idxDC[idx300]

TCC = TC * amplitude_decrease

# Generate spectra from modified cusp fields
SjC = np.zeros((len(freq_final), TCC.shape[0]))
maxSjC = np.zeros(TCC.shape[0])
idxMaxSjC = np.zeros(TCC.shape[0], dtype=int)

for tp in range(TCC.shape[0]):
    contour_smSmallC = TCC[tp, :]
    contour_smLargeC = np.zeros(400)

    fj_final, Sj_finalC, Sj_ave, fj_ave = fft_cusps(contour_smSmallC, contour_smLargeC, delT, dof)

    SjC[:, tp] = Sj_finalC
    maxSjC[tp] = np.max(Sj_finalC)
    idxMaxSjC[tp] = np.argmax(Sj_finalC)

kMaxC = freq_final[idxMaxSjC]

# Fit maxima for modified spectra
poptC, _ = curve_fit(fit_func, kMaxC, maxSjC,p0=[.1,0])# bounds=([0, 0], [np.inf, 0.15]))
maxSjFitC = fit_func(freq_final, *poptC)

plt.figure()
plt.plot(freq_final[idxMaxSj], maxSj, 'r.')
plt.plot(kMaxC, maxSjC, 'b.')
plt.plot(freq_final, maxSjFit, 'k-')
plt.plot(freq_final, maxSjFitC, 'k-')
plt.xlim([0, 0.15])
plt.ylim([0, 600])

# Save threshold
threshold = maxSjFitC
savemat(threshold_name, {'threshold': threshold})

# Make final figure
f3 = plt.figure()#, axs = plt.subplots(2, 2, figsize=(7, 7))
ax1 = plt.subplot2grid((2,2),(1,0),colspan=1,rowspan=1)
ax1.set_title('Synthetic spectra')
for i in range(T.shape[0]):
    ax1.plot(freq_final, Sj[:, i], color=jet8[i], linewidth=1.5)
ax1.plot(freq_final, maxSjFit, 'k-')
ax1.grid(True)
ax1.set_xlim([0, 0.1])
ax1.set_ylim([0, 1200])
ax1.set_xlabel('Wavenumber k')
ax1.set_ylabel('S(m^2 m)')

ax2 = plt.subplot2grid((2,2),(1,1),colspan=1,rowspan=1)

ax2.set_title('Modified synthetic spectra')
for i in range(TCC.shape[0]):
    ax2.plot(freq_final, SjC[:, i], color=jet8[i], linewidth=1.5)
ax2.plot(freq_final, maxSjFitC, 'k-')
ax2.grid(True)
ax2.set_xlim([0, 0.1])
ax2.set_ylim([0, 1200])
ax2.set_xlabel('Wavenumber k')
ax2.set_ylabel('S(m^2 m)')

ax0 = plt.subplot2grid((2,2),(0,0),colspan=2,rowspan=1)

ax0.set_title('Synthetic cusp field - cross-shore contour excursion')
ax0.plot(np.arange(1, 401), T[21, :], 'k', linewidth=1.5)
ax0.plot(np.arange(1, 401), TCC[21, :], 'r', linewidth=1.5)
ax0.grid(True)
ax0.set_xlim([0, 400])
ax0.set_ylim([-4, 4])
ax0.set_xlabel('Alongshore y (m)')
ax0.set_ylabel('Cross-shore x')

plt.show()